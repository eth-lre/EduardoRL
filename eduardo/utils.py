"""Shared utilities for the Eduardo recipe: prompt loading, bounded-concurrency
retrying calls to the serving endpoints, math answer grading, and sanitising
teacher messages before they are shown to the student.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import signal
import threading
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import openai

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "configs" / "prompts"


# ── Per-process concurrency semaphores ───────────────────────────────────────
# Keyed by endpoint label (e.g. "student", "judge") so all call sites share one budget.
_CONCURRENCY_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
_CONCURRENCY_LOCK = threading.Lock()


def get_concurrency_semaphore(label: str, limit: int) -> Optional[asyncio.Semaphore]:
    """Return a process-local cached ``asyncio.Semaphore(limit)`` for ``label``.

    ``limit <= 0`` disables gating (returns ``None``). Each Ray actor is one
    process with one event loop, so caching gives all call sites on an endpoint
    a shared budget. The first ``limit`` seen for a label wins.
    """
    if limit is None or limit <= 0:
        return None
    with _CONCURRENCY_LOCK:
        existing = _CONCURRENCY_SEMAPHORES.get(label)
        if existing is None:
            _CONCURRENCY_SEMAPHORES[label] = asyncio.Semaphore(limit)
            logger.info(
                "eduardo: created concurrency semaphore label=%s limit=%d pid=%d",
                label,
                limit,
                os.getpid(),
            )
            return _CONCURRENCY_SEMAPHORES[label]
        return existing


def load_prompt(name: str) -> str:
    """Load ``configs/prompts/<name>.txt``."""
    path = _PROMPTS_DIR / f"{name}.txt"
    return path.read_text()


# ── Retry / fallback for vLLM serving ────────────────────────────────────────

TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    asyncio.TimeoutError,
)


def _is_context_overflow(exc: BaseException) -> bool:
    """Heuristic: does this 4xx look like a context-length overflow? (logging only)"""
    msg = str(exc).lower()
    return (
        "maximum context length" in msg
        or "context_length_exceeded" in msg
        or "input_tokens" in msg
        or "reduce the length of the input prompt" in msg
    )


async def call_openai_with_retries(
    fn: Callable[..., Awaitable[Any]],
    *args: Any,
    num_retries: int = 3,
    backoff_base: float = 2.0,
    backoff_cap: float = 30.0,
    jitter: bool = True,
    fallback: Any = None,
    use_fallback: bool = False,
    label: str = "serving",
    semaphore: Optional[asyncio.Semaphore] = None,
    **kwargs: Any,
) -> Any:
    """Call ``fn(*args, **kwargs)`` with bounded concurrency + exponential backoff.

    Transient failures (timeouts, 5xx, rate limits) are retried; other 4xx are
    not. The semaphore slot is held across backoff sleeps so retries do not
    re-saturate the endpoint. On exhausted retries or a non-retriable error,
    returns ``fallback`` if ``use_fallback`` else raises.
    """

    async def _run_with_retries() -> Any:
        last_exc: Optional[BaseException] = None
        for attempt in range(num_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except TRANSIENT_EXCEPTIONS as e:
                last_exc = e
                if attempt == num_retries:
                    break
                delay = min(backoff_base ** attempt, backoff_cap)
                if jitter:
                    delay *= random.uniform(0.5, 1.5)
                status = getattr(e, "status_code", None) or type(e).__name__
                logger.warning(
                    f"[{label}] transient failure ({status}), "
                    f"retry {attempt + 1}/{num_retries} in {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)
            except openai.APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status is not None and 500 <= status < 600:
                    last_exc = e
                    if attempt == num_retries:
                        break
                    delay = min(backoff_base ** attempt, backoff_cap)
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)
                    logger.warning(
                        f"[{label}] 5xx ({status}), "
                        f"retry {attempt + 1}/{num_retries} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue

                if use_fallback:
                    kind = (
                        "context_overflow"
                        if _is_context_overflow(e)
                        else f"http_{status}"
                    )
                    logger.error(
                        f"[{label}] non-retriable {status} ({kind}); using fallback "
                        f"to keep training alive. Error: {e}"
                    )
                    return fallback
                raise

        if use_fallback:
            logger.error(
                f"[{label}] gave up after {num_retries} retries, using fallback. "
                f"Last error: {last_exc}"
            )
            return fallback
        assert last_exc is not None
        raise last_exc

    if semaphore is None:
        return await _run_with_retries()
    async with semaphore:
        return await _run_with_retries()


# ── Math answer extraction and grading ───────────────────────────────────────
def last_boxed_only_string(string: str) -> Optional[str]:
    """Return the last ``\\boxed{…}`` substring (including the braces).

    Handles arbitrary nesting depth via brace counting; ``None`` if not found.
    """
    if not string:
        return None
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    depth = 0
    right_idx: Optional[int] = None
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right_idx = i
                break
        i += 1
    return string[idx : right_idx + 1] if right_idx is not None else None


def _remove_boxed(s: str) -> str:
    left = r"\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return ""


class _timeout:
    """Signal-based timeout (works only in the main thread of a process)."""

    def __init__(self, seconds: int = 5):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle)
        signal.alarm(self.seconds)

    def __exit__(self, *a):
        signal.alarm(0)

    @staticmethod
    def _handle(signum, frame):
        raise TimeoutError("math_verify timeout")


def grade_math_answer(solution_str: str, ground_truth: str) -> dict[str, Any]:
    """Grade the last ``\\boxed{}`` answer against ``ground_truth``.

    Returns ``{"score", "acc", "pred", "incorrect_format", "truncated"}``. Falls
    back to ``math_verify`` under a 5 s alarm when exact string match fails.
    """
    if not solution_str:
        return _grade_failure(pred="", incorrect_format=1)

    # Prefer a box near the end, where the final answer is.
    boxed = last_boxed_only_string(solution_str[-200:]) or last_boxed_only_string(solution_str)
    pred = _remove_boxed(boxed) if boxed else ""
    if pred == "":
        return _grade_failure(pred=pred, incorrect_format=1)

    correct = pred.strip() == ground_truth.strip()
    if not correct:
        try:
            with _timeout(seconds=5):
                from math_verify import parse as mv_parse, verify as mv_verify

                gold = mv_parse(ground_truth)
                guess = mv_parse(pred)
                correct = bool(mv_verify(gold, guess))
        except Exception:
            pass

    reward = 1.0 if correct else 0.0
    return {
        "score": reward,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 0,
        "truncated": 0,
    }


def _grade_failure(pred: str, incorrect_format: int) -> dict[str, Any]:
    return {
        "score": 0.0,
        "acc": 0.0,
        "pred": pred,
        "incorrect_format": incorrect_format,
        "truncated": 0,
    }


# ── Teacher-message sanitization for student visibility ──────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def strip_teacher_controls(text: str, assume_thinking: bool = True) -> str:
    """Remove ``<end_of_conversation>`` and private thinking from a teacher message.

    The chat template pre-fills ``<think>``, so content usually arrives as
    ``REASONING</think>REPLY``. Text before an orphan ``</think>`` or after an
    orphan ``<think>`` is dropped. With ``assume_thinking``, content without any
    think tag means thinking hit the token cap, so ``""`` is returned.
    """
    if not text:
        return text
    if assume_thinking and "</think>" not in text and "<think>" not in text:
        return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    text = text.replace("<end_of_conversation>", "")
    return text.strip()


# ── OpenAI async client factory (cached per process) ─────────────────────────
_CLIENT_CACHE: dict[str, openai.AsyncClient] = {}


def get_async_openai_client(
    api_base: str,
    api_key_env: str,
    timeout: float = 300.0,
) -> openai.AsyncClient:
    """Return a process-local AsyncClient cached per (api_base, api_key_env).

    The timeout is generous because, with in-flight requests capped, slow
    responses are usually long decodes rather than hangs.
    """
    key = f"{api_base}::{api_key_env}"
    if key not in _CLIENT_CACHE:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env} is not set; "
                f"cannot create client for {api_base}"
            )
        _CLIENT_CACHE[key] = openai.AsyncClient(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=0,
        )
    return _CLIENT_CACHE[key]
