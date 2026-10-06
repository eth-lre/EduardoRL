"""Student-correctness metric: fraction of sampled final solutions graded correct.

The student sees the dialogue plus a final-attempt prompt and samples
``num_attempts`` solutions; the fraction correct is a smoother reward than
pass/fail. With ``omit_teacher_turns`` each teacher utterance is replaced by a
placeholder, so the teacher is credited only for what the student internalised.
With ``transfer_problem`` the final prompt poses an unseen near-transfer variant
(the system prompt keeps the original problem) and grading uses its answer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import openai

from ..utils import (
    call_openai_with_retries,
    grade_math_answer,
    load_prompt,
    strip_teacher_controls,
)

logger = logging.getLogger(__name__)

_STUDENT_SYSTEM_PROMPT = load_prompt("student_prompt")
_STUDENT_FINAL_PROMPT = load_prompt("student_final_prompt")
_STUDENT_FINAL_TRANSFER_PROMPT = load_prompt("student_final_transfer_prompt")

# Replaces teacher utterances when ``omit_teacher_turns`` is on: keeps turn structure, hides content.
_OMITTED_TEACHER_TURN = "(hidden)"


def _render_final_prompt(transfer_problem: str | None) -> str:
    """Final user turn: plain "now solve it" or the near-transfer variant.

    Uses ``str.replace`` since the prompt contains a literal ``\\boxed{...}``.
    """
    if not transfer_problem:
        return _STUDENT_FINAL_PROMPT
    return _STUDENT_FINAL_TRANSFER_PROMPT.replace(
        "{transfer_problem}", transfer_problem
    )


def _build_final_attempt_messages(
    messages: list[dict],
    problem: str,
    transfer_problem: str | None = None,
    omit_teacher_turns: bool = False,
) -> list[dict]:
    """Build the student-view chat history for a final-attempt call.

    Flips roles (teacher -> user, student -> assistant), strips teacher control
    tokens and thinking (or hides teacher turns if ``omit_teacher_turns``), and
    appends the final-attempt prompt. ``messages`` is not mutated.
    """
    out: list[dict] = [
        {"role": "system", "content": _STUDENT_SYSTEM_PROMPT.format(problem=problem)}
    ]
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "system":
            continue
        if role == "assistant":
            teacher_text = (
                _OMITTED_TEACHER_TURN
                if omit_teacher_turns
                else strip_teacher_controls(content)
            )
            out.append({"role": "user", "content": teacher_text})
        elif role == "user":
            out.append({"role": "assistant", "content": content})
    out.append({"role": "user", "content": _render_final_prompt(transfer_problem)})
    return out


async def _sample_once(
    client: openai.AsyncClient,
    model: str,
    msgs: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


async def compute_fraction(
    messages: list[dict],
    ground_truth: str,
    problem: str,
    client: openai.AsyncClient,
    model: str,
    num_attempts: int = 8,
    temperature: float = 0.6,
    max_tokens: int = 800,
    retry_kwargs: dict[str, Any] | None = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    transfer_problem: str | None = None,
    omit_teacher_turns: bool = False,
) -> tuple[float, int, int]:
    """Sample ``num_attempts`` parallel final solutions from the student.

    Returns ``(fraction_correct, n_incorrect_format, n_failed)``, where
    ``n_failed`` counts calls that exhausted their retries. When
    ``transfer_problem`` is set, ``ground_truth`` must be the variant's answer.
    ``semaphore`` is shared with all student calls in the worker to bound load
    on the student endpoint.
    """
    retry_kwargs = retry_kwargs or {}
    final_msgs = _build_final_attempt_messages(
        messages, problem, transfer_problem, omit_teacher_turns=omit_teacher_turns
    )

    async def one():
        return await call_openai_with_retries(
            _sample_once,
            client,
            model,
            final_msgs,
            temperature,
            max_tokens,
            label="student_final",
            use_fallback=True,
            fallback=None,
            semaphore=semaphore,
            **retry_kwargs,
        )

    texts = await asyncio.gather(*[one() for _ in range(num_attempts)])

    n_correct = 0
    n_format = 0
    n_failed = 0
    n_scored = 0
    for t in texts:
        if t is None:
            n_failed += 1
            continue
        n_scored += 1
        verdict = grade_math_answer(t, ground_truth)
        n_correct += int(verdict["acc"])
        n_format += int(verdict["incorrect_format"])

    frac = (n_correct / n_scored) if n_scored else 0.0
    return frac, n_format, n_failed
