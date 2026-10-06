"""Periodic-eval judge: whole-conversation 1-5 scores for pedagogy and correctness.

Diagnostic only (not part of the reward): reports the same metrics as the
offline benchmark in ``eval/run_middleturn_eval.py`` during veRL validation. The rubrics in
``configs/prompts/judge_eval_*.txt`` must stay identical to ``eval/prompts/``
for the numbers to be comparable. Unparseable scores return ``None`` so judge
failures are counted rather than averaged in.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

import openai

from ..utils import call_openai_with_retries, load_prompt
from .judge_quality import _render_conversation

logger = logging.getLogger(__name__)

CRITERIA = ("pedagogy", "correctness")


def _load_prompts() -> dict[str, str]:
    """Load the two rubrics, or {} if unreadable (disables the judge instead of failing)."""
    try:
        return {
            criterion: load_prompt(f"judge_eval_{criterion}") for criterion in CRITERIA
        }
    except OSError as exc:
        logger.warning("Eval judge disabled: cannot read the 1-5 rubrics (%s)", exc)
        return {}


_PROMPTS = _load_prompts()

_SCORE_RE = re.compile(r'"score"\s*:\s*"?\s*([1-5])\b')


def _parse_score(text: str) -> float | None:
    """Parse one criterion's judge JSON. ``None`` on any parse failure."""
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        start, end = cleaned.find("{"), cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            score = float(json.loads(cleaned[start:end], strict=False)["score"])
            if 1 <= score <= 5:
                return score
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, Exception):
        pass
    # Unescaped LaTeX in `reasoning` breaks json.loads; recover the score by regex.
    matches = _SCORE_RE.findall(text)
    if matches:
        logger.warning("Eval judge JSON malformed, recovered score via regex: %s", matches[-1])
        return float(matches[-1])
    logger.warning("Failed to parse eval judge response: %s", text[:300])
    return None


async def _judge_once(
    client: openai.AsyncClient,
    model: str,
    judge_prompt: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict[str, Any] | None,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body or {},
    )
    return resp.choices[0].message.content or ""


async def score_conversation(
    messages: list[dict],
    problem: str,
    answer: str,
    client: openai.AsyncClient,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 16384,
    extra_body: dict[str, Any] | None = None,
    retry_kwargs: dict[str, Any] | None = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> dict[str, float | None]:
    """Score one conversation on both criteria (one call each, in parallel, no voting).

    Returns ``{"pedagogy": 1-5 or None, "correctness": 1-5 or None}``; ``None``
    means the rubric was unavailable, the call failed, or parsing failed.
    ``messages`` should be the student-visible transcript (thinking stripped).
    """
    if not _PROMPTS:
        return dict.fromkeys(CRITERIA, None)
    retry_kwargs = retry_kwargs or {}
    conv_text = _render_conversation(messages)

    async def criterion(key: str) -> float | None:
        prompt = _PROMPTS[key].format(
            problem=problem, answer=answer, conversation=conv_text
        )
        raw = await call_openai_with_retries(
            _judge_once,
            client,
            model,
            prompt,
            temperature,
            max_tokens,
            extra_body,
            label="eval_judge",
            use_fallback=True,
            fallback=None,
            semaphore=semaphore,
            **retry_kwargs,
        )
        return _parse_score(raw) if raw is not None else None

    scores = await asyncio.gather(*(criterion(key) for key in CRITERIA))
    return dict(zip(CRITERIA, scores))
