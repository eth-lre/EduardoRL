"""LLM-judge metrics for the reward.

Two binary criteria, each judged separately with ``num_votes`` parallel calls
and majority-voted so one verdict cannot anchor the other:

    no_leak   (judge_leakage)   does the teacher leak answers or steps?
    teaching  (judge_quality)   free of factual errors and other disqualifying tutor errors?

Parse failures default to REJECT.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional
import re

import openai

from ..utils import call_openai_with_retries, load_prompt

logger = logging.getLogger(__name__)

# Templates take {problem}, {answer}, {conversation}; judges return {"reasoning", "decision"} JSON.
_JUDGE_PROMPTS = {
    "no_leak": load_prompt("judge_leakage"),
    "teaching": load_prompt("judge_quality"),
}

_DECISION_RE = re.compile(r'"decision"\s*:\s*"?\s*(OK|REJECT)\b', re.I)



def _render_conversation(messages: list[dict]) -> str:
    """Render the conversation as ``- Teacher: ...`` / ``- Student: ...`` lines."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "system":
            continue
        label = "Teacher" if role == "assistant" else "Student"
        lines.append(f"- {label}: {content}")
    return "\n".join(lines)


def _parse_judge_response(text: str) -> dict[str, str]:
    """Parse a single-criterion judge JSON into decision + reasoning.

    Expects bare JSON (judge thinking is disabled); defaults to REJECT on failure.
    """
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            decision = str(data.get("decision", "")).strip().upper()
            return {
                "decision": decision if decision in ("OK", "REJECT") else "REJECT",
                "reasoning": str(data.get("reasoning", "")),
            }
    except (json.JSONDecodeError, Exception):
        pass
    # Unescaped LaTeX in `reasoning` is invalid JSON; recover the verdict by regex.
    matches = _DECISION_RE.findall(text)
    if matches:
        logger.warning(
            f"Judge JSON malformed, recovered decision via regex: {matches[-1].upper()}"
        )
        return {
            "decision": matches[-1].upper(),
            "reasoning": "<unparsed JSON>",
        }
    logger.warning(f"Failed to parse judge response, defaulting to REJECT: {text[:300]}")
    return {
        "decision": "REJECT",
        "reasoning": "Parse failure",
    }


# Default vLLM request extras: judge thinking is disabled, so parsers expect bare
# JSON. Non-vLLM backends pass their own ``extra_body``.
_VLLM_NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


async def _judge_once(
    client: openai.AsyncClient,
    model: str,
    judge_prompt: str,
    temperature: float,
    max_tokens: int = 2048,
    extra_body: dict[str, Any] | None = None,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=_VLLM_NO_THINKING if extra_body is None else extra_body,
    )
    return resp.choices[0].message.content or ""


async def _vote_on_criterion(
    prompt: str,
    client: openai.AsyncClient,
    model: str,
    num_votes: int,
    temperature: float,
    max_tokens: int,
    retry_kwargs: dict[str, Any],
    semaphore: Optional[asyncio.Semaphore],
    extra_body: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Run one criterion's judge ``num_votes`` times in parallel and majority-vote.

    Returns (decision, joined_reasoning). A vote that exhausts retries counts as REJECT.
    """

    async def one():
        return await call_openai_with_retries(
            _judge_once,
            client,
            model,
            prompt,
            temperature,
            max_tokens,
            extra_body,
            label="judge",
            use_fallback=True,
            fallback=None,
            semaphore=semaphore,
            **retry_kwargs,
        )

    raw_responses = await asyncio.gather(*[one() for _ in range(num_votes)])

    votes: list[dict[str, str]] = []
    for resp in raw_responses:
        if resp is None:
            votes.append(
                {
                    "decision": "REJECT",
                    "reasoning": "Judge call failed after retries",
                }
            )
        else:
            votes.append(_parse_judge_response(resp))

    rejects = sum(1 for v in votes if v["decision"] == "REJECT")

    threshold = len(votes) // 2 + 1
    decision = "REJECT" if rejects >= threshold else "OK"

    reasoning = " | ".join(
        f"[Vote {i + 1}: {v['decision']}] {v['reasoning']}"
        for i, v in enumerate(votes)
    )
    return decision, reasoning


async def majority_vote(
    messages: list[dict],
    problem: str,
    answer: str,
    client: openai.AsyncClient,
    model: str,
    num_votes: int = 3,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    retry_kwargs: dict[str, Any] | None = None,
    semaphore: Optional[asyncio.Semaphore] = None,
    extra_body: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, str]]:
    """Judge both criteria independently, `num_votes` votes each.

    Returns (no_leak_decision, teaching_decision, reasoning_dict).
    ``semaphore`` bounds concurrent judge calls across rollouts in the worker;
    ``extra_body=None`` disables thinking via vLLM ``chat_template_kwargs``.
    """
    retry_kwargs = retry_kwargs or {}
    conv_text = _render_conversation(messages)

    async def criterion(key: str) -> tuple[str, str]:
        prompt = _JUDGE_PROMPTS[key].format(
            problem=problem, answer=answer, conversation=conv_text
        )
        return await _vote_on_criterion(
            prompt,
            client=client,
            model=model,
            num_votes=num_votes,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_kwargs=retry_kwargs,
            semaphore=semaphore,
            extra_body=extra_body,
        )

    (no_leak, no_leak_reasoning), (teaching, teaching_reasoning) = await asyncio.gather(
        criterion("no_leak"), criterion("teaching")
    )

    return no_leak, teaching, {
        "no_leak": no_leak_reasoning,
        "teaching": teaching_reasoning,
    }
