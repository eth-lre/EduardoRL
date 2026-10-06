"""Eduardo pedagogical evaluation pipeline.

Benchmarks a tutor model (HF hub id, local path, or API model) and reports the
pedagogical reward-model score (macro + micro average), following
PedagogicalRL (TutorRL) and mathtutorbench.

1. Rollout: ``bigmath`` runs full dialogues against a student simulator;
   ``mathdial``/``eedi`` generate the middle teacher turn of real human dialogs.
2. Scoring: every teacher turn (or only the generated one) is scored by the
   pedagogical RM; human reference turns are scored for a win rate.
3. Judge: an LLM judge scores each conversation 1-5 on pedagogy and factual
   correctness (``--gemini`` uses Gemini as the judge; ``--no-judge`` skips).

Usage::

    python -m eval.run_middleturn_eval --model eth-nlped/TutorRL-7B --num-problems 50
    python -m eval.run_middleturn_eval --model eth-nlped/TutorRL-7B --benchmark mathdial
    python -m eval.run_middleturn_eval --model openai/gpt-4o --use-openrouter --benchmark eedi
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eduardo.utils import (  # noqa: E402
    call_openai_with_retries,
    get_async_openai_client,
    get_concurrency_semaphore,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eduardo.eval")

_END_OF_CONVERSATION = "<end_of_conversation>"
# Same opening as process_dataset.py so eval matches the training distribution.
_STUDENT_OPENING_TEMPLATE = "Hi, can you help me understand this problem? {problem}"

# Eval-local prompt copies, independent of the training prompts.
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_TEACHER_SYSTEM_PROMPT = (_PROMPTS_DIR / "teacher_prompt.txt").read_text()
_STUDENT_SYSTEM_PROMPT = (_PROMPTS_DIR / "student_prompt.txt").read_text()
# One prompt per criterion: single-criterion JSON responses parse more reliably.
_JUDGE_PROMPTS = {
    "pedagogy": (_PROMPTS_DIR / "judge_eval_pedagogy.txt").read_text(),
    "correctness": (_PROMPTS_DIR / "judge_eval_correctness.txt").read_text(),
}

_DECISION_RE = re.compile(r'"score"\s*:\s*"?\s*([1-5])\b')

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


def split_thinking(text: str) -> tuple[str, str]:
    """Split a raw teacher message into ``(private reasoning, visible reply)``.

    Unlike the training version, tag-free content is treated as the visible
    reply (non-thinking models, API teachers). Handled forms:
      - balanced ``<think>…</think>`` blocks;
      - the Qwen3-style prefill ``REASONING</think>VISIBLE`` (orphan close,
        the opening tag lives in the prompt);
      - reasoning truncated by the per-turn token cap (orphan open → nothing
        after it is visible).
    ``<end_of_conversation>`` is dropped from the visible half.
    """
    if not text:
        return "", ""
    reasoning = [m.group(1) for m in _THINK_BLOCK_RE.finditer(text)]
    visible = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in visible:
        head, visible = visible.split("</think>", 1)
        reasoning.append(head.split("<think>", 1)[-1])
    if "<think>" in visible:
        visible, tail = visible.split("<think>", 1)
        reasoning.append(tail)
    return (
        "\n".join(p.strip() for p in reasoning if p.strip()),
        visible.replace(_END_OF_CONVERSATION, "").strip(),
    )


def strip_teacher_controls(text: str) -> str:
    """The student/judge/RM-visible half of a teacher message."""
    return split_thinking(text)[1]


def thinking_text(text: str) -> str:
    """The private-reasoning half of a teacher message. Empty when the model
    did not think (or the provider dropped the reasoning)."""
    return split_thinking(text)[0]



# --- Conversation container ---


class Conversation:
    """One teacher↔student rollout. ``messages`` uses neutral roles
    (``teacher`` / ``student``); perspective flipping + chat templating happen
    at generation time.
    """

    def __init__(self, problem: str, answer: str):
        self.problem = problem
        self.answer = answer
        self.messages: list[dict[str, str]] = [
            {
                "role": "student",
                "content": _STUDENT_OPENING_TEMPLATE.format(problem=problem),
            }
        ]
        self.done = False
        self.end_reason: str | None = None
        # One entry per teacher turn; source is "tokenizer", "api" or "estimate".
        self.thinking_tokens: list[int] = []
        self.thinking_tokens_source: str | None = None

    @property
    def num_teacher_turns(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "teacher")

    def finish(self, reason: str) -> None:
        self.done = True
        self.end_reason = reason

    def teacher_view(self) -> list[dict[str, str]]:
        """Teacher = assistant, student = user, plus the teacher system prompt."""
        out = [
            {
                "role": "system",
                "content": _TEACHER_SYSTEM_PROMPT.format(problem=self.problem),
            }
        ]
        for m in self.messages:
            role = "assistant" if m["role"] == "teacher" else "user"
            out.append({"role": role, "content": m["content"]})
        return out

    def student_view(self) -> list[dict[str, str]]:
        """Student = assistant, teacher = user (control tokens stripped)."""
        out = [
            {
                "role": "system",
                "content": _STUDENT_SYSTEM_PROMPT.format(problem=self.problem),
            }
        ]
        for m in self.messages:
            if m["role"] == "teacher":
                out.append({"role": "user", "content": strip_teacher_controls(m["content"])})
            else:
                out.append({"role": "assistant", "content": m["content"]})
        return out

    def scoring_messages(self) -> list[dict[str, str]]:
        """Transcript for RM scoring, teacher thinking/EoC tokens stripped
        (same as ``_hide_thinking`` in PedagogicalRL (TutorRL))."""
        return [
            {"role": m["role"], "content": strip_teacher_controls(m["content"])}
            for m in self.messages
        ]

    def to_record(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "answer": self.answer,
            "messages": self.scoring_messages(),
            "raw_messages": self.messages,
            "num_teacher_turns": self.num_teacher_turns,
            "thinking_tokens": self.thinking_tokens,
            "thinking_tokens_source": self.thinking_tokens_source,
            "end_reason": self.end_reason,
        }


# --- Data ---


def load_problems(dataset: str, split: str, num_problems: int, seed: int):
    import datasets

    ds = datasets.load_dataset(dataset, split=split)
    ds = ds.shuffle(seed=seed)
    if num_problems > 0:
        ds = ds.select(range(min(num_problems, len(ds))))
    logger.info("Loaded %d problems from %s[%s]", len(ds), dataset, split)
    return [(r["problem"], str(r["answer"])) for r in ds]


# --- Continuation benchmarks (mathdial / eedi): generate the middle teacher turn ---

# MathDial teacher turns are prefixed with an intent tag, e.g. "(focus)Okay…".
_MATHDIAL_INTENT_RE = re.compile(r"^\([a-z ]+\)\s*", flags=re.IGNORECASE)


def _merge_consecutive_roles(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge consecutive turns by the same speaker into one message."""
    merged: list[dict[str, str]] = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


def _middle_teacher_cut(
    messages: list[dict[str, str]]
) -> tuple[list[dict[str, str]], str] | None:
    """Split at the middle teacher turn: (context before it, reference turn)."""
    teacher_idxs = [i for i, m in enumerate(messages) if m["role"] == "teacher"]
    if not teacher_idxs:
        return None
    mid = teacher_idxs[len(teacher_idxs) // 2]
    return messages[:mid], messages[mid]["content"]


def _select_samples(samples: list[dict[str, Any]], num_problems: int, seed: int):
    import random

    random.Random(seed).shuffle(samples)
    if num_problems > 0:
        samples = samples[:num_problems]
    return samples


def load_mathdial_samples(split: str, num_problems: int, seed: int) -> list[dict[str, Any]]:
    """MathDial: the ``conversation`` column is ``|EOM|``-separated
    ``Speaker: text`` strings; teacher turns carry a leading intent tag.
    """
    import datasets

    ds = datasets.load_dataset("eth-nlped/mathdial", split=split)
    samples = []
    for row in ds:
        messages = []
        for part in row["conversation"].split("|EOM|"):
            speaker, sep, content = part.partition(":")
            if not sep:
                continue
            content = content.strip()
            if speaker.strip() == "Teacher":
                messages.append(
                    {"role": "teacher", "content": _MATHDIAL_INTENT_RE.sub("", content)}
                )
            else:
                messages.append({"role": "student", "content": content})
        cut = _middle_teacher_cut(_merge_consecutive_roles(messages))
        if cut is None:
            continue
        context, reference = cut
        samples.append(
            {
                "problem": row["question"],
                "answer": str(row["ground_truth"]),
                "context": context,
                "reference_teacher_turn": reference,
            }
        )
    samples = _select_samples(samples, num_problems, seed)
    logger.info("Loaded %d mathdial[%s] continuation samples", len(samples), split)
    return samples


def load_eedi_samples(split: str, num_problems: int, seed: int) -> list[dict[str, Any]]:
    """Eedi: one row per message (grouped by ``InterventionId``, ordered by
    ``MessageSequence``); consecutive turns by the same speaker are merged.
    Question text + answer options come from the ``dq-question-metadata``
    config (train split covers all questions); no correct-answer label exists,
    so ``answer`` stays empty. Image-only questions are skipped.
    """
    import datasets

    name = "Eedi/Question-Anchored-Tutoring-Dialogues-2k"
    dialogs = datasets.load_dataset(name, "anchored-dialogues", split=split)
    meta = datasets.load_dataset(name, "dq-question-metadata", split="train")

    # qid → {Label: [(Sequence, Text), …]}
    questions: dict[int, dict[str, list[tuple[int, str]]]] = {}
    for row in meta:
        questions.setdefault(row["QuestionId_DQ"], {}).setdefault(
            row["Label"], []
        ).append((row["Sequence"], row["Text"] or ""))

    def question_text(qid: int) -> str | None:
        labels = questions.get(qid, {})
        if "Question Text" not in labels:
            return None  # image-only question
        parts = ["".join(t for _, t in sorted(labels["Question Text"]))]
        for option in "ABCD":
            texts = labels.get(f"Answer {option} Text")
            if texts:
                parts.append(f"{option}) " + "".join(t for _, t in sorted(texts)))
        return "\n".join(parts)

    # InterventionId → ordered messages
    by_conv: dict[int, list[dict[str, Any]]] = {}
    for row in dialogs:
        by_conv.setdefault(row["InterventionId"], []).append(row)

    samples = []
    for conv_id in sorted(by_conv):
        rows = sorted(by_conv[conv_id], key=lambda r: r["MessageSequence"])
        problem = question_text(rows[0]["QuestionId_DQ"])
        if problem is None:
            continue
        messages = _merge_consecutive_roles(
            [
                {
                    "role": "teacher" if r["IsTutor"] == 1 else "student",
                    "content": (r["MessageString"] or "").strip(),
                }
                for r in rows
                if (r["MessageString"] or "").strip()
            ]
        )
        cut = _middle_teacher_cut(messages)
        if cut is None:
            continue
        context, reference = cut
        samples.append(
            {
                "problem": problem,
                "answer": "",
                "context": context,
                "reference_teacher_turn": reference,
            }
        )
    samples = _select_samples(samples, num_problems, seed)
    logger.info("Loaded %d eedi[%s] continuation samples", len(samples), split)
    return samples


async def continuation_rollout(args, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate the middle teacher turn for every sample in one batch.
    Returns records in the same shape as the bigmath rollout (so RM + judge
    scoring work unchanged), with the generated turn as the final message.
    """
    conversations = []
    for sample in samples:
        conv = Conversation(sample["problem"], sample["answer"])
        conv.messages = list(sample["context"])
        conversations.append(conv)

    teacher = make_teacher(args)
    texts = await teacher.generate(conversations)

    records = []
    for i, (sample, conv, text) in enumerate(zip(samples, conversations, texts)):
        conv.messages.append({"role": "teacher", "content": text})
        tokens, source = teacher.thinking_tokens(i, text)
        conv.thinking_tokens.append(tokens)
        conv.thinking_tokens_source = source
        conv.finish("continuation" if text else "teacher_failed")
        record = conv.to_record()
        record["benchmark"] = args.benchmark
        record["reference_teacher_turn"] = sample["reference_teacher_turn"]
        records.append(record)

    teacher.close()
    return records


# --- Rollout ---


def _render_teacher_prompts(tokenizer, conversations, enable_thinking: bool) -> list[str]:
    """Explicit apply_chat_template with a generation prompt for the teacher.
    ``enable_thinking`` is forwarded to the template (used by Qwen3-style
    templates, harmlessly ignored by others).
    """
    return [
        tokenizer.apply_chat_template(
            conv.teacher_view(),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for conv in conversations
    ]


def _verify_prompt_consistency(rendered_prompt: str, conv: Conversation) -> None:
    """Check that our system prompt survived chat templating, so every
    benchmarked model sees the identical prompt.
    """
    expected_system = _TEACHER_SYSTEM_PROMPT.format(problem=conv.problem)
    if expected_system not in rendered_prompt:
        logger.warning(
            "PROMPT CONSISTENCY: the rendered prompt does NOT contain our teacher "
            "system prompt — this model's chat template may override or drop system "
            "messages, making results incomparable across models."
        )
    logger.info(
        "First rendered teacher prompt (verify the system prompt is ours):\n%s",
        rendered_prompt,
    )


# --enable-thinking configures the tutor only; --gemini* flags configure the judge.


def teacher_thinking_extra_body(args) -> dict[str, Any]:
    """Translate ``--enable-thinking`` into the teacher provider's parameter
    (none for Gemini, which always uses its default thinking level).
    """
    if args.use_gemini:
        return {}
    if "openrouter.ai" in args.teacher_api_base:
        return {"reasoning": {"enabled": args.enable_thinking}}
    return {"chat_template_kwargs": {"enable_thinking": args.enable_thinking}}


def describe_teacher_thinking(args) -> str:
    """One-line description of the tutor's thinking configuration."""
    if args.use_gemini:
        return "gemini tutor — API default thinking level (--enable-thinking does not apply)"
    if args.use_openrouter:
        return (
            f"enable_thinking={args.enable_thinking} (api tutor, requested as "
            f"{json.dumps(teacher_thinking_extra_body(args))})"
        )
    return f"enable_thinking={args.enable_thinking} (local vLLM tutor chat template)"


_logged_thinking_sample = False


def _log_first_thinking_sample(texts: list[str]) -> None:
    """Log once whether the first teacher response actually contains reasoning."""
    global _logged_thinking_sample
    if _logged_thinking_sample or not texts:
        return
    _logged_thinking_sample = True
    text = texts[0]
    reasoning, visible = split_thinking(text)
    logger.info(
        "Teacher thinking (observed on the first response): %s "
        "(%d reasoning chars, %d visible chars)",
        "PRESENT" if reasoning else "ABSENT",
        len(reasoning),
        len(visible),
    )
    if not visible:
        logger.warning(
            "First teacher response has NO visible content after stripping "
            "thinking — reasoning likely hit --max-tokens-per-turn (%s chars raw)",
            len(text),
        )


# --- Teacher backends: local vLLM (HF id / checkpoint path) or OpenRouter API ---


class LocalVLLMTeacher:
    """Tutor model loaded locally via vLLM; chat template applied explicitly."""

    def __init__(self, args):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        logger.info("Loading teacher model %s via vLLM…", args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        # Optional engine-level seed: reproducible, while per-request seeds
        # would make repeated samples of one prompt identical.
        self.llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            seed=getattr(args, "engine_seed", None),
        )
        self.sampling_params = SamplingParams(
            temperature=args.teacher_temperature,
            top_p=args.teacher_top_p,
            max_tokens=args.max_tokens_per_turn,
        )
        self.enable_thinking = args.enable_thinking
        self._verified = False
        logger.info("Teacher thinking: %s", describe_teacher_thinking(args))

    async def generate(self, conversations: list[Conversation]) -> list[str]:
        prompts = _render_teacher_prompts(self.tokenizer, conversations, self.enable_thinking)
        if not self._verified:
            _verify_prompt_consistency(prompts[0], conversations[0])
            # Qwen3-style templates prefill "<think>" when thinking is on; a
            # mismatch means the template ignores enable_thinking.
            has_prefill = "<think>" in prompts[0][-200:]
            logger.info(
                "Teacher thinking: enable_thinking=%s, rendered prompt %s a <think> prefill%s",
                self.enable_thinking,
                "HAS" if has_prefill else "has NO",
                "" if has_prefill == self.enable_thinking
                else " — MISMATCH: this template appears to ignore enable_thinking",
            )
            self._verified = True
        outputs = self.llm.generate(prompts, self.sampling_params)
        texts = [o.outputs[0].text.strip() for o in outputs]
        _log_first_thinking_sample(texts)
        return texts

    def count_tokens(self, conv: Conversation) -> int:
        return sum(len(self.tokenizer.encode(m["content"])) for m in conv.messages)

    def count_text_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text)) if text else 0

    def thinking_tokens(self, index: int, text: str) -> tuple[int, str]:
        """Reasoning tokens for the ``index``-th response of the last batch."""
        return self.count_text_tokens(thinking_text(text)), "tokenizer"

    def close(self) -> None:
        """Free the engine so the reward model can be loaded afterwards."""
        del self.llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def _field(obj, name):
    """Read ``name`` from a usage block (pydantic model, dict, or None),
    including unknown fields kept in ``model_extra``."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return value


def _usage_dict(resp) -> Any:
    """The response's usage block in a loggable form."""
    usage = getattr(resp, "usage", None)
    if usage is None or isinstance(usage, dict):
        return usage
    dump = getattr(usage, "model_dump", None)
    return dump() if dump else str(usage)


def _reported_reasoning_tokens(resp, has_inline_reasoning: bool) -> int | None:
    """Provider-reported thinking tokens for one completion, or None.

    Uses ``completion_tokens_details.reasoning_tokens`` when present, else
    ``total - prompt - completion`` (Gemini reports hidden thoughts only in
    the total), which is valid only when no reasoning was returned inline.
    """
    usage = getattr(resp, "usage", None)
    details = _field(usage, "completion_tokens_details")
    tokens = _field(details, "reasoning_tokens")
    if tokens is None:
        # Vertex/native spelling.
        tokens = _field(details, "thoughts_token_count") or _field(
            usage, "thoughts_token_count"
        )
    if tokens is not None:
        return int(tokens)
    if has_inline_reasoning:
        return None
    total, prompt, completion = (
        _field(usage, "total_tokens"),
        _field(usage, "prompt_tokens"),
        _field(usage, "completion_tokens"),
    )
    if None in (total, prompt, completion):
        return None
    return max(0, int(total) - int(prompt) - int(completion))


class OpenRouterTeacher:
    """Tutor model behind an OpenAI-compatible API (OpenRouter by default,
    Gemini with --use-gemini); the chat template is applied server-side.
    Failed calls (after retries) yield an empty string so the caller can mark
    the conversation failed.
    """

    def __init__(self, args):
        if not os.environ.get(args.teacher_api_key_env):
            raise SystemExit(
                f"API teacher requires the {args.teacher_api_key_env} environment variable"
            )
        logger.info(
            "Using API teacher model %s at %s", args.model, args.teacher_api_base
        )
        self.client = get_async_openai_client(args.teacher_api_base, args.teacher_api_key_env)
        self.semaphore = get_concurrency_semaphore("eval_teacher", args.teacher_concurrency)
        self.extra_body = teacher_thinking_extra_body(args)
        self.args = args
        self._logged = False
        self._logged_token_source = False
        # Provider-reported reasoning tokens for the last generate() batch.
        self._reasoning_tokens: list[int | None] = []
        self._first_usage: Any = None
        logger.info("Teacher thinking: %s", describe_teacher_thinking(args))

    async def generate(self, conversations: list[Conversation]) -> list[str]:
        if not self._logged:
            logger.info(
                "First teacher request messages (template applied server-side):\n%s",
                json.dumps(conversations[0].teacher_view(), indent=2, ensure_ascii=False),
            )
            self._logged = True

        async def one(conv: Conversation) -> tuple[str, int | None]:
            async def _call():
                resp = await self.client.chat.completions.create(
                    model=self.args.model,
                    messages=conv.teacher_view(),
                    temperature=self.args.teacher_temperature,
                    top_p=self.args.teacher_top_p,
                    max_tokens=self.args.max_tokens_per_turn,
                    extra_body=self.extra_body,
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                # Re-inline out-of-band reasoning as a <think> block so it is
                # kept in raw_messages but stripped downstream.
                reasoning = getattr(msg, "reasoning_content", None) or getattr(
                    msg, "reasoning", None
                )
                if reasoning and "</think>" not in content:
                    content = f"<think>{reasoning}</think>{content}"
                if self._first_usage is None:
                    self._first_usage = _usage_dict(resp)
                return content, _reported_reasoning_tokens(
                    resp, bool(thinking_text(content))
                )

            text, reasoning_tokens = await call_openai_with_retries(
                _call,
                label="eval_teacher",
                use_fallback=True,
                fallback=("", None),
                semaphore=self.semaphore,
                num_retries=self.args.num_retries,
            )
            return (text or "").strip(), reasoning_tokens

        results = await asyncio.gather(*(one(c) for c in conversations))
        texts = [text for text, _ in results]
        self._reasoning_tokens = [tokens for _, tokens in results]
        self._log_token_source(texts)
        _log_first_thinking_sample(texts)
        return texts

    def _log_token_source(self, texts: list[str]) -> None:
        """Log once where thinking-token counts come from, plus the raw usage block."""
        if self._logged_token_source or not texts:
            return
        self._logged_token_source = True
        logger.info("Teacher usage block (first response): %s", self._first_usage)
        reported = [t for t in self._reasoning_tokens if t is not None]
        if reported:
            logger.info(
                "Teacher thinking tokens: provider-reported for %d/%d responses "
                "in the first batch (avg %.1f tokens)",
                len(reported),
                len(texts),
                sum(reported) / len(reported),
            )
            return
        logger.info(
            "Teacher thinking tokens: no usage counts available — falling back "
            "to a ~4 chars/token estimate over inlined <think> text%s",
            " (a Gemini tutor hides its thoughts, so this will read as 0)"
            if self.args.use_gemini
            else "",
        )

    def count_tokens(self, conv: Conversation) -> int:
        # No local tokenizer: ~4 chars/token heuristic for the budget check.
        return sum(len(m["content"]) for m in conv.messages) // 4

    def count_text_tokens(self, text: str) -> int:
        return len(text) // 4 if text else 0

    def thinking_tokens(self, index: int, text: str) -> tuple[int, str]:
        """Reasoning tokens for the ``index``-th response of the last batch:
        provider count if available, else a 4 chars/token estimate.
        """
        reported = (
            self._reasoning_tokens[index] if index < len(self._reasoning_tokens) else None
        )
        if reported is not None:
            return reported, "api"
        return self.count_text_tokens(thinking_text(text)), "estimate"

    def close(self) -> None:
        pass


def make_teacher(args):
    if args.use_gemini:
        args.teacher_api_base = args.gemini_api_base
        args.teacher_api_key_env = args.gemini_api_key_env
        return OpenRouterTeacher(args)
    return OpenRouterTeacher(args) if args.use_openrouter else LocalVLLMTeacher(args)


async def _student_turn(conv: Conversation, client, args, semaphore) -> None:
    async def _call():
        resp = await client.chat.completions.create(
            model=args.student_model,
            messages=conv.student_view(),
            temperature=args.student_temperature,
            max_tokens=args.student_max_tokens,
        )
        return resp.choices[0].message.content or ""

    try:
        text = await call_openai_with_retries(
            _call,
            label="eval_student",
            use_fallback=False,
            semaphore=semaphore,
            num_retries=args.num_retries,
        )
    except Exception as e:
        logger.error("Student call failed hard, ending conversation: %s", e)
        conv.finish("student_failed")
        return
    conv.messages.append({"role": "student", "content": text.strip()})


async def rollout(args, problems: list[tuple[str, str]]) -> list[Conversation]:
    conversations = [
        Conversation(problem, answer)
        for problem, answer in problems
        for _ in range(args.num_samples_per_problem)
    ]

    teacher = make_teacher(args)
    client = get_async_openai_client(args.student_api_base, args.student_api_key_env)
    semaphore = get_concurrency_semaphore("eval_student", args.student_concurrency)

    round_idx = 0
    while True:
        active = [c for c in conversations if not c.done]
        if not active:
            break
        round_idx += 1
        logger.info(
            "Round %d: teacher turn for %d active conversations", round_idx, len(active)
        )

        texts = await teacher.generate(active)
        for i, (conv, text) in enumerate(zip(active, texts)):
            conv.messages.append({"role": "teacher", "content": text})
            tokens, source = teacher.thinking_tokens(i, text)
            conv.thinking_tokens.append(tokens)
            conv.thinking_tokens_source = source
            if not text:
                conv.finish("teacher_failed")
            elif _END_OF_CONVERSATION in text:
                conv.finish("end_of_conversation")
            elif conv.num_teacher_turns >= args.max_teacher_turns:
                conv.finish("max_teacher_turns")
            elif teacher.count_tokens(conv) > args.max_conversation_tokens:
                conv.finish("token_budget")

        active = [c for c in conversations if not c.done]
        if not active:
            break
        logger.info(
            "Round %d: student turn for %d active conversations", round_idx, len(active)
        )
        await asyncio.gather(
            *(_student_turn(conv, client, args, semaphore) for conv in active)
        )
        for conv in active:
            if not conv.done and teacher.count_tokens(conv) > args.max_conversation_tokens:
                conv.finish("token_budget")

    teacher.close()
    return conversations


def make_judge_client(args) -> tuple[Any, str, dict[str, Any]]:
    """(client, model, extra_body) for all judge calls; ``--gemini`` uses
    Gemini via its OpenAI-compatible endpoint instead of the judge server."""
    if args.gemini:
        client = get_async_openai_client(args.gemini_api_base, args.gemini_api_key_env)
        logger.info(
            "JUDGE BACKEND: Gemini API — model=%s api_base=%s thinking_level=%s "
            "(key from $%s); the local judge server (%s at %s) is NOT used",
            args.gemini_model, args.gemini_api_base, args.gemini_reasoning_effort,
            args.gemini_api_key_env, args.judge_model, args.judge_api_base,
        )
        # reasoning_effort maps to Gemini's thinking_level.
        return client, args.gemini_model, {"reasoning_effort": args.gemini_reasoning_effort}
    client = get_async_openai_client(args.judge_api_base, args.judge_api_key_env)
    logger.info(
        "JUDGE BACKEND: local judge server — model=%s api_base=%s",
        args.judge_model, args.judge_api_base,
    )
    # Qwen3 judge: disable thinking so the response is pure JSON.
    return client, args.judge_model, {"chat_template_kwargs": {"enable_thinking": False}}


_judge_response_logged = False


def _log_first_judge_response(resp) -> None:
    """Log once the model id reported by the first judge response."""
    global _judge_response_logged
    if not _judge_response_logged:
        _judge_response_logged = True
        logger.info(
            "First judge response served by model=%r (response id=%s)",
            getattr(resp, "model", None), getattr(resp, "id", None),
        )


# --- LLM judge: whole-conversation scores for pedagogy + factual correctness ---

_JUDGE_CRITERIA = ("pedagogy", "correctness")


def _render_conversation_for_judge(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- {'Teacher' if m['role'] == 'teacher' else 'Student'}: {m['content']}"
        for m in messages
    )


def _parse_judge_score(text: str) -> dict[str, Any] | None:
    """Parse one criterion's judge JSON (``{"reasoning", "score"}``).
    Returns None on any parse failure so bad votes are simply dropped.
    """
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        start, end = text.find("{"), text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        data = json.loads(text[start:end])
        score = float(data["score"])
        if not 1 <= score <= 5:
            return None
        return {"score": score, "reasoning": str(data.get("reasoning", ""))}
    except (json.JSONDecodeError, Exception):
        pass
    # Unescaped LaTeX in `reasoning` breaks json.loads; recover the score by regex.
    matches = _DECISION_RE.findall(text)
    if matches:
        logger.warning(
            f"Judge JSON malformed, recovered score via regex: {matches[-1]}"
        )
        try:
            return {"score": float(matches[-1]), "reasoning": "<unparsed JSON>"}
        except ValueError:
            pass
    logger.warning(f"Failed to parse judge response, score counted as missing: {text[:500]}")
    return None


async def _judge_record(
    record: dict[str, Any], client, model: str, extra_body: dict[str, Any], args, semaphore
) -> None:
    """Score one conversation on each criterion separately with
    ``--judge-votes`` parallel votes; store the mean of the valid votes.
    """
    conversation = _render_conversation_for_judge(record["messages"])

    async def _call(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            extra_body=extra_body,
        )
        _log_first_judge_response(resp)
        return resp.choices[0].message.content or ""

    async def _criterion_votes(criterion: str) -> list[dict[str, Any]]:
        prompt = _JUDGE_PROMPTS[criterion].format(
            problem=record["problem"],
            answer=record.get("answer", ""),
            conversation=conversation,
        )
        raw_votes = await asyncio.gather(
            *(
                call_openai_with_retries(
                    _call,
                    prompt,
                    label=f"eval_judge_{criterion}",
                    use_fallback=True,
                    fallback=None,
                    semaphore=semaphore,
                    num_retries=args.num_retries,
                )
                for _ in range(args.judge_votes)
            )
        )
        return [p for p in (_parse_judge_score(v) for v in raw_votes if v) if p]

    all_votes = await asyncio.gather(
        *(_criterion_votes(criterion) for criterion in _JUDGE_CRITERIA)
    )
    votes = dict(zip(_JUDGE_CRITERIA, all_votes))

    record["judge"] = {
        **{
            f"{criterion}_score": (
                sum(v["score"] for v in votes[criterion]) / len(votes[criterion])
                if votes[criterion]
                else None
            )
            for criterion in _JUDGE_CRITERIA
        },
        **{
            f"{criterion}_num_valid_votes": len(votes[criterion])
            for criterion in _JUDGE_CRITERIA
        },
        # A conversation counts as judged only if every criterion got ≥1 valid vote.
        "num_valid_votes": min(len(votes[c]) for c in _JUDGE_CRITERIA),
        "votes": votes,
    }


async def judge_conversations(records: list[dict[str, Any]], args) -> dict[str, float]:
    """Judge every conversation; returns aggregate metrics and annotates each
    record with a ``judge`` dict.
    """
    client, model, extra_body = make_judge_client(args)
    semaphore = get_concurrency_semaphore("eval_judge", args.judge_concurrency)

    await asyncio.gather(
        *(_judge_record(r, client, model, extra_body, args, semaphore) for r in records)
    )

    metrics: dict[str, float] = {}
    for criterion in _JUDGE_CRITERIA:
        values = [
            r["judge"][f"{criterion}_score"]
            for r in records
            if r["judge"][f"{criterion}_score"] is not None
        ]
        metrics[f"judge_{criterion}_avg"] = sum(values) / len(values) if values else 0.0
    metrics["judge_num_failed_conversations"] = sum(
        1 for r in records if r["judge"]["num_valid_votes"] == 0
    )
    return metrics


# --- Metrics ---


def compute_metrics(scores: list[list[float]]) -> dict[str, float]:
    per_conv_means = [sum(s) / len(s) for s in scores if len(s) > 0]
    macro = sum(per_conv_means) / len(per_conv_means) if per_conv_means else 0.0
    all_scores = [x for s in scores for x in s]
    micro = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {
        "pedagogical_reward_macro_avg": macro,
        "pedagogical_reward_micro_avg": micro,
        "num_conversations": len(scores),
        "num_scored_teacher_turns": len(all_scores),
    }


def compute_thinking_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Average private reasoning tokens per teacher turn and per conversation.

    Returns {} when records carry no thinking-token counts.
    """
    scored = [r for r in records if r.get("thinking_tokens") is not None]
    per_conv = [r["thinking_tokens"] for r in scored]
    if not per_conv:
        return {}
    per_turn = [t for turns in per_conv for t in turns]
    if not per_turn:
        return {}
    # An estimate anywhere makes the aggregate an estimate.
    sources = {r.get("thinking_tokens_source") for r in scored}
    source = next((s for s in ("estimate", "api", "tokenizer") if s in sources), None)
    return {
        "thinking_tokens_avg_per_turn": sum(per_turn) / len(per_turn),
        "thinking_tokens_avg_per_conversation": sum(per_turn) / len(per_conv),
        "thinking_turns_fraction": sum(1 for t in per_turn if t > 0) / len(per_turn),
        "thinking_tokens_source": source,
    }


def compute_reference_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """RM comparison against the human reference turn (continuation benchmarks):
    ``rm_win_rate_vs_reference`` is the fraction where the model scores higher.
    """
    pairs = [
        (r["pedagogical_reward"][-1], r["reference_pedagogical_reward"][-1])
        for r in records
        if r.get("pedagogical_reward") and r.get("reference_pedagogical_reward")
    ]
    if not pairs:
        return {}
    ref_scores = [ref for _, ref in pairs]
    wins = sum(1 for model, ref in pairs if model > ref)
    return {
        "reference_pedagogical_reward_avg": sum(ref_scores) / len(ref_scores),
        "rm_win_rate_vs_reference": wins / len(pairs),
        "num_reference_comparisons": len(pairs),
    }


# --- Main ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eduardo pedagogical-RM evaluation")

    # Teacher (model under evaluation)
    p.add_argument("--model", required=True, help="HF hub id or local path of the tutor model to evaluate (e.g. eth-nlped/TutorRL-7B), or an OpenRouter model id with --use-openrouter (e.g. openai/gpt-4o)")
    p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--teacher-top-p", type=float, default=1.0)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=12000)
    p.add_argument("--enable-thinking", action="store_true", help="Let the TUTOR think: enable_thinking on the chat template (local / vLLM-served) or reasoning.enabled (OpenRouter). A Gemini tutor always uses the API default and ignores this. Unrelated to the judge's --gemini-reasoning-effort")

    # Teacher via OpenRouter (or any OpenAI-compatible API) instead of local vLLM
    p.add_argument("--use-openrouter", action="store_true", help="Call --model via the OpenRouter API instead of loading it locally (requires OPENROUTER_API_KEY)")
    p.add_argument("--use-gemini", action="store_true", help="Call --model via the Gemini API instead of loading it locally (e.g. --model gemini-2.5-pro or gemini-3.1-pro-preview; requires GEMINI_API_KEY)")
    p.add_argument("--teacher-api-base", default="https://openrouter.ai/api/v1")
    p.add_argument("--teacher-api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--teacher-concurrency", type=int, default=16)

    # Dataset
    p.add_argument(
        "--benchmark",
        choices=["bigmath", "mathdial", "eedi"],
        default="bigmath",
        help="bigmath: full rollout with the student simulator; mathdial/eedi: "
        "generate the middle teacher turn of real human dialogs (no student server needed)",
    )
    p.add_argument("--dataset", default="rd211/Big-Math-RL-Verified-Filtered", help="Only used for --benchmark bigmath")
    p.add_argument("--split", default="test")
    p.add_argument("--num-problems", type=int, default=500, help="-1 for the full split")
    p.add_argument("--num-samples-per-problem", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # Conversation limits
    p.add_argument("--max-teacher-turns", type=int, default=5)
    p.add_argument("--max-tokens-per-turn", type=int, default=4192)
    p.add_argument("--max-conversation-tokens", type=int, default=16384)

    # Student simulator (OpenAI-compatible endpoint, as in training)
    p.add_argument("--student-model", default=os.environ.get("STUDENT_MODEL_NAME", f"Llama-3.1-8B-Instruct-{os.environ.get('USER', '')}"))
    p.add_argument("--student-api-base", default=os.environ.get("STUDENT_API_BASE", "http://localhost:8080/v1"))
    p.add_argument("--student-api-key-env", default="SERVING_API_KEY")
    p.add_argument("--student-temperature", type=float, default=0.0)
    p.add_argument("--student-max-tokens", type=int, default=500)
    p.add_argument("--student-concurrency", type=int, default=64)
    p.add_argument("--num-retries", type=int, default=3)

    # Reward model
    p.add_argument("--reward-model", default="eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel")
    p.add_argument("--rm-only-last-message", action="store_true", help="Score only the final teacher turn per conversation")

    # LLM judge (OpenAI-compatible endpoint, as in training)
    p.add_argument("--no-judge", action="store_true", help="Skip the LLM-judge scoring (pedagogy + correctness)")
    p.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL_NAME", f"Qwen3.6-27B-{os.environ.get('USER', '')}"))
    p.add_argument("--judge-api-base", default=os.environ.get("JUDGE_API_BASE", "http://localhost:8080/v1"))
    p.add_argument("--judge-api-key-env", default="SERVING_API_KEY")
    p.add_argument("--judge-temperature", type=float, default=0.3)
    p.add_argument("--judge-max-tokens", type=int, default=16384)
    p.add_argument("--judge-votes", type=int, default=1)
    p.add_argument("--judge-concurrency", type=int, default=64)

    # Gemini judge (opt-in): replaces the judge endpoint above for all judge calls.
    p.add_argument("--gemini", action="store_true",
                   help="Judge with Gemini instead of the judge server (key in GEMINI_API_KEY)")
    p.add_argument("--gemini-model", default="gemini-3.1-pro-preview")
    p.add_argument("--gemini-reasoning-effort", default="high",
                   choices=["low", "medium", "high"],
                   help="Gemini thinking level for JUDGE calls only; never applied to the tutor")
    p.add_argument("--gemini-api-base",
                   default="https://generativelanguage.googleapis.com/v1beta/openai/")
    p.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")

    # Pipeline control / IO
    p.add_argument("--stage", choices=["all", "rollout", "score"], default="all")
    p.add_argument("--conversations", default=None, help="Existing conversations .jsonl to score (required for --stage score)")
    p.add_argument("--output-dir", default=None, help="Defaults to $WORK_DIR/eval_results or ./eval_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Fail fast: from_pretrained would otherwise treat a bad path as a hub id.
    if not (args.use_openrouter or args.use_gemini) and args.model.startswith(("/", "./", "~")):
        model_path = Path(args.model).expanduser()
        if not model_path.is_dir():
            raise SystemExit(f"--model points to a local path that does not exist: {model_path}")
        args.model = str(model_path)

    out_dir = Path(
        args.output_dir
        or (Path(os.environ["WORK_DIR"]) / "eval_results" if os.environ.get("WORK_DIR") else Path("eval_results"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = Path(args.model.rstrip("/")).name

    # Stage 1: rollout
    if args.stage in ("all", "rollout"):
        start = time.time()
        if args.benchmark == "bigmath":
            problems = load_problems(args.dataset, args.split, args.num_problems, args.seed)
            conversations = asyncio.run(rollout(args, problems))
            records = [c.to_record() for c in conversations]
        else:
            loader = load_mathdial_samples if args.benchmark == "mathdial" else load_eedi_samples
            samples = loader(args.split, args.num_problems, args.seed)
            records = asyncio.run(continuation_rollout(args, samples))
        logger.info("Rollout of %d conversations took %.1fs", len(records), time.time() - start)

        conv_path = out_dir / f"conversations_{args.benchmark}_{model_tag}_{ts}.jsonl"
        with open(conv_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote conversations to %s", conv_path)
    else:
        if not args.conversations:
            raise SystemExit("--stage score requires --conversations <path.jsonl>")
        conv_path = Path(args.conversations)
        with open(conv_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        logger.info("Loaded %d conversations from %s", len(records), conv_path)

    if args.stage == "rollout":
        logger.info(
            "Rollout-only stage done. Score with:\n"
            "  python -m eval.run_middleturn_eval --model %s --stage score --conversations %s",
            args.model,
            conv_path,
        )
        return

    # Stage 2: pedagogical RM scoring
    from eval.pedagogical_rm import score_conversations

    # Continuation benchmarks: only the generated (final) teacher turn is RM-scored.
    is_continuation = bool(records and records[0].get("benchmark") in ("mathdial", "eedi"))
    only_last_message = args.rm_only_last_message or is_continuation

    logger.info("Scoring %d conversations with RM %s…", len(records), args.reward_model)
    # Score the human reference turn in the same context for a win rate.
    ref_indices = [i for i, r in enumerate(records) if r.get("reference_teacher_turn")]
    ref_records = [
        {
            "problem": records[i]["problem"],
            "messages": records[i]["messages"][:-1]
            + [{"role": "teacher", "content": records[i]["reference_teacher_turn"]}],
        }
        for i in ref_indices
    ]
    all_scores = score_conversations(
        records + ref_records,
        reward_model_path=args.reward_model,
        only_last_message=only_last_message,
    )
    scores = all_scores[: len(records)]
    for record, conv_scores in zip(records, scores):
        record["pedagogical_reward"] = conv_scores
    for i, conv_scores in zip(ref_indices, all_scores[len(records):]):
        records[i]["reference_pedagogical_reward"] = conv_scores

    metrics = compute_metrics(scores)
    metrics.update(compute_reference_metrics(records))
    metrics.update(compute_thinking_metrics(records))

    # Stage 3: LLM-judge scoring
    if not args.no_judge:
        logger.info(
            "Judging %d conversations with %s (%d votes each)…",
            len(records),
            args.gemini_model if args.gemini else args.judge_model,
            args.judge_votes,
        )
        judge_metrics = asyncio.run(judge_conversations(records, args))
        metrics.update(judge_metrics)

    results = {
        "model": args.model,
        "benchmark": args.benchmark,
        "teacher_thinking": describe_teacher_thinking(args),
        "reward_model": args.reward_model,
        "dataset": args.dataset,
        "split": args.split,
        "timestamp": ts,
        "args": vars(args),
        "metrics": metrics,
        "conversations": records,
    }
    results_path = out_dir / f"eval_{args.benchmark}_{model_tag}_{ts}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"Model:                        {args.model}")
    print(f"Benchmark:                    {records[0].get('benchmark', args.benchmark) if records else args.benchmark}")
    print(f"Teacher thinking:             {describe_teacher_thinking(args)}")
    print(f"Reward model:                 {args.reward_model}")
    print(f"Conversations:                {metrics['num_conversations']}")
    print(f"Scored teacher turns:         {metrics['num_scored_teacher_turns']}")
    print(f"Pedagogical reward macro avg: {metrics['pedagogical_reward_macro_avg']:.4f}")
    print(f"Pedagogical reward micro avg: {metrics['pedagogical_reward_micro_avg']:.4f}")
    if "thinking_tokens_avg_per_turn" in metrics:
        # Records without a source: infer it from the backend flags.
        source = metrics.get("thinking_tokens_source") or (
            "estimate" if (args.use_openrouter or args.use_gemini) else "tokenizer"
        )
        note = {"api": " (provider-reported)", "estimate": " (≈, 4 chars/token)"}.get(source, "")
        print(f"Thinking tokens / turn:       {metrics['thinking_tokens_avg_per_turn']:.1f}{note}")
        print(f"Thinking tokens / conv:       {metrics['thinking_tokens_avg_per_conversation']:.1f}")
        print(f"Teacher turns that thought:   {metrics['thinking_turns_fraction']:.2%}")
    if "rm_win_rate_vs_reference" in metrics:
        print(f"Reference (human) RM avg:     {metrics['reference_pedagogical_reward_avg']:.4f}")
        print(f"RM win rate vs reference:     {metrics['rm_win_rate_vs_reference']:.2%} (model turn scored higher on {int(metrics['rm_win_rate_vs_reference'] * metrics['num_reference_comparisons'])}/{metrics['num_reference_comparisons']})")
    if "judge_pedagogy_avg" in metrics:
        print(f"Judge pedagogy:               {metrics['judge_pedagogy_avg']:.4f} (1-5, higher = better scaffolding, no premature answers)")
        print(f"Judge factual correctness:    {metrics['judge_correctness_avg']:.4f} (1-5, higher = factually correct tutoring)")
        print(f"Judge failed conversations:   {metrics['judge_num_failed_conversations']}")
    print(f"Results written to:           {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
