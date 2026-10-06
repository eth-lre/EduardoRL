"""Offline Δ_solve / leak evaluation of a checkpoint on test.parquet.

Reproduces the training validation metrics offline:

* ``delta_transfer`` -- normalised gain ``(post - pre) / (1 - pre)`` on the
  unseen near-transfer variant (the dialogue is about the original problem).
* ``delta_same`` -- the same quantity graded on the original problem;
  ``transfer_gap = delta_same - delta_transfer``.
* ``leak_rate`` -- fraction of conversations rejected by the leakage judge.

Both post-solve rates come from one rollout per row. Deltas are reported both
ungated and gated (zeroed on leak-rejected conversations, as in training).

Usage::

    python -m eval.run_env_test_eval --model <checkpoint> \\
        --test-parquet $WORK_DIR/data/eduardo/test.parquet --enable-thinking --gemini

Requires a running student server and either a judge server or ``--gemini``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval import run_middleturn_eval  # noqa: E402
from eval.run_middleturn_eval import (  # noqa: E402
    Conversation,
    _END_OF_CONVERSATION,
    _student_turn,
    describe_teacher_thinking,
    make_judge_client,
    make_teacher,
)
from eduardo.metrics import correctness, judge_eval, judge_quality  # noqa: E402
from eduardo.reward_function import _normalized_delta  # noqa: E402
from eduardo.utils import (  # noqa: E402
    get_async_openai_client,
    get_concurrency_semaphore,
    load_prompt,
    strip_teacher_controls,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eduardo.transfer_eval")

# Training prompts (not eval/prompts/), for parity with the training validation pass.
_TEACHER_SYSTEM_PROMPT = load_prompt("teacher_prompt")
_STUDENT_SYSTEM_PROMPT = load_prompt("student_prompt")

# Makes run_middleturn_eval._verify_prompt_consistency check against the training prompt.
run_middleturn_eval._TEACHER_SYSTEM_PROMPT = _TEACHER_SYSTEM_PROMPT


# A tag-free teacher turn is either unterminated reasoning (thinking-prefill
# templates, e.g. Qwen3.6) or the visible reply (non-prefill models, API tutors).
# Choosing wrongly blanks every teacher turn, so the rule is decided from the
# chat template rather than from markers in the responses.


def _has_think_prefill(teacher: Any, conv: "ValConversation", args) -> bool | None:
    """Whether the chat template pre-fills ``<think>`` into the prompt.

    Returns ``None`` for an API teacher (no local tokenizer).
    """
    tokenizer = getattr(teacher, "tokenizer", None)
    if tokenizer is None:
        return None
    rendered = tokenizer.apply_chat_template(
        conv.teacher_view(),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    # The prefill sits right after the assistant header at the end of the prompt.
    return "<think>" in rendered[-200:]


def resolve_assume_thinking(teacher: Any, conv: "ValConversation", args, texts: list[str]) -> bool:
    """Decide the ``assume_thinking`` rule for this run from the chat template.

    Response markers are only used to warn about contradictions, since models
    that think on some turns and not others produce mixed batches.
    """
    if args.assume_thinking in ("yes", "no"):
        assume = args.assume_thinking == "yes"
        logger.info("Thinking mode FORCED: assume_thinking=%s (--assume-thinking %s)",
                    assume, args.assume_thinking)
        return assume

    prefill = _has_think_prefill(teacher, conv, args)
    if prefill is None:
        logger.info(
            "Thinking mode AUTO: API teacher (no local template) -> "
            "assume_thinking=False; tag-free turns are the visible reply."
        )
        return False

    logger.info(
        "Thinking mode AUTO: chat template %s <think> prefill -> "
        "assume_thinking=%s (%s)",
        "HAS a" if prefill else "has NO", prefill,
        "tag-free turns are unterminated reasoning and get blanked"
        if prefill else
        "tag-free turns are kept as the visible reply; only <think> without "
        "</think> is blanked",
    )

    n_marked = sum(1 for t in texts if t and ("</think>" in t or "<think>" in t))
    if prefill and n_marked == 0 and texts:
        logger.warning(
            "Template prefills <think>, yet NONE of the %d first-batch turns "
            "closed it. Either every turn overran --max-tokens-per-turn=%d or "
            "the prefill detection is wrong; watch blanked_teacher_turn_rate "
            "and rerun with --assume-thinking no if it is high.",
            len(texts), args.max_tokens_per_turn,
        )
    elif not prefill and n_marked and n_marked < len(texts):
        logger.info(
            "%d/%d first-batch turns think explicitly; the rest are replies "
            "written without thinking and are kept (this mixed case is exactly "
            "what the old response-counting heuristic got wrong).",
            n_marked, len(texts),
        )
    return prefill


# --- Conversation seeded from a dataset row ---


class ValConversation(Conversation):
    """One rollout over a ``test.parquet`` row, using the row's own teacher
    system prompt and the training student prompt.
    """

    def __init__(self, row: dict[str, Any], row_index: int = -1, sample_index: int = 0):
        extra = row["extra_info"]
        # Row index is the clustering unit for standard errors (compute_summary).
        self.row_index = row_index
        self.sample_index = sample_index
        super().__init__(extra["problem"], str(extra["answer"]))
        # Teacher system message and pre-seeded student opener, verbatim from the parquet.
        seed = list(row["prompt"])
        self.system_prompt = next(
            (m["content"] for m in seed if m["role"] == "system"),
            _TEACHER_SYSTEM_PROMPT.format(problem=self.problem),
        )
        opener = next((m["content"] for m in seed if m["role"] == "user"), None)
        if opener is None:
            raise ValueError("dataset row has no pre-seeded student opener")
        self.messages = [{"role": "student", "content": opener}]

        self.transfer_problem = str(extra.get("transfer_problem") or "")
        self.transfer_answer = str(extra.get("transfer_answer") or "")
        # Δ baselines, measured with the same student model at dataset creation.
        self.pre_solve_rate = float(extra.get("pre_solve_rate") or 0.0)
        self.pre_solve_rate_transfer = float(
            extra.get("pre_solve_rate_transfer_problem")
            if extra.get("pre_solve_rate_transfer_problem") is not None
            else extra.get("pre_solve_rate") or 0.0
        )
        self.eoc_used = False
        self.metrics: dict[str, Any] = {}
        # Set by resolve_assume_thinking after the first teacher batch.
        self.assume_thinking = True

    def teacher_view(self) -> list[dict[str, str]]:
        out = [{"role": "system", "content": self.system_prompt}]
        for m in self.messages:
            out.append(
                {
                    "role": "assistant" if m["role"] == "teacher" else "user",
                    "content": m["content"],
                }
            )
        return out

    def student_view(self) -> list[dict[str, str]]:
        out = [
            {
                "role": "system",
                "content": _STUDENT_SYSTEM_PROMPT.format(problem=self.problem),
            }
        ]
        for m in self.messages:
            if m["role"] == "teacher":
                out.append(
                    {
                        "role": "user",
                        "content": strip_teacher_controls(
                            m["content"], assume_thinking=self.assume_thinking
                        ),
                    }
                )
            else:
                out.append({"role": "assistant", "content": m["content"]})
        return out

    def reward_view(self) -> list[dict[str, str]]:
        """The transcript shape ``eduardo_reward_func`` receives: teacher =
        assistant, student = user, teacher content untouched (thinking intact).
        """
        return [
            {
                "role": "assistant" if m["role"] == "teacher" else "user",
                "content": m["content"],
            }
            for m in self.messages
        ]

    def judge_view(self) -> list[dict[str, str]]:
        """Student-visible transcript the judges score — teacher ``<think>``
        blocks and control tokens stripped (reward_function's ``judge_msgs``).
        """
        return [
            {
                **m,
                "content": strip_teacher_controls(
                    m["content"], assume_thinking=self.assume_thinking
                ),
            }
            if m["role"] == "assistant"
            else m
            for m in self.reward_view()
        ]

    def blanked_teacher_turns(self) -> int:
        """Teacher turns that strip to nothing in the student's/judge's view.

        Near 100% indicates a wrong ``assume_thinking`` rule.
        """
        return sum(
            1
            for m in self.messages
            if m["role"] == "teacher"
            and not strip_teacher_controls(
                m["content"], assume_thinking=self.assume_thinking
            ).strip()
        )

    def to_record(self) -> dict[str, Any]:
        rec = super().to_record()
        rec.update(
            {
                "transfer_problem": self.transfer_problem,
                "transfer_answer": self.transfer_answer,
                "pre_solve_rate": self.pre_solve_rate,
                "pre_solve_rate_transfer": self.pre_solve_rate_transfer,
                "system_prompt": self.system_prompt,
                "row_index": self.row_index,
                "sample_index": self.sample_index,
                "eoc_used": self.eoc_used,
                "assume_thinking": self.assume_thinking,
                "num_blanked_teacher_turns": self.blanked_teacher_turns(),
                "blanked_teacher_turn_rate": (
                    self.blanked_teacher_turns() / self.num_teacher_turns
                    if self.num_teacher_turns else 0.0
                ),
                **self.metrics,
            }
        )
        return rec


def load_val_rows(path: str, num_problems: int) -> list[dict[str, Any]]:
    """Read the eval parquet in file order (no shuffle), so ``--num-problems``
    selects the same rows across arms and reruns.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    if num_problems > 0:
        rows = rows[:num_problems]
    n_transfer = sum(1 for r in rows if r["extra_info"].get("transfer_problem"))
    logger.info(
        "Loaded %d rows from %s (%d carry a transfer variant)",
        len(rows),
        path,
        n_transfer,
    )
    if n_transfer < len(rows):
        logger.warning(
            "%d/%d rows have no transfer variant — delta_transfer falls back to "
            "the original problem for those, matching reward_function.py",
            len(rows) - n_transfer,
            len(rows),
        )
    return rows


# --- Rollout (mirrors EduardoAgentLoop + EduardoInteraction) ---


async def transfer_rollout(args, rows: list[dict[str, Any]]) -> list[ValConversation]:
    conversations = [
        ValConversation(row, row_index=i, sample_index=k)
        for i, row in enumerate(rows)
        for k in range(args.num_samples_per_problem)
    ]
    logger.info(
        "%d problems x %d samples = %d conversations",
        len(rows), args.num_samples_per_problem, len(conversations),
    )

    args.engine_seed = args.seed
    teacher = make_teacher(args)
    client = get_async_openai_client(args.student_api_base, args.student_api_key_env)
    # Same "student" concurrency budget as in training.
    semaphore = get_concurrency_semaphore("student", args.student_concurrency)

    round_idx = 0
    while True:
        active = [c for c in conversations if not c.done]
        if not active:
            break
        round_idx += 1
        logger.info("Round %d: teacher turn for %d conversations", round_idx, len(active))

        texts = await teacher.generate(active)
        if round_idx == 1:
            # Must be resolved before student_view() is first used.
            assume = resolve_assume_thinking(teacher, active[0], args, texts)
            for conv in conversations:
                conv.assume_thinking = assume
        for i, (conv, text) in enumerate(zip(active, texts)):
            conv.messages.append({"role": "teacher", "content": text})
            tokens, source = teacher.thinking_tokens(i, text)
            conv.thinking_tokens.append(tokens)
            conv.thinking_tokens_source = source
            if not text:
                conv.finish("teacher_failed")
                continue
            # As in training, EoC is ignored until min_teacher_turns_before_eoc
            # turns precede the tagged one.
            if (
                _END_OF_CONVERSATION in text
                and conv.num_teacher_turns - 1 >= args.min_teacher_turns_before_eoc
            ):
                conv.eoc_used = True
                conv.finish("end_of_conversation")
            elif conv.num_teacher_turns >= args.max_teacher_turns:
                conv.finish("max_teacher_turns")
            elif teacher.count_tokens(conv) > args.max_conversation_tokens:
                conv.finish("token_budget")

        active = [c for c in conversations if not c.done]
        if not active:
            break
        logger.info("Round %d: student turn for %d conversations", round_idx, len(active))
        await asyncio.gather(
            *(_student_turn(conv, client, args, semaphore) for conv in active)
        )
        for conv in active:
            if not conv.done and teacher.count_tokens(conv) > args.max_conversation_tokens:
                conv.finish("token_budget")

    teacher.close()
    return conversations


# --- Scoring: Δ on the transfer variant, Δ on the original, leak judge ---


def _retry_kwargs(args) -> dict[str, Any]:
    return {
        "num_retries": args.num_retries,
        "backoff_base": 2.0,
        "backoff_cap": 30.0,
        "jitter": True,
    }


async def _score_one(
    rec: dict[str, Any],
    args,
    student_client,
    judge_client,
    judge_model: str,
    judge_extra_body: dict[str, Any],
    eval_judge_client,
    student_semaphore,
    judge_semaphore,
    eval_judge_semaphore,
) -> dict[str, Any]:
    """All post-rollout measurements for one conversation, concurrently."""
    msgs = rec["reward_messages"]
    judge_msgs = rec["judge_messages"]
    problem, answer = rec["problem"], rec["answer"]
    transfer_problem = rec.get("transfer_problem") or ""
    transfer_answer = rec.get("transfer_answer") or ""
    # As in training: without a usable variant, test on the original problem.
    use_transfer = bool(transfer_problem and transfer_answer)

    common = dict(
        problem=problem,
        client=student_client,
        model=args.student_model,
        num_attempts=args.num_student_attempts,
        temperature=args.student_final_temperature,
        max_tokens=800,
        retry_kwargs=_retry_kwargs(args),
        semaphore=student_semaphore,
        omit_teacher_turns=args.omit_teacher_turns,
    )

    async def transfer_attempts():
        return await correctness.compute_fraction(
            messages=msgs,
            ground_truth=transfer_answer if use_transfer else answer,
            transfer_problem=transfer_problem if use_transfer else None,
            **common,
        )

    async def same_attempts():
        return await correctness.compute_fraction(
            messages=msgs,
            ground_truth=answer,
            transfer_problem=None,
            **common,
        )

    async def judges():
        return await judge_quality.majority_vote(
            messages=judge_msgs,
            problem=problem,
            answer=answer,
            client=judge_client,
            model=judge_model,
            num_votes=args.judge_votes,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            retry_kwargs=_retry_kwargs(args),
            semaphore=judge_semaphore,
            extra_body=judge_extra_body,
        )

    async def eval_judge():
        if eval_judge_client is None:
            return {"pedagogy": None, "correctness": None}
        return await judge_eval.score_conversation(
            messages=judge_msgs,
            problem=problem,
            answer=answer,
            client=eval_judge_client,
            model=args.eval_judge_model,
            temperature=args.judge_temperature,
            max_tokens=args.eval_judge_max_tokens,
            extra_body={"reasoning_effort": args.gemini_reasoning_effort},
            retry_kwargs=_retry_kwargs(args),
            semaphore=eval_judge_semaphore,
        )

    (
        (post_transfer, fmt_t, failed_t),
        (post_same, fmt_s, failed_s),
        (no_leak, teaching, judge_reasoning),
        eval_scores,
    ) = await asyncio.gather(
        transfer_attempts(), same_attempts(), judges(), eval_judge()
    )

    # Without a variant, the "transfer" attempts use the original's baseline.
    pre_transfer = float(
        rec["pre_solve_rate_transfer"] if use_transfer else rec["pre_solve_rate"]
    )
    pre_same = float(rec["pre_solve_rate"])
    leaked = int(no_leak == "REJECT")

    delta_transfer = _normalized_delta(post_transfer, pre_transfer)
    delta_same = _normalized_delta(post_same, pre_same)

    return {
        "used_transfer_problem": int(use_transfer),
        # Baselines Δ was actually measured against.
        "pre_solve_rate_transfer": pre_transfer,
        "pre_solve_rate": pre_same,
        "post_solve_transfer": post_transfer,
        "post_solve_same": post_same,
        "delta_transfer": delta_transfer,
        "delta_same": delta_same,
        # Raw percentage-point gain, un-normalised by the headroom (1 − pre).
        "delta_transfer_raw": post_transfer - pre_transfer,
        "delta_same_raw": post_same - pre_same,
        # Training-equivalent: judge_hard_gate zeroes Δ on a rejected rollout.
        "delta_transfer_gated": 0.0 if leaked else delta_transfer,
        "delta_same_gated": 0.0 if leaked else delta_same,
        # Portion of the gain that does not survive a change of surface story.
        "transfer_gap": delta_same - delta_transfer,
        "leaked": leaked,
        "teaching_rejected": int(teaching == "REJECT"),
        "judge_reasoning": judge_reasoning,
        "n_incorrect_format": fmt_t + fmt_s,
        "n_student_calls_failed": failed_t + failed_s,
        "eval_judge_pedagogy": eval_scores["pedagogy"],
        "eval_judge_correctness": eval_scores["correctness"],
        "eval_judge_ok": int(
            eval_scores["pedagogy"] is not None
            and eval_scores["correctness"] is not None
        ),
    }


async def score_records(records: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    student_client = get_async_openai_client(
        args.student_api_base, args.student_api_key_env
    )
    judge_client, judge_model, judge_extra_body = make_judge_client(args)
    eval_judge_client = None
    if args.eval_judge_model:
        if os.environ.get(args.gemini_api_key_env):
            eval_judge_client = get_async_openai_client(
                args.gemini_api_base, args.gemini_api_key_env
            )
        else:
            logger.warning(
                "1-5 eval judge disabled: $%s is not set", args.gemini_api_key_env
            )

    student_semaphore = get_concurrency_semaphore("student", args.student_concurrency)
    judge_semaphore = get_concurrency_semaphore("judge", args.judge_concurrency)
    eval_judge_semaphore = get_concurrency_semaphore(
        "eval_judge", args.eval_judge_concurrency
    )

    logger.info(
        "Scoring %d conversations: %d student final attempts × 2 (transfer + "
        "original), %d judge votes × 2 criteria%s",
        len(records),
        args.num_student_attempts,
        args.judge_votes,
        ", + 2 eval-judge calls" if eval_judge_client else "",
    )
    scored = await asyncio.gather(
        *(
            _score_one(
                rec,
                args,
                student_client,
                judge_client,
                judge_model,
                judge_extra_body,
                eval_judge_client,
                student_semaphore,
                judge_semaphore,
                eval_judge_semaphore,
            )
            for rec in records
        )
    )
    for rec, metrics in zip(records, scored):
        rec.update(metrics)
    return records


# --- Aggregation ---

_MEAN_KEYS = (
    "delta_transfer",
    "delta_same",
    "delta_transfer_raw",
    "delta_same_raw",
    "delta_transfer_gated",
    "delta_same_gated",
    "transfer_gap",
    "post_solve_transfer",
    "post_solve_same",
    "pre_solve_rate",
    "pre_solve_rate_transfer",
    "leaked",
    "teaching_rejected",
    "used_transfer_problem",
    "eoc_used",
    "num_teacher_turns",
    "eval_judge_ok",
    "blanked_teacher_turn_rate",
)


def compute_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        vals = [float(r[key]) for r in records if r.get(key) is not None]
        return statistics.fmean(vals) if vals else 0.0

    summary = {"num_conversations": len(records)}
    summary.update({k: mean(k) for k in _MEAN_KEYS})
    summary["leak_rate"] = summary.pop("leaked")
    summary["teaching_reject_rate"] = summary.pop("teaching_rejected")
    summary["eoc_rate"] = summary.pop("eoc_used")
    summary["transfer_tested_rate"] = summary.pop("used_transfer_problem")

    # 1-5 judges: averaged over parsed conversations only (see eval_judge_ok).
    for crit in ("pedagogy", "correctness"):
        vals = [
            float(r[f"eval_judge_{crit}"])
            for r in records
            if r.get(f"eval_judge_{crit}") is not None
        ]
        summary[f"eval_judge_{crit}"] = statistics.fmean(vals) if vals else 0.0

    summary["end_reasons"] = {
        reason: sum(1 for r in records if r.get("end_reason") == reason)
        for reason in sorted({r.get("end_reason") or "unknown" for r in records})
    }
    summary["n_student_calls_failed"] = sum(
        int(r.get("n_student_calls_failed", 0)) for r in records
    )
    summary["assume_thinking"] = bool(records[0].get("assume_thinking", True)) if records else True

    # Standard errors are clustered by problem: samples of one problem are
    # correlated, so average within a problem, then take the error across problems.
    groups: dict[Any, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r.get("row_index", r.get("problem")), []).append(r)
    summary["num_problems"] = len(groups)
    summary["samples_per_problem"] = (
        len(records) / len(groups) if groups else 0.0
    )
    for key, out in (("delta_transfer", "delta_transfer_stderr"),
                     ("delta_same", "delta_same_stderr"),
                     ("leaked", "leak_rate_stderr")):
        per_problem = [
            statistics.fmean([float(r[key]) for r in g if r.get(key) is not None])
            for g in groups.values()
            if any(r.get(key) is not None for r in g)
        ]
        summary[out] = (
            statistics.stdev(per_problem) / (len(per_problem) ** 0.5)
            if len(per_problem) > 1 else 0.0
        )

    # Report the tail of per-turn thinking length to check whether the cap binds.
    per_turn = [t for r in records for t in (r.get("thinking_tokens") or [])]
    summary["thinking_tokens_avg_per_turn"] = statistics.fmean(per_turn) if per_turn else 0.0
    summary["thinking_tokens_max_per_turn"] = max(per_turn) if per_turn else 0
    summary["thinking_tokens_p95_per_turn"] = (
        sorted(per_turn)[int(0.95 * (len(per_turn) - 1))] if per_turn else 0
    )
    # Truncated dialogues understate both leak rate and Δ.
    summary["token_budget_truncated_rate"] = (
        sum(1 for r in records if r.get("end_reason") == "token_budget") / len(records)
        if records else 0.0
    )
    if summary["token_budget_truncated_rate"] > 0.05:
        logger.warning(
            "%.1f%% of dialogues were cut off by --max-conversation-tokens. "
            "Those never reached a natural end, so leak_rate and Δ are both "
            "understated for this model — raise the budget before comparing it "
            "with models that were not truncated.",
            100 * summary["token_budget_truncated_rate"],
        )

    # A high rate means the student and judges saw empty teacher turns.
    if summary["blanked_teacher_turn_rate"] > 0.5:
        logger.error(
            "%.0f%% of teacher turns strip to NOTHING in the student/judge view "
            "(assume_thinking=%s). The tutor's replies never reached the student "
            "or the judges, so Δ and the judge scores are invalid. Rerun with "
            "--assume-thinking no if this model has no <think> prefill.",
            100 * summary["blanked_teacher_turn_rate"], summary["assume_thinking"],
        )
    elif summary["blanked_teacher_turn_rate"] > 0.05:
        logger.warning(
            "%.1f%% of teacher turns are empty after stripping — reasoning is "
            "overrunning the per-turn cap on those turns.",
            100 * summary["blanked_teacher_turn_rate"],
        )
    return summary


def print_summary(model: str, summary: dict[str, Any], args) -> None:
    s = summary
    print("=" * 72)
    print(f"Model:                        {model}")
    print(f"Teacher thinking:             {describe_teacher_thinking(args)}")
    print(f"Conversations:                {s['num_conversations']}"
          f"  ({s.get('num_problems', 0)} problems x "
          f"{s.get('samples_per_problem', 0):.0f} samples)")
    print("  (+/- is the standard error across PROBLEMS, not rollouts)")
    print(f"Teacher turns (avg):          {s['num_teacher_turns']:.2f}")
    print(f"EoC rate:                     {s['eoc_rate']:.2%}")
    print(f"Transfer-tested rate:         {s['transfer_tested_rate']:.2%}")
    print(f"assume_thinking:              {s.get('assume_thinking')}")
    print(f"Blanked teacher turns:        {s['blanked_teacher_turn_rate']:.2%}"
          f"{'   <-- INVALID RUN, see log' if s['blanked_teacher_turn_rate'] > 0.5 else ''}")
    print("Do the caps bind?")
    print(f"  thinking tok/turn:          avg {s['thinking_tokens_avg_per_turn']:.0f}  "
          f"p95 {s['thinking_tokens_p95_per_turn']}  max {s['thinking_tokens_max_per_turn']}"
          f"   (cap {args.max_tokens_per_turn} incl. visible)")
    print(f"  cut off by token budget:    {s['token_budget_truncated_rate']:.2%}"
          f"   (budget {args.max_conversation_tokens})")
    print("-" * 72)
    print("Near-transfer variant (what training rewards)")
    print(f"  pre_solve_rate:             {s['pre_solve_rate_transfer']:.3f}")
    print(f"  post_solve_rate:            {s['post_solve_transfer']:.3f}")
    print(f"  DELTA_TRANSFER:             {s['delta_transfer']:+.4f} "
          f"+/- {s.get('delta_transfer_stderr', 0.0):.4f}   "
          f"(raw {s['delta_transfer_raw']:+.4f}, gated {s['delta_transfer_gated']:+.4f})")
    print("Original problem (test_on_transfer=false's target)")
    print(f"  pre_solve_rate:             {s['pre_solve_rate']:.3f}")
    print(f"  post_solve_rate:            {s['post_solve_same']:.3f}")
    print(f"  DELTA_SAME:                 {s['delta_same']:+.4f} "
          f"+/- {s.get('delta_same_stderr', 0.0):.4f}   "
          f"(raw {s['delta_same_raw']:+.4f}, gated {s['delta_same_gated']:+.4f})")
    print(f"  transfer_gap (same−transfer): {s['transfer_gap']:+.4f}")
    print("-" * 72)
    print(f"LEAK_RATE:                    {s['leak_rate']:.2%} "
          f"+/- {s.get('leak_rate_stderr', 0.0):.2%}")
    print(f"Teaching-reject rate:         {s['teaching_reject_rate']:.2%}")
    print(f"Eval judge pedagogy (1-5):    {s['eval_judge_pedagogy']:.2f}")
    print(f"Eval judge correctness (1-5): {s['eval_judge_correctness']:.2f}")
    print(f"Eval judge parsed:            {s['eval_judge_ok']:.2%}")
    print(f"End reasons:                  {s['end_reasons']}")
    if s["n_student_calls_failed"]:
        print(f"Student calls failed:         {s['n_student_calls_failed']}")
    print("=" * 72)


# --- CLI ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Δ_transfer / Δ_same / leak_rate on test.parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Teacher (model under evaluation)
    p.add_argument("--model", required=True, help="HF hub id or checkpoint path")
    p.add_argument("--teacher-temperature", type=float, default=0.6,
                   help="Matches rollout.val_kwargs.temperature in configs/user.yaml")
    p.add_argument("--teacher-top-p", type=float, default=0.95,
                   help="Matches rollout.val_kwargs.top_p")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    # Must cover the system prompt plus the full conversation budget.
    p.add_argument("--max-model-len", type=int, default=20000)
    p.add_argument("--enable-thinking", action="store_true",
                   help="enable_thinking on the tutor's chat template")
    p.add_argument("--assume-thinking", choices=["auto", "yes", "no"], default="auto",
                   help="How to read a teacher turn with no <think>/</think> "
                        "marker. yes = overrun reasoning, blank it (correct for "
                        "a Qwen3.6-style prefill template, the training "
                        "default); no = it IS the visible reply (correct for "
                        "TutorRL, a Gemini tutor, or any non-prefill model); "
                        "auto = decide from the first teacher batch. Getting "
                        "this wrong blanks every teacher turn — watch "
                        "blanked_teacher_turn_rate")
    p.add_argument("--use-openrouter", action="store_true")
    p.add_argument("--use-gemini", action="store_true")
    p.add_argument("--teacher-api-base", default="https://openrouter.ai/api/v1")
    p.add_argument("--teacher-api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--teacher-concurrency", type=int, default=16)

    # Data (test.parquet is never used for checkpoint selection)
    p.add_argument(
        "--test-parquet", "--val-parquet",
        dest="test_parquet",
        default=os.path.join(
            os.environ.get("WORK_DIR", "."), "data", "eduardo", "test.parquet"
        ),
        help="Offline eval split from eduardo.process_dataset (test.parquet)",
    )
    p.add_argument("--num-problems", type=int, default=-1,
                   help="-1 for every row; otherwise the first N (no shuffle)")
    # Standard errors are clustered by problem (see the summary).
    p.add_argument("--num-samples-per-problem", type=int, default=1)
    # Seeds the vLLM engine, not each request (per-request seeds would make
    # repeated samples identical). Remote student/judge/API tutors are unseeded.
    p.add_argument("--seed", type=int, default=42,
                   help="vLLM engine seed for the tutor rollout")

    # Conversation limits — mirror configs/user.yaml.
    p.add_argument("--max-teacher-turns", type=int, default=11,
                   help="max_user_turns=10 + the opener ⇒ 11 teacher turns")
    p.add_argument("--min-teacher-turns-before-eoc", type=int, default=3,
                   help="eduardo_interaction_config.yaml")
    # Deliberately larger than the training caps (662 / 8192): caps that bind
    # unevenly across models would favour terser ones. The summary reports
    # whether they bind.
    p.add_argument("--max-tokens-per-turn", type=int, default=4192,
                   help="Hard per-turn cap (thinking + visible). Training uses "
                        "662; the default here is deliberately slack so the cap "
                        "does not bind for the longest-reasoning model in the "
                        "comparison")
    p.add_argument("--max-conversation-tokens", type=int, default=16384,
                   help="Whole-conversation budget. Training uses 8192 "
                        "(data.max_response_length); raised for the same reason")

    # Student simulator + final attempts
    p.add_argument("--student-model", default=os.environ.get(
        "STUDENT_MODEL_NAME", f"Llama-3.1-8B-Instruct-{os.environ.get('USER', '')}"))
    p.add_argument("--student-api-base", default=os.environ.get(
        "STUDENT_API_BASE", "http://localhost:8080/v1"))
    p.add_argument("--student-api-key-env", default="SERVING_API_KEY")
    p.add_argument("--student-temperature", type=float, default=0.6,
                   help="Dialogue turns (EduardoInteraction)")
    p.add_argument("--student-final-temperature", type=float, default=0.6,
                   help="Graded final attempts (reward_kwargs.student_temperature)")
    p.add_argument("--student-max-tokens", type=int, default=500)
    p.add_argument("--student-concurrency", type=int, default=64)
    p.add_argument("--num-student-attempts", type=int, default=16,
                   help="reward_kwargs.num_student_attempts; sampled twice per "
                        "conversation (transfer + original)")
    p.add_argument("--no-omit-teacher-turns", dest="omit_teacher_turns",
                   action="store_false",
                   help="Show the teacher's real words in the graded final "
                        "attempt (reward_kwargs.omit_teacher_turns=false). The "
                        "default blanks them, as training does")
    p.set_defaults(omit_teacher_turns=True)
    p.add_argument("--num-retries", type=int, default=3)

    # Leak / teaching judges (identical prompts + aggregation to training)
    p.add_argument("--judge-model", default=os.environ.get(
        "JUDGE_MODEL_NAME", f"Qwen3.6-27B-{os.environ.get('USER', '')}"))
    p.add_argument("--judge-api-base", default=os.environ.get(
        "JUDGE_API_BASE", "http://localhost:8080/v1"))
    p.add_argument("--judge-api-key-env", default="SERVING_API_KEY")
    p.add_argument("--judge-temperature", type=float, default=0.3)
    # Larger than training's 4096: Gemini thinking counts against this budget,
    # and a truncated verdict defaults to REJECT.
    p.add_argument("--judge-max-tokens", type=int, default=16384)
    p.add_argument("--judge-votes", type=int, default=3,
                   help="reward_kwargs.judge_votes")
    p.add_argument("--judge-concurrency", type=int, default=64)
    p.add_argument("--gemini", action="store_true",
                   help="Judge with Gemini instead of the judge server")
    p.add_argument("--gemini-model", default="gemini-3.1-pro-preview")
    p.add_argument("--gemini-reasoning-effort", default="high",
                   choices=["low", "medium", "high"])
    p.add_argument("--gemini-api-base",
                   default="https://generativelanguage.googleapis.com/v1beta/openai/")
    p.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")

    # 1-5 periodic-eval judge (the val-aux/* wandb curves)
    p.add_argument("--eval-judge-model", default="gemini-3.1-pro-preview",
                   help='"" to skip the 1-5 pedagogy/correctness scores')
    p.add_argument("--eval-judge-max-tokens", type=int, default=16384,
                   help="reward_kwargs.eval_judge_max_tokens")
    p.add_argument("--eval-judge-concurrency", type=int, default=16)

    # IO
    p.add_argument("--stage", choices=["all", "rollout", "score"], default="all")
    p.add_argument("--conversations", default=None,
                   help="Existing transfer_conversations_*.jsonl (--stage score)")
    p.add_argument("--output-dir", default=None,
                   help="Defaults to $WORK_DIR/eval_results or ./eval_results")
    p.add_argument("--tag", default=None,
                   help="Label for the output filenames; defaults to the "
                        "checkpoint's parent dir + step (the ablation arm name)")
    return p.parse_args()


def default_tag(model: str) -> str:
    """Output tag from a checkpoint path, e.g. ``…/ablation-4b-nogates/step_75``
    → ``ablation-4b-nogates_step_75``.
    """
    path = Path(model.rstrip("/"))
    if path.name.startswith("step_") and path.parent.name:
        return f"{path.parent.name}_{path.name}"
    return path.name


def main() -> None:
    args = parse_args()

    if not (args.use_openrouter or args.use_gemini) and args.model.startswith(("/", "./", "~")):
        model_path = Path(args.model).expanduser()
        if not model_path.is_dir():
            raise SystemExit(f"--model path does not exist: {model_path}")
        args.model = str(model_path)

    out_dir = Path(
        args.output_dir
        or (Path(os.environ["WORK_DIR"]) / "eval_results"
            if os.environ.get("WORK_DIR") else Path("eval_results"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or default_tag(args.model)

    # Rollout and scoring share one event loop: the cached student client and
    # semaphore are bound to the loop that created them.
    if args.stage in ("all", "rollout") and not Path(args.test_parquet).is_file():
        raise SystemExit(f"--test-parquet not found: {args.test_parquet}")
    conv_path = out_dir / f"transfer_conversations_{tag}_{ts}.jsonl"

    def write_conversations(records: list[dict[str, Any]]) -> None:
        with open(conv_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote conversations to %s", conv_path)

    async def run_stages() -> list[dict[str, Any]] | None:
        if args.stage == "score":
            if not args.conversations:
                raise SystemExit("--stage score requires --conversations <path.jsonl>")
            with open(args.conversations, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            logger.info("Loaded %d conversations from %s", len(records), args.conversations)
        else:
            rows = load_val_rows(args.test_parquet, args.num_problems)
            start = time.time()
            conversations = await transfer_rollout(args, rows)
            logger.info(
                "Rollout of %d conversations took %.1fs",
                len(conversations), time.time() - start,
            )
            records = []
            for conv in conversations:
                rec = conv.to_record()
                # Stored so --stage score can run without re-running the teacher.
                rec["reward_messages"] = conv.reward_view()
                rec["judge_messages"] = conv.judge_view()
                records.append(rec)
            write_conversations(records)
            if args.stage == "rollout":
                return None

        start = time.time()
        records = await score_records(records, args)
        logger.info("Scoring took %.1fs", time.time() - start)
        return records

    records = asyncio.run(run_stages())
    if records is None:
        logger.info(
            "Rollout-only stage done. Score with:\n"
            "  python -m eval.run_env_test_eval --model %s --stage score "
            "--conversations %s",
            args.model, conv_path,
        )
        return

    summary = compute_summary(records)
    results = {
        "model": args.model,
        "tag": tag,
        "benchmark": "transfer",
        "test_parquet": args.test_parquet,
        "teacher_thinking": describe_teacher_thinking(args),
        "judge_model": args.gemini_model if args.gemini else args.judge_model,
        "timestamp": ts,
        "args": vars(args),
        "metrics": summary,
        "conversations": records,
    }
    results_path = out_dir / f"transfer_eval_{tag}_{ts}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print_summary(args.model, summary, args)
    print(f"Results written to: {results_path}")


if __name__ == "__main__":
    main()
