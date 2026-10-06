"""Eduardo composite reward for veRL (``custom_reward_function``).

Soft mode:   R = Δ·d·d_think − judge_penalty·[REJECT] + r_eoc + r_think
Hard gate:   R = Δ·d·d_think + r_eoc + r_think  if judge OK, else −gate_penalty

Δ = (post − pre)/(1 − pre) if post > pre else post − pre, where pre/post are the
student's solve rates before/after tutoring, measured on a near-transfer variant
of the discussed problem when available. d, d_think are token-weighted length
decays on visible and thinking tokens, applied to positive Δ only.
Each rollout is also logged as a JSONL line (and a wandb table row when available).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics import correctness, judge_eval, judge_quality
from .utils import (
    get_async_openai_client,
    get_concurrency_semaphore,
    strip_teacher_controls,
)

logger = logging.getLogger(__name__)

# The chat template pre-fills "<think>\n", so a well-formed teacher turn
# arrives as "REASONING</think>VISIBLE" (a close tag with no opening one).
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>")
_EMPTY_THINK_RE = re.compile(r"<think>\s*</think>")


def _analyze_teacher_turn(
    text: str, enable_thinking: bool = True
) -> tuple[str, str, int]:
    """Split one teacher turn into ``(thinking, visible, n_violations)``.

    Violations: no ``</think>`` at all (truncated mid-thought), empty
    ``<think></think>`` blocks, and stray tags beyond the single expected close.
    With ``enable_thinking=False`` the whole turn is visible and never violates.
    """
    if not enable_thinking:
        return "", text, 0  # no thinking segment exists: all of it is visible

    if "</think>" not in text and "<think>" not in text:
        return text, "", 1  # truncated mid-thought: nothing visible

    violations = len(_EMPTY_THINK_RE.findall(text))
    thinking_parts = [b for b in _THINK_BLOCK_RE.findall(text) if b.strip()]
    rest = _THINK_BLOCK_RE.sub("", text)

    if "</think>" in rest:
        # Prefill form: everything before the first close is thinking.
        head, _, rest = rest.partition("</think>")
        if head.strip():
            thinking_parts.append(head)

    if "<think>" in rest:
        # Reopened but never closed: the tail is truncated thinking.
        rest, _, tail = rest.partition("<think>")
        violations += 1 + len(_THINK_TAG_RE.findall(tail))
        tail = _THINK_TAG_RE.sub("", tail)
        if tail.strip():
            thinking_parts.append(tail)

    n_stray_closes = rest.count("</think>")
    if n_stray_closes:
        violations += n_stray_closes
        rest = rest.replace("</think>", "")

    return "\n".join(thinking_parts), rest, violations


def _is_substantive_visible(visible: str) -> bool:
    """True when the visible part contains more than ``<end_of_conversation>``."""
    return bool(visible.replace("<end_of_conversation>", "").strip())


# Conversation sinks (JSONL + best-effort wandb.Table). Module-level state is
# safe because each Ray reward worker runs in its own process.
_SINK_LOCK = threading.Lock()
_JSONL_FILE = None
_JSONL_PATH: Path | None = None
_JSONL_DISABLED = False
_WANDB_MISSING_LOGGED = False

# Batch-level statistics for periodic summary logging.
_BATCH_LOCK = threading.Lock()
_BATCH_LOG_INTERVAL = 16  # rollouts per summary line
_BATCH_STATS: dict[str, Any] = {
    "rollouts": [],  # (score, row) tuples
    "count": 0,
    "start_time": None,
}


def _conv_log_dir() -> Path | None:
    """Resolve the directory for conversation JSONL. Returns None to disable."""
    override = os.environ.get("EDUARDO_CONV_DIR")
    if override:
        return Path(override)
    work_dir = os.environ.get("WORK_DIR")
    if work_dir:
        return Path(work_dir) / "eduardo_conversations"
    return None


def _get_jsonl_file():
    global _JSONL_FILE, _JSONL_PATH, _JSONL_DISABLED
    if _JSONL_DISABLED:
        return None
    if _JSONL_FILE is not None:
        return _JSONL_FILE
    with _SINK_LOCK:
        if _JSONL_FILE is not None:
            return _JSONL_FILE
        if _JSONL_DISABLED:
            return None
        base = _conv_log_dir()
        if base is None:
            _JSONL_DISABLED = True
            logger.info(
                "eduardo_reward_func: no WORK_DIR/EDUARDO_CONV_DIR env var; "
                "skipping per-rollout JSONL conversation log"
            )
            return None
        try:
            base.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("WANDB_RUN_ID") or f"pid{os.getpid()}"
            _JSONL_PATH = base / f"conversations_{ts}_{job_id}.jsonl"
            # Line-buffered so tailing the file shows live rollouts.
            _JSONL_FILE = open(_JSONL_PATH, "a", buffering=1, encoding="utf-8")
            logger.info(
                "eduardo_reward_func: writing rollout conversations to %s",
                _JSONL_PATH,
            )
        except Exception as exc:
            logger.warning(
                "eduardo_reward_func: could not open conversation JSONL (%s); "
                "disabling file sink",
                exc,
            )
            _JSONL_DISABLED = True
            return None
    return _JSONL_FILE


def _dump_jsonl(row: dict[str, Any]) -> None:
    f = _get_jsonl_file()
    if f is None:
        return
    try:
        with _SINK_LOCK:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("eduardo_reward_func: JSONL write failed: %s", exc)


def _log_wandb(row: dict[str, Any]) -> None:
    """Append one row to a ``eduardo/conversations`` wandb table if wandb is live in this process."""
    global _WANDB_MISSING_LOGGED
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        if not _WANDB_MISSING_LOGGED:
            logger.info(
                "eduardo_reward_func: wandb.run is None in this process; "
                "skipping wandb conversation log (JSONL file is still written)"
            )
            _WANDB_MISSING_LOGGED = True
        return
    try:
        table = wandb.Table(
            columns=list(row.keys()),
            data=[[row[k] for k in row.keys()]],
        )
        # Piggyback on the trainer's next wandb step.
        wandb.log({"eduardo/conversations": table}, commit=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("eduardo_reward_func: wandb.log failed: %s", exc)


def _format_transcript(msgs: list[dict[str, Any]], max_chars: int = 800) -> str:
    """Compact human-readable transcript for JSONL / wandb / stdout."""
    out = []
    for m in msgs:
        role = (m.get("role") or "?").upper()
        content = (m.get("content") or "").replace("\r", "").strip()
        out.append(f"[{role}] {content}")
    text = "\n".join(out)
    if len(text) > max_chars:
        text = text[:max_chars] + f"…(+{len(text) - max_chars} chars)"
    return text


def _preview(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def _normalized_delta(post: float, pre: float) -> float:
    """Fraction of the remaining gap closed, (post − pre) / (1 − pre).

    Regressions return the raw difference (floor −pre), so failures are not
    over-penalised on high-baseline problems. Range: [−1, 1].
    """
    if post <= pre:
        return post - pre
    return (post - pre) / (1.0 - pre)


def _pre_rate(extra_info: dict[str, Any], *keys: str) -> float:
    """First present (not merely truthy) solve-rate field among ``keys``, else 0.0."""
    for k in keys:
        v = extra_info.get(k)
        if v is not None:
            return float(v)
    return 0.0


# Batch-level logging


def _accumulate_batch_stats(row: dict[str, Any]) -> None:
    """Accumulate rollout into batch statistics for periodic summary logging."""
    global _BATCH_STATS
    with _BATCH_LOCK:
        if _BATCH_STATS["start_time"] is None:
            _BATCH_STATS["start_time"] = time.time()
        # Drop the full transcript to bound memory until the batch flush.
        metrics_only = {k: v for k, v in row.items() if k != "messages"}
        _BATCH_STATS["rollouts"].append((row.get("score", 0.0), metrics_only))
        _BATCH_STATS["count"] += 1

        if _BATCH_STATS["count"] >= _BATCH_LOG_INTERVAL:
            _log_batch_summary()
            _reset_batch_stats()


def _reset_batch_stats() -> None:
    """Reset batch accumulator for next batch."""
    global _BATCH_STATS
    _BATCH_STATS = {
        "rollouts": [],
        "count": 0,
        "start_time": None,
    }


def _log_batch_summary() -> None:
    """Log batch-level summary metrics."""
    rollouts = _BATCH_STATS["rollouts"]
    if not rollouts:
        return

    n = len(rollouts)
    elapsed = time.time() - (_BATCH_STATS["start_time"] or time.time())

    scores = [r[0] for r in rollouts]
    rows = [r[1] for r in rollouts]

    def mean(key: str) -> float:
        vals = [r.get(key, 0.0) for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    def count_true(key: str) -> int:
        return sum(1 for r in rows if r.get(key))

    metrics = {
        "reward/mean": sum(scores) / n,
        "reward/min": min(scores),
        "reward/max": max(scores),
        "reward/correctness": mean("r_correctness"),
        "reward/judge_penalty": mean("r_judge"),
        "reward/gate_penalty": mean("r_gate"),
        "reward/gated_rate": mean("gated"),
        "reward/length_penalty": mean("r_length"),
        "reward/think_length_penalty": mean("r_think_length"),
        "reward/eoc_bonus": mean("r_eoc"),
        "reward/think_reward": mean("r_think"),
        "conversation/correct_think_rate": mean("correct_think_rate"),
        "accuracy/student": mean("post_solve_rate"),
        "accuracy/judge_pass": mean("judge_pass"),
        "accuracy/eoc_rate": mean("eoc_used"),
        # pre_solve_rate is the tested (transfer) problem's baseline;
        # pre_solve_rate_original is the tutored problem's.
        "accuracy/delta_solve_rate": mean("delta_solve"),
        "accuracy/delta_solve_rate_raw": mean("delta_solve_raw"),
        "accuracy/pre_solve_rate": mean("pre_solve_rate"),
        "accuracy/pre_solve_rate_original": mean("pre_solve_rate_original"),
        "accuracy/post_solve_rate": mean("post_solve_rate"),
        "accuracy/transfer_tested_rate": mean("used_transfer_problem"),
        "judge/no_leak_pass": mean("judge_no_leak_ok"),
        "judge/leak_rate": 1.0 - mean("judge_no_leak_ok"),
        "judge/teaching_quality": mean("judge_teaching_ok"),
        "judge/leak_rate_early": mean("leak_rate_early"),
        "conversation/mean_turns": mean("num_turns"),
        "conversation/mean_teacher_tokens": mean("teacher_tokens_mean"),
        "conversation/mean_thinking_tokens": mean("thinking_tokens_mean"),
        "conversation/mean_visible_tokens": mean("visible_tokens_mean"),
        "conversation/mean_student_token_share": mean("student_token_share"),
        "conversation/mean_student_tokens": mean("student_tokens_total"),
    }

    logger.info(
        "═══ BATCH SUMMARY (%d rollouts, %.1fs) ═══\n"
        "  reward/mean=%.3f  reward/min=%.3f  reward/max=%.3f\n"
        "  reward/correctness=%.3f  reward/judge_penalty=%.3f  reward/gate_penalty=%.3f (gated=%.3f)  reward/length_penalty=%.3f  reward/think_length_penalty=%.3f  reward/eoc_bonus=%.3f  reward/think_reward=%.3f (correct_think_rate=%.2f)\n"
        "  accuracy/student=%.3f  accuracy/judge_pass=%.3f  accuracy/eoc_rate=%.3f\n"
        "  accuracy/delta_solve_rate=%.3f (raw=%.3f)  (pre=%.3f → post=%.3f)  [pre_orig=%.3f  transfer_tested=%.2f]\n"
        "  judge/no_leak_pass=%.3f  judge/leak_rate=%.3f  judge/teaching_quality=%.3f  judge/leak_rate_early=%.3f\n"
        "  conversation/mean_turns=%.1f  conversation/mean_teacher_tokens=%.0f (think=%.0f, visible=%.0f)\n"
        "  conversation/mean_student_token_share=%.2f (student_tokens=%.0f)",
        n, elapsed,
        metrics["reward/mean"], metrics["reward/min"], metrics["reward/max"],
        metrics["reward/correctness"], metrics["reward/judge_penalty"],
        metrics["reward/gate_penalty"], metrics["reward/gated_rate"],
        metrics["reward/length_penalty"], metrics["reward/think_length_penalty"],
        metrics["reward/eoc_bonus"],
        metrics["reward/think_reward"], metrics["conversation/correct_think_rate"],
        metrics["accuracy/student"], metrics["accuracy/judge_pass"], metrics["accuracy/eoc_rate"],
        metrics["accuracy/delta_solve_rate"], metrics["accuracy/delta_solve_rate_raw"],
        metrics["accuracy/pre_solve_rate"],
        metrics["accuracy/post_solve_rate"],
        metrics["accuracy/pre_solve_rate_original"],
        metrics["accuracy/transfer_tested_rate"],
        metrics["judge/no_leak_pass"], metrics["judge/leak_rate"],
        metrics["judge/teaching_quality"], metrics["judge/leak_rate_early"],
        metrics["conversation/mean_turns"], metrics["conversation/mean_teacher_tokens"],
        metrics["conversation/mean_thinking_tokens"], metrics["conversation/mean_visible_tokens"],
        metrics["conversation/mean_student_token_share"],
        metrics["conversation/mean_student_tokens"],
    )
    
    sorted_rollouts = sorted(rollouts, key=lambda x: x[0])
    _log_sample("Worst", sorted_rollouts[0][1])
    _log_sample("Best", sorted_rollouts[-1][1])


def _log_sample(label: str, row: dict[str, Any]) -> None:
    """Log a single sample with a transcript preview."""
    problem = row.get("problem", "")[:80]
    score = row.get("score", 0.0)

    logger.info(
        "\n  %s sample (reward=%.3f, problem: %s...)",
        label, score, problem
    )

    transcript = row.get("transcript", "")
    if transcript:
        for line in transcript.split("\n")[:6]:
            logger.info("    %s", line[:140])

    logger.info(
        "    Judge: no_leak=%s, teaching=%s",
        row.get("judge_no_leak", "?"),
        row.get("judge_teaching", "?"),
    )
    logger.info(
        "    Δsolve: %.3f (pre=%.2f → post=%.2f)  eoc=%d",
        row.get("delta_solve", 0.0),
        row.get("pre_solve_rate", 0.0),
        row.get("post_solve_rate", 0.0),
        row.get("eoc_used", 0),
    )


# Checkpoint / validation logging (called from veRL trainer callbacks)


def log_checkpoint_saved(
    step: int,
    path: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Log a checkpoint save event, optionally with current metrics."""
    logger.info(
        "\n"
        "════════════════════════════════════════════════════════════════════════════════\n"
        "  CHECKPOINT SAVED: step=%d  path=%s\n"
        "════════════════════════════════════════════════════════════════════════════════",
        step, path,
    )
    if metrics:
        logger.info(
            "  Checkpoint metrics:\n"
            "    reward/mean=%.3f  accuracy/student=%.3f  accuracy/judge_pass=%.3f\n"
            "    accuracy/delta_solve_rate=%.3f  judge/leak_rate=%.3f\n"
            "    conversation/mean_turns=%.1f",
            metrics.get("reward/mean", metrics.get("critic/score/mean", 0.0)),
            metrics.get("accuracy/student", metrics.get("acc", 0.0)),
            metrics.get("accuracy/judge_pass", metrics.get("judge_pass", 0.0)),
            metrics.get("accuracy/delta_solve_rate", metrics.get("delta_solve", 0.0)),
            metrics.get("judge/leak_rate", 1.0 - metrics.get("judge_no_leak_ok", 0.0)),
            metrics.get("conversation/mean_turns", metrics.get("num_turns", 0.0)),
        )


def log_validation_results(
    step: int,
    metrics: dict[str, Any],
    label: str = "VALIDATION",
) -> None:
    """Log validation/test results under ``label``."""
    logger.info(
        "\n"
        "────────────────────────────────────────────────────────────────────────────────\n"
        "  %s RESULTS: step=%d\n"
        "────────────────────────────────────────────────────────────────────────────────\n"
        "    reward/mean=%.3f  reward/min=%.3f  reward/max=%.3f\n"
        "    accuracy/student=%.3f  accuracy/judge_pass=%.3f  accuracy/eoc_rate=%.3f\n"
        "    accuracy/delta_solve_rate=%.3f  (pre=%.3f → post=%.3f)\n"
        "    judge/leak_rate=%.3f  judge/leak_rate_early=%.3f\n"
        "    conversation/mean_turns=%.1f  conversation/mean_teacher_tokens=%.0f\n"
        "────────────────────────────────────────────────────────────────────────────────",
        label, step,
        metrics.get("reward/mean", metrics.get("critic/score/mean", 0.0)),
        metrics.get("reward/min", metrics.get("critic/score/min", 0.0)),
        metrics.get("reward/max", metrics.get("critic/score/max", 0.0)),
        metrics.get("accuracy/student", metrics.get("acc", 0.0)),
        metrics.get("accuracy/judge_pass", metrics.get("judge_pass", 0.0)),
        metrics.get("accuracy/eoc_rate", metrics.get("eoc_used", 0.0)),
        metrics.get("accuracy/delta_solve_rate", metrics.get("delta_solve", 0.0)),
        metrics.get("accuracy/pre_solve_rate", metrics.get("pre_solve_rate", 0.0)),
        metrics.get("accuracy/post_solve_rate", metrics.get("post_solve_rate", 0.0)),
        metrics.get("judge/leak_rate", 1.0 - metrics.get("judge_no_leak_ok", 0.0)),
        metrics.get("judge/leak_rate_early", metrics.get("leak_rate_early", 0.0)),
        metrics.get("conversation/mean_turns", metrics.get("num_turns", 0.0)),
        metrics.get("conversation/mean_teacher_tokens", metrics.get("teacher_tokens_mean", 0.0)),
    )


# Periodic-eval judge (validation rollouts only)

# Gemini's OpenAI-compatible endpoint.
_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# veRL batches reward-extra keys per key, so every rollout must report them
# (zeros when not judged); ``eval_judge_ok`` marks rollouts that got both scores.
_EVAL_JUDGE_BLANK = {
    "eval_judge_pedagogy": 0.0,
    "eval_judge_correctness": 0.0,
    "eval_judge_ok": 0,
}

_EVAL_JUDGE_SKIP_LOGGED = False


def _log_eval_judge_skip(reason: str) -> None:
    global _EVAL_JUDGE_SKIP_LOGGED
    if not _EVAL_JUDGE_SKIP_LOGGED:
        _EVAL_JUDGE_SKIP_LOGGED = True
        logger.info("eduardo_reward_func: periodic-eval judge disabled — %s", reason)


async def _maybe_eval_judge(
    msgs: list[dict[str, Any]],
    problem: str,
    answer: str,
    extra_info: dict[str, Any],
    kwargs: dict[str, Any],
    retry_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Score a validation rollout 1-5 on pedagogy and factual correctness.

    Diagnostic only (never part of ``score``); training rollouts get zeros.
    """
    if str(extra_info.get("split", "")).lower() not in ("val", "validation", "test"):
        return dict(_EVAL_JUDGE_BLANK)

    model = str(kwargs.get("eval_judge_model", "") or "")
    key_env = str(kwargs.get("eval_judge_api_key_env", "GEMINI_API_KEY"))
    if not model:
        _log_eval_judge_skip("no eval_judge_model in reward_kwargs")
        return dict(_EVAL_JUDGE_BLANK)
    if not os.environ.get(key_env):
        _log_eval_judge_skip(f"${key_env} is not set")
        return dict(_EVAL_JUDGE_BLANK)

    try:
        scores = await judge_eval.score_conversation(
            messages=msgs,
            problem=problem,
            answer=answer,
            client=get_async_openai_client(
                str(kwargs.get("eval_judge_api_base", _GEMINI_API_BASE)), key_env
            ),
            model=model,
            temperature=float(kwargs.get("eval_judge_temperature", 0.3)),
            max_tokens=int(kwargs.get("eval_judge_max_tokens", 16384)),
            # Maps to Gemini's thinking_level on the OpenAI-compat endpoint.
            extra_body={
                "reasoning_effort": str(kwargs.get("eval_judge_reasoning_effort", "high"))
            },
            retry_kwargs=retry_kwargs,
            semaphore=get_concurrency_semaphore(
                "eval_judge", int(kwargs.get("eval_judge_concurrency", 32))
            ),
        )
    except Exception as exc:  # never let a diagnostic break validation
        logger.warning("eduardo_reward_func: periodic-eval judge failed: %s", exc)
        return dict(_EVAL_JUDGE_BLANK)

    pedagogy, correctness_score = scores["pedagogy"], scores["correctness"]
    return {
        "eval_judge_pedagogy": float(pedagogy or 0.0),
        "eval_judge_correctness": float(correctness_score or 0.0),
        "eval_judge_ok": int(pedagogy is not None and correctness_score is not None),
    }


# Reward entry point


async def eduardo_reward_func(
    data_source: str,
    solution_str: str | list[dict] | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    *,
    messages: list[dict] | None = None,
    tokenizer: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """veRL-compatible async reward function.

    Main ``reward_kwargs``: judge/student endpoints and sampling settings,
    ``judge_penalty``, ``judge_hard_gate``/``gate_penalty``, length budgets
    ``max_tokens_per_turn``/``max_thinking_tokens_per_turn``, bonuses
    ``eoc_bonus``, ``think_reward``, and
    ``test_on_transfer``, ``omit_teacher_turns``, ``enable_thinking``.
    ``eval_judge_*`` enables a diagnostic-only judge on validation rollouts.
    """
    if extra_info is None:
        extra_info = {}
    problem: str = extra_info.get("problem", "")
    if ground_truth is None:
        ground_truth = extra_info.get("answer", "") or ""

    # Priority: ``messages`` kwarg, ``extra_info["messages"]`` (set by
    # EduardoAgentLoop), then ``solution_str`` if already a message list.
    msgs: list[dict] = messages or extra_info.get("messages") or []
    if not msgs and isinstance(solution_str, list):
        msgs = solution_str

    num_assistant_turns_hint = extra_info.get("num_assistant_turns")
    num_user_turns_hint = extra_info.get("num_user_turns")

    if not msgs:
        ei_keys = list(extra_info.keys()) if isinstance(extra_info, dict) else None
        logger.warning(
            "eduardo_reward_func: empty transcript; returning zero reward. "
            "data_source=%r problem=%r extra_info.keys=%s turns=(A=%s,U=%s)",
            data_source,
            _preview(problem, 80),
            ei_keys,
            num_assistant_turns_hint,
            num_user_turns_hint,
        )
        empty = _empty_reward()
        _emit_sinks(
            row=_conversation_row(
                data_source=data_source,
                problem=problem,
                ground_truth=ground_truth,
                msgs=[],
                reward=empty,
                extra_info=extra_info,
                note="empty_transcript",
                run_meta=_run_meta(kwargs),
            )
        )
        return empty

    retry_kwargs = {
        "num_retries": kwargs.get("num_retries", 3),
        "backoff_base": kwargs.get("backoff_base", 2.0),
        "backoff_cap": kwargs.get("backoff_cap", 30.0),
        "jitter": kwargs.get("jitter", True),
    }

    # The "student" semaphore is shared with EduardoInteraction, bounding total
    # load on the student server; 0 disables gating.
    student_semaphore = get_concurrency_semaphore(
        "student", int(kwargs.get("student_concurrency", 64))
    )
    judge_semaphore = get_concurrency_semaphore(
        "judge", int(kwargs.get("judge_concurrency", 128))
    )

    judge_client = get_async_openai_client(
        kwargs["judge_api_base"], kwargs["judge_api_key_env"]
    )
    student_client = get_async_openai_client(
        kwargs["student_api_base"], kwargs["student_api_key_env"]
    )

    # The student is tested on an unseen near-transfer variant, so a leaked
    # answer to `problem` earns nothing. Falls back to `problem` when absent
    # or when test_on_transfer is false.
    test_on_transfer = bool(kwargs.get("test_on_transfer", True))
    transfer_problem: str = extra_info.get("transfer_problem") or ""
    transfer_answer: str = extra_info.get("transfer_answer") or ""
    use_transfer = test_on_transfer and bool(transfer_problem and transfer_answer)
    if test_on_transfer and transfer_problem and not transfer_answer:
        logger.warning(
            "eduardo_reward_func: transfer_problem present but transfer_answer "
            "is empty; falling back to testing on the original problem. "
            "problem=%r",
            _preview(problem, 80),
        )
    test_problem = transfer_problem if use_transfer else problem
    test_answer = transfer_answer if use_transfer else ground_truth

    # Untutored baseline on the tested problem.
    pre_solve_rate_original = _pre_rate(
        extra_info, "pre_solve_rate", "solve_rate", "llama8b_solve_rate"
    )
    if use_transfer:
        pre_solve_rate = _pre_rate(
            extra_info, "pre_solve_rate_transfer_problem"
        )
    else:
        pre_solve_rate = pre_solve_rate_original

    # Judge before sampling final attempts (rejected rollouts skip them). The
    # judge sees the original problem and the student-visible transcript,
    # with teacher <think> blocks stripped.
    judge_msgs = [
        {**m, "content": strip_teacher_controls(m.get("content") or "")}
        if m.get("role") == "assistant"
        else m
        for m in msgs
    ]
    # The independent eval judge rides along in the same gather.
    (
        (no_leak, teaching, judge_reasoning),
        eval_judge_metrics,
    ) = await asyncio.gather(
        judge_quality.majority_vote(
            messages=judge_msgs,
            problem=problem,
            answer=ground_truth,
            client=judge_client,
            model=kwargs["judge_model"],
            num_votes=kwargs.get("judge_votes", 3),
            temperature=kwargs.get("judge_temperature", 0.3),
            max_tokens=int(kwargs.get("judge_max_tokens", 2048)),
            retry_kwargs=retry_kwargs,
            semaphore=judge_semaphore,
        ),
        _maybe_eval_judge(
            judge_msgs, problem, ground_truth, extra_info, kwargs, retry_kwargs
        ),
    )

    hard_gate = bool(kwargs.get("judge_hard_gate", False))
    pedagogy_violated = (no_leak == "REJECT") or (teaching == "REJECT")
    gated = hard_gate and pedagogy_violated

    # Soft mode only: a flat penalty on rejection. In hard-gate mode it would
    # cancel under mean-centering when a group's verdicts are uniform.
    judge_pen = float(kwargs.get("judge_penalty", 0.75))
    r_judge = -judge_pen if (pedagogy_violated and not hard_gate) else 0.0

    # Paid only by gated rollouts (Δ = 0), so a caught leak scores strictly
    # below an honest 0/n rollout and survives mean-centering in mixed groups.
    gate_pen = float(kwargs.get("gate_penalty", 0.35))
    r_gate = -gate_pen if gated else 0.0

    # When true, teacher utterances are replaced with "(hidden)" in the graded
    # final attempt only, so the teacher scores through what the student wrote.
    omit_teacher_turns = bool(kwargs.get("omit_teacher_turns", True))

    if gated:
        correct_fraction, n_format, n_failed = 0.0, 0, 0
        # No post_solve was measured; the flat r_gate is the whole deterrent.
        delta_solve = 0.0
    else:
        correct_fraction, n_format, n_failed = await correctness.compute_fraction(
            messages=msgs,
            ground_truth=test_answer,
            # The dialogue is about the original problem; only the final
            # turn switches to the variant.
            problem=problem,
            transfer_problem=transfer_problem if use_transfer else None,
            client=student_client,
            model=kwargs["student_model"],
            num_attempts=kwargs.get("num_student_attempts", 8),
            temperature=kwargs.get("student_temperature", 0.6),
            max_tokens=800,
            retry_kwargs=retry_kwargs,
            semaphore=student_semaphore,
            omit_teacher_turns=omit_teacher_turns,
        )
        # Improvement over the untutored baseline, normalised by headroom.
        delta_solve = _normalized_delta(float(correct_fraction), pre_solve_rate)
    # Un-normalised gain, logged only.
    delta_solve_raw = 0.0 if gated else float(correct_fraction) - pre_solve_rate
    r_correct = delta_solve

    # Teacher turn stats
    teacher_contents = [
        (m.get("content") or "") for m in msgs if m.get("role") == "assistant"
    ]
    num_turns = len(teacher_contents)
    # Thinking and visible tokens get separate budgets and decays.
    enable_thinking = bool(kwargs.get("enable_thinking", True))
    turn_analyses = [
        _analyze_teacher_turn(c, enable_thinking) for c in teacher_contents
    ]
    thinking_token_counts = _count_tokens_per_turn(
        [thinking for thinking, _, _ in turn_analyses], tokenizer
    )
    visible_token_counts = _count_tokens_per_turn(
        [visible for _, visible, _ in turn_analyses], tokenizer
    )
    thinking_tokens_total = sum(thinking_token_counts)
    visible_tokens_total = sum(visible_token_counts)
    teacher_tokens_total = thinking_tokens_total + visible_tokens_total
    teacher_tokens_mean = (
        teacher_tokens_total / num_turns if num_turns > 0 else 0.0
    )
    thinking_tokens_mean = (
        thinking_tokens_total / num_turns if num_turns > 0 else 0.0
    )
    visible_tokens_mean = (
        visible_tokens_total / num_turns if num_turns > 0 else 0.0
    )

    # Diagnostic: student / (student + teacher visible) tokens. Excludes teacher
    # thinking and the scripted opening (the only user turn before the first
    # teacher turn), which quotes the problem.
    student_contents: list[str] = []
    seen_teacher = False
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            seen_teacher = True
        elif role == "user" and seen_teacher:
            student_contents.append(m.get("content") or "")
    student_tokens_total = sum(_count_tokens_per_turn(student_contents, tokenizer))
    _shared_tokens = student_tokens_total + visible_tokens_total
    student_token_share = (
        student_tokens_total / _shared_tokens if _shared_tokens > 0 else 0.0
    )

    # Token-weighted rational decay, d = Σ t_i·d_i / Σ t_i with
    # d_i = 1 / (1 + ((t_i − T0)/T0)^k) for t_i > T0; token-weighting stops short
    # filler turns from buying budget for long ones. Applied to positive Δ only.
    target_tokens = float(kwargs.get("max_tokens_per_turn", 300))
    think_target_tokens = float(
        kwargs.get("max_thinking_tokens_per_turn", target_tokens)
    )
    length_decay = _token_weighted_decay(visible_token_counts, target_tokens)
    thinking_decay = _token_weighted_decay(
        thinking_token_counts, think_target_tokens
    )
    # Decays are logged as additive terms (≤ 0) that telescope to Δ·d·d_think
    # for Δ > 0 and are 0 otherwise.
    if delta_solve > 0.0:
        r_length = delta_solve * (length_decay - 1.0)
        r_think_length = delta_solve * length_decay * (thinking_decay - 1.0)
    else:
        r_length = 0.0
        r_think_length = 0.0

    eoc_used = any(
        "<end_of_conversation>" in (m.get("content") or "")
        and m.get("role") == "assistant"
        for m in msgs
    )
    # Only when Δ > 0; otherwise ending on turn 1 would be free reward.
    r_eoc = float(kwargs.get("eoc_bonus", 0.1)) if (eoc_used and delta_solve > 0.0) else 0.0



    # Fraction of turns with well-formed, non-empty thinking (bare EoC turns
    # count as correct). Not paid on gated rollouts.
    n_think_violations = sum(v for _, _, v in turn_analyses)
    n_missing_think = sum(
        1
        for thinking, visible, _ in turn_analyses
        if not thinking.strip() and _is_substantive_visible(visible)
    )
    n_correct_think_turns = sum(
        1
        for thinking, visible, violations in turn_analyses
        if violations == 0
        and (bool(thinking.strip()) or not _is_substantive_visible(visible))
    )
    correct_think_rate = (
        n_correct_think_turns / num_turns if num_turns > 0 else 0.0
    )
    think_rew = float(kwargs.get("think_reward", 0.1))
    r_think = 0.0 if gated else think_rew * correct_think_rate

    total = (
        r_correct
        + r_judge
        + r_gate
        + r_length
        + r_think_length
        + r_eoc
        + r_think
    )

    judge_pass = (no_leak == "OK") and (teaching == "OK")
    leak_rate_early = (no_leak == "REJECT") and (num_turns <= 1)

    result = {
        # Numeric entries are batch-averaged by veRL via reward_extra_info.
        "score": float(total),
        "acc": float(correct_fraction),
        "r_correctness": float(r_correct),  # undiscounted delta_solve
        "r_judge": float(r_judge),
        "r_gate": float(r_gate),
        "gated": int(gated),
        "r_length": float(r_length),
        "length_decay": float(length_decay),
        "r_think_length": float(r_think_length),
        "thinking_decay": float(thinking_decay),
        "r_eoc": float(r_eoc),
        "student_token_share": float(student_token_share),
        "student_tokens_total": int(student_tokens_total),
        "r_think": float(r_think),
        "correct_think_rate": float(correct_think_rate),
        "n_correct_think_turns": int(n_correct_think_turns),
        "n_think_violations": int(n_think_violations),
        "n_missing_think": int(n_missing_think),
        "pre_solve_rate": float(pre_solve_rate),
        "post_solve_rate": float(correct_fraction),
        "delta_solve": float(delta_solve),  # headroom-normalised
        "delta_solve_raw": float(delta_solve_raw),  # post − pre
        "used_transfer_problem": int(use_transfer),
        "pre_solve_rate_original": float(pre_solve_rate_original),
        "omit_teacher_turns": int(omit_teacher_turns),
        "student_n_incorrect_format": int(n_format),
        "student_n_failed": int(n_failed),
        "judge_no_leak_ok": int(no_leak == "OK"),
        "leak_rate": int(no_leak == "REJECT"),
        "judge_teaching_ok": int(teaching == "OK"),
        "num_turns": int(num_turns),
        "teacher_tokens_total": int(teacher_tokens_total),
        "teacher_tokens_mean": float(teacher_tokens_mean),
        "thinking_tokens_total": int(thinking_tokens_total),
        "thinking_tokens_mean": float(thinking_tokens_mean),
        "visible_tokens_mean": float(visible_tokens_mean),
        "eoc_used": int(eoc_used),
        "judge_pass": int(judge_pass),
        "leak_rate_early": int(leak_rate_early),
        **eval_judge_metrics,
        # Non-numeric debug fields (ignored by veRL's metric logger).
        "judge_no_leak": no_leak,
        "judge_teaching": teaching,
        "judge_no_leak_reasoning": judge_reasoning.get("no_leak", ""),
        "judge_teaching_reasoning": judge_reasoning.get("teaching", ""),
    }

    _emit_sinks(
        row=_conversation_row(
            data_source=data_source,
            problem=problem,
            ground_truth=ground_truth,
            msgs=msgs,
            reward=result,
            extra_info=extra_info,
            note="ok",
            transfer_problem=transfer_problem,
            transfer_answer=transfer_answer,
            run_meta=_run_meta(kwargs),
        )
    )

    return result


# Run identity (stamped into every exported conversation row)

_RUN_META: dict[str, str] | None = None


def _run_meta(kwargs: dict[str, Any]) -> dict[str, str]:
    """Best-effort run name, id and model path, cached per process.

    Resolved from ``reward_kwargs``, then a live ``wandb.run``, then env vars;
    unresolvable fields stay ``""``.
    """
    global _RUN_META
    if _RUN_META is not None:
        return _RUN_META

    def _first(*vals: Any) -> str:
        for v in vals:
            s = str(v or "").strip()
            if s:
                return s
        return ""

    wandb_name = wandb_id = ""
    try:
        import wandb

        if wandb.run is not None:
            wandb_name = wandb.run.name or ""
            wandb_id = wandb.run.id or ""
    except Exception:  # pragma: no cover - wandb absent / not initialised
        pass

    env = os.environ.get
    _RUN_META = {
        "run_name": _first(
            kwargs.get("run_name"), wandb_name, env("WANDB_NAME"), env("EXPERIMENT")
        ),
        "run_id": _first(wandb_id, env("WANDB_RUN_ID")),
        "model_path": _first(kwargs.get("model_path"), env("MODEL_PATH")),
    }
    return _RUN_META


def _conversation_row(
    *,
    data_source: str,
    problem: str,
    ground_truth: str,
    msgs: list[dict[str, Any]],
    reward: dict[str, Any],
    extra_info: dict[str, Any],
    note: str,
    transfer_problem: str = "",
    transfer_answer: str = "",
    run_meta: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a flat dict describing one rollout. Used for JSONL + wandb."""
    run_meta = run_meta or {}
    return {
        "ts": time.time(),
        "uuid": uuid.uuid4().hex,
        "pid": os.getpid(),
        "run_name": run_meta.get("run_name", ""),
        "run_id": run_meta.get("run_id", ""),
        "model_path": run_meta.get("model_path", ""),
        "note": note,
        "data_source": data_source,
        "problem": problem,
        "ground_truth": ground_truth,
        # Empty ⇒ the student was tested on `problem` itself.
        "transfer_problem": transfer_problem,
        "transfer_answer": transfer_answer,
        "used_transfer_problem": reward.get("used_transfer_problem", 0),
        "omit_teacher_turns": reward.get("omit_teacher_turns", 0),
        "num_messages": len(msgs),
        "num_turns": reward.get("num_turns", 0),
        "teacher_tokens_total": reward.get("teacher_tokens_total", 0),
        "thinking_tokens_total": reward.get("thinking_tokens_total", 0),
        "eoc_used": reward.get("eoc_used", 0),
        "judge_no_leak": reward.get("judge_no_leak", ""),
        "judge_teaching": reward.get("judge_teaching", ""),
        "judge_pass": reward.get("judge_pass", 0),
        "leak_rate_early": reward.get("leak_rate_early", 0),
        "eval_judge_pedagogy": reward.get("eval_judge_pedagogy", 0.0),
        "eval_judge_correctness": reward.get("eval_judge_correctness", 0.0),
        "pre_solve_rate": reward.get("pre_solve_rate", 0.0),
        "pre_solve_rate_original": reward.get("pre_solve_rate_original", 0.0),
        "post_solve_rate": reward.get("post_solve_rate", 0.0),
        "delta_solve": reward.get("delta_solve", 0.0),
        "delta_solve_raw": reward.get("delta_solve_raw", 0.0),
        "r_correctness": reward.get("r_correctness", 0.0),
        "r_judge": reward.get("r_judge", 0.0),
        "r_gate": reward.get("r_gate", 0.0),
        "gated": reward.get("gated", 0),
        "r_length": reward.get("r_length", 0.0),
        "length_decay": reward.get("length_decay", 1.0),
        "r_think_length": reward.get("r_think_length", 0.0),
        "thinking_decay": reward.get("thinking_decay", 1.0),
        "r_eoc": reward.get("r_eoc", 0.0),
        "student_token_share": reward.get("student_token_share", 0.0),
        "student_tokens_total": reward.get("student_tokens_total", 0),
        "r_think": reward.get("r_think", 0.0),
        "correct_think_rate": reward.get("correct_think_rate", 0.0),
        "n_think_violations": reward.get("n_think_violations", 0),
        "n_missing_think": reward.get("n_missing_think", 0),
        "score": reward.get("score", 0.0),
        "extra_info_keys": sorted(extra_info.keys()) if isinstance(extra_info, dict) else [],
        "transcript": _format_transcript(msgs),
        "messages": msgs,
    }


def _emit_sinks(*, row: dict[str, Any]) -> None:
    """Push one rollout to stdout + JSONL file + best-effort wandb + batch stats."""
    logger.info(
        "eduardo_rollout score=%.3f Δsolve=%.3f(%.2f→%.2f) r_judge=%.3f r_gate=%.3f r_len=%.3f r_eoc=%.3f "
        "turns=%d eoc=%d judge(leak=%s,teach=%s) problem=%r",
        row.get("score", 0.0),
        row.get("delta_solve", 0.0),
        row.get("pre_solve_rate", 0.0),
        row.get("post_solve_rate", 0.0),
        row.get("r_judge", 0.0),
        row.get("r_gate", 0.0),
        row.get("r_length", 0.0),
        row.get("r_eoc", 0.0),
        row.get("num_turns", 0),
        row.get("eoc_used", 0),
        row.get("judge_no_leak", ""),
        row.get("judge_teaching", ""),
        _preview(row.get("problem") or "", 80),
    )

    _dump_jsonl(row)

    # wandb tables use the pre-rendered transcript instead of raw messages.
    wandb_row = {k: v for k, v in row.items() if k != "messages"}
    _log_wandb(wandb_row)

    _accumulate_batch_stats(row)


def _count_tokens_per_turn(
    teacher_contents: list[str], tokenizer: Any
) -> list[int]:
    """Token count per turn; falls back to ~4 chars/token without a tokenizer."""
    if not teacher_contents:
        return []
    if tokenizer is not None:
        try:
            encode = getattr(tokenizer, "encode", None)
            if encode is None:
                raise AttributeError("tokenizer has no .encode")
            try:
                return [
                    len(encode(c, add_special_tokens=False)) for c in teacher_contents
                ]
            except TypeError:
                return [len(encode(c)) for c in teacher_contents]
        except Exception as e:
            logger.debug("teacher-token count: tokenizer failed (%s), using heuristic", e)
    return [len(c) // 4 for c in teacher_contents]


def _count_teacher_tokens(
    teacher_contents: list[str], tokenizer: Any
) -> int:
    """Sum tokens across teacher turns (see ``_count_tokens_per_turn``)."""
    return sum(_count_tokens_per_turn(teacher_contents, tokenizer))


def _token_weighted_decay(
    token_counts: list[int], target: float, power: float = 2.0
) -> float:
    """Token-weighted rational decay, Σ t_i·d_i / Σ t_i with d_i = 1/(1 + ((t_i − T0)/T0)^k).

    Returns 1.0 when there are no tokens.
    """
    total = sum(token_counts)
    if total <= 0 or target <= 0:
        return 1.0
    weighted = 0.0
    for t_i in token_counts:
        overage = max(0.0, t_i - target) / target
        weighted += t_i / (1.0 + overage**power)
    return weighted / total


def _empty_reward() -> dict[str, Any]:
    return {
        "score": 0.0,
        "acc": 0.0,
        "r_correctness": 0.0,
        "r_judge": 0.0,
        "r_gate": 0.0,
        "gated": 0,
        "r_length": 0.0,
        "length_decay": 1.0,
        "r_think_length": 0.0,
        "thinking_decay": 1.0,
        "r_eoc": 0.0,
        "student_token_share": 0.0,
        "student_tokens_total": 0,
        "r_think": 0.0,
        "correct_think_rate": 0.0,
        "n_correct_think_turns": 0,
        "n_think_violations": 0,
        "n_missing_think": 0,
        "pre_solve_rate": 0.0,
        "post_solve_rate": 0.0,
        "delta_solve": 0.0,
        "delta_solve_raw": 0.0,
        "used_transfer_problem": 0,
        "pre_solve_rate_original": 0.0,
        "student_n_incorrect_format": 0,
        "student_n_failed": 0,
        "judge_no_leak_ok": 0,
        "leak_rate": 1,
        "judge_teaching_ok": 0,
        "num_turns": 0,
        "teacher_tokens_total": 0,
        "teacher_tokens_mean": 0.0,
        "thinking_tokens_total": 0,
        "thinking_tokens_mean": 0.0,
        "visible_tokens_mean": 0.0,
        "eoc_used": 0,
        "judge_pass": 0,
        "leak_rate_early": 0,
        # veRL requires every rollout to report the same keys.
        **_EVAL_JUDGE_BLANK,
        "judge_no_leak": "REJECT",
        "judge_teaching": "REJECT",
        "judge_no_leak_reasoning": "",
        "judge_teaching_reasoning": "",
    }
