"""Build Big-Math-RL-transfer: near-transfer problem variants plus student solve rates.

For every problem in ``rd211/Big-Math-RL-Verified-Filtered``:

1. A rewriter model writes a variant with new numbers and/or story but the same
   solution structure (``transfer_problem``, ``transfer_answer``). It is kept
   only if a blind solve of the variant reproduces the stated answer.
2. The student samples ``k`` solutions of the original -> ``pre_solve_rate``.
3. The student samples ``k`` solutions of the variant ->
   ``pre_solve_rate_transfer_problem``.

Rows are checkpointed to ``<out-dir>/<split>_rows.jsonl`` so interrupted runs resume.

Usage::

    export SERVING_API_KEY=<token>  HF_TOKEN=...
    python -m eduardo.create_transfer_dataset \\
        --out-dir "$SCRATCH/data/bigmath_transfer" --push

Endpoints: ``JUDGE_API_BASE``/``JUDGE_MODEL_NAME``, ``STUDENT_API_BASE``/
``STUDENT_MODEL_NAME``, key in ``SERVING_API_KEY``. With ``--gemini`` (key in
``GEMINI_API_KEY``), Gemini is the rewriter and verification is a majority vote
over ``--rewrite-attempts`` solves.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

import openai

from .utils import call_openai_with_retries, get_async_openai_client, grade_math_answer

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

_REWRITE_PROMPT = """You are an expert math problem writer. Given a math problem and its final answer, write a NEAR-TRANSFER variant of the problem.

Requirements:
- Preserve the underlying solution structure exactly: the same concepts, the same solution steps in the same order, and the same difficulty.
- Change the numbers and/or the surface story (names, objects, real-world context). Do NOT copy the original phrasing verbatim.
- The new numbers must lead to a clean, well-defined final answer (similar "niceness" to the original answer — e.g. if the original answer is an integer, the new one should be too).
- The variant must be fully self-contained and unambiguous.

First, solve your new problem step by step to verify it is well-posed and to derive its final answer. Then output the variant in EXACTLY this format:

<transfer_problem>
...the rewritten problem statement...
</transfer_problem>
<transfer_answer>
...the final answer only, formatted like the original answer (as it would appear inside \\boxed{{}})...
</transfer_answer>

Original problem:
{problem}

Original final answer: {answer}"""

_SOLVE_PROMPT = """Solve the following math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {problem}"""

_TP_RE = re.compile(r"<transfer_problem>(.*?)</transfer_problem>", re.DOTALL)
_TA_RE = re.compile(r"<transfer_answer>(.*?)</transfer_answer>", re.DOTALL)


def _parse_rewrite(text: str) -> tuple[str, str] | None:
    """Extract (transfer_problem, transfer_answer) from a judge response."""
    if not text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    m_p = _TP_RE.search(text)
    m_a = _TA_RE.search(text)
    if not (m_p and m_a):
        return None
    problem = m_p.group(1).strip()
    answer = m_a.group(1).strip()
    if answer.startswith(r"\boxed{") and answer.endswith("}"):
        answer = answer[len(r"\boxed{"):-1].strip()
    if not problem or not answer:
        return None
    return problem, answer


# ── API calls ────────────────────────────────────────────────────────────────

async def _chat_once(
    client: openai.AsyncClient,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    extra_body: dict | None = None,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body or {},
    )
    return resp.choices[0].message.content or ""


async def _verify_rewrite(
    t_problem: str,
    t_answer: str,
    client: openai.AsyncClient,
    args: argparse.Namespace,
    sem: asyncio.Semaphore,
) -> bool:
    """Blind check: solve the variant without its answer and require ``t_answer``.

    With ``--gemini`` a strict majority of ``rewrite_attempts`` samples must
    agree; otherwise a single greedy solve is used."""
    if args.gemini:
        model = args.gemini_model
        n_checks = args.rewrite_attempts
        temperature = args.judge_temperature  # temp 0 would make the N checks identical
        extra_body = {"reasoning_effort": args.gemini_reasoning_effort}
    else:
        model = args.judge_model
        n_checks = 1
        temperature = 0.0
        extra_body = {"chat_template_kwargs": {"enable_thinking": args.judge_enable_thinking}}

    async def one():
        return await call_openai_with_retries(
            _chat_once,
            client,
            model,
            [{"role": "user", "content": _SOLVE_PROMPT.format(problem=t_problem)}],
            temperature,
            args.judge_max_tokens,
            extra_body,
            label="verify",
            use_fallback=True,
            fallback=None,
            semaphore=sem,
        )

    texts = await asyncio.gather(*[one() for _ in range(n_checks)])
    n_correct = sum(
        bool(grade_math_answer(t, t_answer)["acc"]) for t in texts if t
    )
    return n_correct > n_checks // 2


async def _rewrite_problem(
    problem: str,
    answer: str,
    client: openai.AsyncClient,
    args: argparse.Namespace,
    sem: asyncio.Semaphore,
    hint: str = "",
) -> tuple[str, str] | None:
    """Request a near-transfer variant until one parses and passes the blind
    check. ``hint`` carries feedback from a variant rejected for difficulty."""
    prompt = _REWRITE_PROMPT.format(problem=problem, answer=answer) + hint
    model = args.gemini_model if args.gemini else args.judge_model
    extra_body = (
        {"reasoning_effort": args.gemini_reasoning_effort} if args.gemini
        else {"chat_template_kwargs": {"enable_thinking": args.judge_enable_thinking}}
    )
    for attempt in range(args.rewrite_attempts):
        text = await call_openai_with_retries(
            _chat_once,
            client,
            model,
            [{"role": "user", "content": prompt}],
            args.judge_temperature,
            args.judge_max_tokens,
            extra_body,
            label="rewrite",
            use_fallback=True,
            fallback=None,
            semaphore=sem,
        )
        parsed = _parse_rewrite(text) if text else None
        if parsed is None:
            logger.warning(
                "rewrite parse failure (attempt %d/%d): %r",
                attempt + 1, args.rewrite_attempts, (text or "")[:200],
            )
            continue
        if await _verify_rewrite(parsed[0], parsed[1], client, args, sem):
            return parsed
        logger.warning(
            "rewrite verify failure (attempt %d/%d): judge could not reproduce "
            "answer %r for its own variant",
            attempt + 1, args.rewrite_attempts, parsed[1][:80],
        )
    return None


async def _solve_rate(
    problem: str,
    ground_truth: str,
    client: openai.AsyncClient,
    args: argparse.Namespace,
    sem: asyncio.Semaphore,
) -> tuple[float, int, int]:
    """Sample k solutions from the student; return (frac_correct, n_scored, n_failed)."""
    msgs = [{"role": "user", "content": _SOLVE_PROMPT.format(problem=problem)}]

    async def one():
        return await call_openai_with_retries(
            _chat_once,
            client,
            args.student_model,
            msgs,
            args.student_temperature,
            args.student_max_tokens,
            # Ensures no thinking for Qwen-style students; no-op otherwise.
            {"chat_template_kwargs": {"enable_thinking": False}},
            label="solve",
            use_fallback=True,
            fallback=None,
            semaphore=sem,
        )

    texts = await asyncio.gather(*[one() for _ in range(args.k)])
    n_correct = n_scored = n_failed = 0
    for t in texts:
        if t is None:
            n_failed += 1
            continue
        n_scored += 1
        n_correct += int(grade_math_answer(t, ground_truth)["acc"])
    frac = (n_correct / n_scored) if n_scored else 0.0
    return frac, n_scored, n_failed


async def _process_row(
    idx: int,
    row: dict,
    rewrite_client: openai.AsyncClient,
    student_client: openai.AsyncClient,
    judge_sem: asyncio.Semaphore,
    student_sem: asyncio.Semaphore,
    args: argparse.Namespace,
) -> dict:
    problem = row["problem"]
    answer = str(row["answer"])

    rewrite_task = _rewrite_problem(problem, answer, rewrite_client, args, judge_sem)
    pre_task = _solve_rate(problem, answer, student_client, args, student_sem)
    rewrite, (pre_rate, pre_scored, pre_failed) = await asyncio.gather(
        rewrite_task, pre_task
    )

    out = {
        "idx": idx,
        "problem": problem,
        "answer": answer,
        "llama8b_solve_rate": float(row.get("llama8b_solve_rate") or 0.0),
        "transfer_problem": None,
        "transfer_answer": None,
        "pre_solve_rate": pre_rate,
        "pre_solve_rate_n_scored": pre_scored,
        "pre_solve_rate_transfer_problem": None,
        "pre_solve_rate_transfer_n_scored": 0,
    }
    if pre_failed:
        logger.warning("row %d: %d/%d original solve calls failed", idx, pre_failed, args.k)

    if rewrite is None:
        logger.error("row %d: rewrite failed after %d attempts, transfer fields null",
                     idx, args.rewrite_attempts)
        return out

    # Difficulty gate: regenerate (with feedback) variants at the solve-rate
    # ceiling or far from the original's rate. If none passes, keep the one with
    # the smallest gap; recorded rates allow filtering downstream.
    best = None  # (passes_gate, gap, t_problem, t_answer, t_rate, t_scored)
    for d_attempt in range(args.difficulty_attempts + 1):
        t_problem, t_answer = rewrite
        t_rate, t_scored, t_failed = await _solve_rate(
            t_problem, t_answer, student_client, args, student_sem
        )
        if t_failed:
            logger.warning("row %d: %d/%d transfer solve calls failed", idx, t_failed, args.k)

        gap = abs(t_rate - pre_rate)
        ok_ceiling = args.max_transfer_solve_rate < 0 or t_rate <= args.max_transfer_solve_rate
        ok_gap = args.max_solve_rate_gap < 0 or gap <= args.max_solve_rate_gap
        passed = ok_ceiling and ok_gap
        if best is None or (passed, -gap) > (best[0], -best[1]):
            best = (passed, gap, t_problem, t_answer, t_rate, t_scored)
        if passed or d_attempt == args.difficulty_attempts:
            break

        reason = "too easy (ceiling)" if not ok_ceiling else (
            "much easier than the original" if t_rate > pre_rate else "much harder than the original")
        logger.info(
            "row %d: variant rejected by difficulty gate (%s: transfer=%.2f vs "
            "original=%.2f), regenerating (%d/%d)",
            idx, reason, t_rate, pre_rate, d_attempt + 1, args.difficulty_attempts,
        )
        hint = (
            f"\n\nIMPORTANT: A previous variant of this problem turned out {reason} "
            f"for our reference solver (variant solve rate {t_rate:.2f} vs original "
            f"{pre_rate:.2f}). Write a NEW variant — different numbers and story from "
            "both the original and any earlier variant — whose difficulty matches the "
            "original as closely as possible. Do not make it a near-copy of the original."
        )
        rewrite = await _rewrite_problem(
            problem, answer, rewrite_client, args, judge_sem, hint=hint
        )
        if rewrite is None:
            logger.warning("row %d: regeneration failed, keeping best candidate so far", idx)
            break

    _, _, t_problem, t_answer, t_rate, t_scored = best
    out["transfer_problem"] = t_problem
    out["transfer_answer"] = t_answer
    out["pre_solve_rate_transfer_problem"] = t_rate
    out["pre_solve_rate_transfer_n_scored"] = t_scored
    return out


# ── Split driver with jsonl checkpointing ────────────────────────────────────

async def _run_split(
    split_name: str,
    split,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    ckpt_path = out_dir / f"{split_name}_rows.jsonl"
    done: dict[int, dict] = {}
    if ckpt_path.exists():
        with ckpt_path.open() as f:
            for line in f:
                r = json.loads(line)
                done[r["idx"]] = r
        print(f"[{split_name}] resuming: {len(done)} rows already done")

    if args.gemini:
        rewrite_client = get_async_openai_client(args.gemini_api_base, args.gemini_api_key_env)
    else:
        rewrite_client = get_async_openai_client(args.judge_api_base, args.api_key_env)
    student_client = get_async_openai_client(args.student_api_base, args.api_key_env)
    judge_sem = asyncio.Semaphore(args.judge_concurrency)
    student_sem = asyncio.Semaphore(args.student_concurrency)
    # Bound rows in flight so later-stage calls don't queue behind the whole
    # dataset's first-stage calls (which would delay all checkpointing).
    row_sem = asyncio.Semaphore(args.row_concurrency)

    todo = [(i, split[i]) for i in range(len(split)) if i not in done]
    n_total = len(split)
    n_done = len(done)
    lock = asyncio.Lock()

    async def worker(i: int, row: dict):
        nonlocal n_done
        async with row_sem:
            result = await _process_row(
                i, row, rewrite_client, student_client, judge_sem, student_sem, args
            )
        async with lock:
            done[i] = result
            with ckpt_path.open("a") as f:
                f.write(json.dumps(result) + "\n")
            n_done += 1
            if n_done % 25 == 0 or n_done == n_total:
                print(f"[{split_name}] {n_done}/{n_total} rows done")

    await asyncio.gather(*[worker(i, row) for i, row in todo])
    return [done[i] for i in sorted(done)]


def main():
    ap = argparse.ArgumentParser(description="Build Big-Math-RL-transfer")
    ap.add_argument("--hf-dataset", default="rd211/Big-Math-RL-Verified-Filtered")
    ap.add_argument("--num-problems", type=int, default=-1,
                    help="Cap per split (-1 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=16,
                    help="Student samples per problem for solve rates")
    ap.add_argument("--row-concurrency", type=int, default=256,
                    help="Max rows in flight; keeps later-stage calls from "
                         "starving behind the whole dataset's stage-1 calls")
    ap.add_argument("--out-dir", default="./transfer_dataset")

    # Judge (rewriter)
    ap.add_argument("--judge-model",
                    default=os.environ.get("JUDGE_MODEL_NAME",
                                           f"Qwen3.6-27B-{os.environ.get('USER', '')}"))
    ap.add_argument("--judge-api-base",
                    default=os.environ.get("JUDGE_API_BASE",
                                           "http://localhost:8080/v1"))
    ap.add_argument("--judge-temperature", type=float, default=0.7)
    ap.add_argument("--judge-max-tokens", type=int, default=16384)
    ap.add_argument("--judge-concurrency", type=int, default=128)
    ap.add_argument("--judge-enable-thinking",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rewrite-attempts", type=int, default=3)

    # Difficulty gate
    ap.add_argument("--max-transfer-solve-rate", type=float, default=0.75,
                    help="Reject variants the student pre-solves above this "
                         "rate (ceiling — no headroom to measure transfer); "
                         "-1 disables")
    ap.add_argument("--max-solve-rate-gap", type=float, default=0.375,
                    help="Reject variants whose student solve rate differs "
                         "from the original's by more than this (difficulty "
                         "mismatch); -1 disables. Keep well above the k=16 "
                         "sampling noise (~0.12 std) or matched pairs get "
                         "rejected by chance")
    ap.add_argument("--difficulty-attempts", type=int, default=2,
                    help="Extra regeneration rounds when a variant fails the "
                         "difficulty gate; the best candidate is kept if all "
                         "fail")

    # Gemini rewriter (opt-in): replaces the judge model for rewrite + verify.
    ap.add_argument("--gemini", action="store_true",
                    help="Use Gemini for transfer-problem generation and "
                         "multi-sample answer verification")
    ap.add_argument("--gemini-model", default="gemini-3.1-pro-preview")
    # Maps to Gemini's thinking_level.
    ap.add_argument("--gemini-reasoning-effort", default="low",
                    choices=["low", "medium", "high"],
                    help="Gemini thinking level for rewrite + verify calls")
    ap.add_argument("--gemini-api-base",
                    default="https://generativelanguage.googleapis.com/v1beta/openai/")
    ap.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")

    # Student (solver)
    ap.add_argument("--student-model",
                    default=os.environ.get("STUDENT_MODEL_NAME",
                                           f"Llama-3.1-8B-Instruct-{os.environ.get('USER', '')}"))
    ap.add_argument("--student-api-base",
                    default=os.environ.get("STUDENT_API_BASE",
                                           "http://localhost:8080/v1"))
    ap.add_argument("--student-temperature", type=float, default=0.6)
    ap.add_argument("--student-max-tokens", type=int, default=1024)
    ap.add_argument("--student-concurrency", type=int, default=64)

    ap.add_argument("--api-key-env", default="SERVING_API_KEY")

    # HF upload
    ap.add_argument("--push", action="store_true",
                    help="Push the result to the private HF repo")
    ap.add_argument("--hf-repo", default="dmacjam/eduardoRL-transfer-dataset")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import datasets
    import pandas as pd

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = datasets.load_dataset(args.hf_dataset)
    result_splits: dict[str, datasets.Dataset] = {}
    for split_name in ds:
        split = ds[split_name]
        if args.num_problems > 0:
            n = min(args.num_problems, len(split))
            split = split.shuffle(seed=args.seed).select(range(n))
        rewriter = args.gemini_model if args.gemini else args.judge_model
        print(f"[{split_name}] processing {len(split)} problems "
              f"(k={args.k}, rewriter={rewriter}, student={args.student_model})")

        rows = asyncio.run(_run_split(split_name, split, out_dir, args))

        n_failed = sum(1 for r in rows if r["transfer_problem"] is None)
        if n_failed:
            print(f"[{split_name}] WARNING: {n_failed}/{len(rows)} rows have no "
                  f"transfer_problem (rewrite failed); kept with null fields")

        df = pd.DataFrame(rows).drop(columns=["idx"])
        parquet_path = out_dir / f"{split_name}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"[{split_name}] wrote {len(df)} rows → {parquet_path}")
        print(f"[{split_name}] mean pre_solve_rate={df['pre_solve_rate'].mean():.3f}  "
              f"mean pre_solve_rate_transfer_problem="
              f"{df['pre_solve_rate_transfer_problem'].dropna().mean():.3f}")
        result_splits[split_name] = datasets.Dataset.from_pandas(df, preserve_index=False)

    if args.push:
        print("[push]")
        dd = datasets.DatasetDict(result_splits)
        dd.push_to_hub(args.hf_repo, private=True)
        print(f"[push] uploaded to https://huggingface.co/datasets/{args.hf_repo} (private)")
    else:
        print(f"[push] skipped — rerun with --push to upload to {args.hf_repo}")


if __name__ == "__main__":
    main()
