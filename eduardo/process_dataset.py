"""Near-transfer dataset -> Parquet for veRL's data loader.

Reads the released near-transfer dataset (by default from the Hugging Face Hub,
``dmacjam/eduardoRL-transfer-dataset``), where each problem has a
near-transfer variant and student solve rates for both. The teacher tutors on
the original and the student is tested on the variant, so the train/val
difficulty band is applied to both problems. Each prompt ends with a fixed
student opening turn, since ``ToolAgentLoop`` always generates the teacher first.

Usage::

    python -m eduardo.process_dataset \\
        --num-train 1000 --num-test 50 --num-holdout 500 --seed 42 \\
        --out-dir $WORK_DIR/data/eduardo

Outputs: ``train.parquet`` (training) and ``val.parquet`` (veRL validation),
disjoint subsets of the train split, and ``test.parquet`` (offline eval, from
the test split, drawn without the difficulty band).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils import load_prompt

_TEACHER_SYSTEM_PROMPT = load_prompt("teacher_prompt")

_STUDENT_OPENING_TEMPLATE = (
    "This is the problem I am trying to understand: {problem}"
)


def _row(
    problem: str,
    answer: str,
    solve_rate: float,
    transfer_problem: str,
    transfer_answer: str,
    transfer_solve_rate: float,
    split: str,
) -> dict:
    """One dataset row (schema consumed by veRL's RLHFDataset + ToolAgentLoop).

    ``ground_truth`` is the original answer (used by the leakage judge); the
    transfer answer is used only to grade the student's final attempts.
    """
    prompt = [
        {
            "role": "system",
            "content": _TEACHER_SYSTEM_PROMPT.format(problem=problem),
        },
        {
            "role": "user",
            "content": _STUDENT_OPENING_TEMPLATE.format(problem=problem),
        },
    ]
    return {
        "data_source": "bigmath",
        "prompt": prompt,
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "problem": problem,
            "answer": answer,
            # Lets the reward function run the periodic-eval judge on val rows only.
            "split": split,
            # Original problem's rate; not the Δ_solve baseline.
            "solve_rate": float(solve_rate),
            "pre_solve_rate": float(solve_rate),
            "transfer_problem": transfer_problem,
            "transfer_answer": transfer_answer,
            # Δ_solve baseline: untutored rate on the transfer problem.
            "pre_solve_rate_transfer_problem": float(transfer_solve_rate),
            # The live dialogue covers only the original problem; the variant stays unseen.
            "interaction_kwargs": {
                "name": "eduardo",
                "problem": problem,
                "answer": answer,
                "ground_truth": answer,
            },
        },
        # EduardoAgentLoop exposes the transcript to the reward function.
        "agent_name": "eduardo_agent",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-train", type=int, default=1000)
    ap.add_argument("--num-test", type=int, default=50,
                    help="Rows for val.parquet — the set veRL rolls out every "
                         "trainer.test_freq steps during training. Held out "
                         "from the train split, disjoint from train.parquet")
    ap.add_argument("--num-holdout", type=int, default=500,
                    help="Rows for test.parquet — the offline eval set used by "
                         "eval/run_env_test_eval.py. Never read by training. "
                         "0 to skip the file")
    ap.add_argument("--holdout-solve-rate-min", type=float, default=0.0,
                    help="Solve-rate floor for test.parquet only (default: no "
                         "floor — unlike --solve-rate-min for train/val)")
    ap.add_argument("--holdout-solve-rate-max", type=float, default=1.0,
                    help="Solve-rate ceiling for test.parquet only (default: no "
                         "ceiling). Consider 0.9: normalised Δ divides by "
                         "(1 − pre), so near-ceiling rows amplify small gains")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--solve-rate-min", type=float, default=0.02,
        help="Keep problems whose untutored solve rate is >= this, applied to "
             "BOTH the original and its transfer variant",
    )
    ap.add_argument(
        "--solve-rate-max", type=float, default=0.65,
        help="Keep problems whose untutored solve rate is <= this, applied to "
             "BOTH the original and its transfer variant",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--hf-dataset", default="dmacjam/eduardoRL-transfer-dataset",
        help="Hugging Face dataset id with train/test splits",
    )
    ap.add_argument(
        "--data-dir", default=None,
        help="Optional local directory with train.jsonl / test.jsonl to load "
             "instead of --hf-dataset",
    )
    args = ap.parse_args()

    import datasets
    import pandas as pd  # noqa: F401 — pyarrow via pandas.to_parquet

    if args.data_dir is None:
        ds = datasets.load_dataset(args.hf_dataset)
    else:
        data_dir = Path(args.data_dir)
        files = {
            split: str(data_dir / f"{split}.jsonl")
            for split in ("train", "test")
            if (data_dir / f"{split}.jsonl").exists()
        }
        if "train" not in files:
            raise SystemExit(f"[data] {data_dir}/train.jsonl not found")
        ds = datasets.load_dataset("json", data_files=files)
    train_split = ds["train"]

    def _keep(x: dict) -> bool:
        """Require a transfer variant and both solve rates inside the difficulty band."""
        tp = x.get("transfer_problem")
        if not tp:
            return False
        rates = (
            x.get("pre_solve_rate"),
            x.get("pre_solve_rate_transfer_problem"),
        )
        return all(
            r is not None and args.solve_rate_min <= r <= args.solve_rate_max
            for r in rates
        )

    n_before = len(train_split)
    filtered = train_split.filter(_keep)
    print(
        f"[data] {len(filtered)}/{n_before} problems pass the difficulty filter "
        f"({args.solve_rate_min} ≤ solve rate ≤ {args.solve_rate_max}, "
        f"required for the original AND its transfer variant)"
    )
    if not len(filtered):
        raise SystemExit(
            "[data] no problems left after filtering — widen "
            "--solve-rate-min/--solve-rate-max"
        )

    # Val is carved out of the train split before train is drawn, so the two are
    # disjoint and test.parquet stays untouched until the offline eval.
    shuffled = filtered.shuffle(seed=args.seed)
    n_val = min(args.num_test, len(shuffled))
    val_data = shuffled.select(range(n_val))
    n_train = min(args.num_train, len(shuffled) - n_val)
    if n_train <= 0:
        raise SystemExit(
            "[data] no problems left for train.parquet after reserving "
            f"{n_val} for val.parquet — lower --num-test"
        )
    train_data = shuffled.select(range(n_val, n_val + n_train))
    print(f"[data] {n_train} train / {n_val} val problems (disjoint, both from train split)")

    # test.parquet (offline eval) skips the train/val difficulty band, which exists
    # to keep the training signal clean, and only requires a variant with measured
    # rates. Normalised Δ divides by (1 − pre), so near-ceiling rows can dominate;
    # --holdout-solve-rate-max trims them.
    def _keep_holdout(x: dict) -> bool:
        if not x.get("transfer_problem") or not x.get("transfer_answer"):
            return False
        return all(
            r is not None
            and args.holdout_solve_rate_min <= r <= args.holdout_solve_rate_max
            for r in (x.get("pre_solve_rate"), x.get("pre_solve_rate_transfer_problem"))
        )

    holdout_data = None
    test_split = ds["test"] if "test" in ds else None
    if args.num_holdout and test_split is None:
        print("[data] NOTE: no test split; skipping test.parquet")
    elif args.num_holdout:
        holdout_pool = test_split.filter(_keep_holdout)
        print(
            f"[data] {len(holdout_pool)}/{len(test_split)} test problems usable for "
            f"test.parquet (transfer variant + measured rates, solve rate in "
            f"[{args.holdout_solve_rate_min}, {args.holdout_solve_rate_max}])"
        )
        n_holdout = min(args.num_holdout, len(holdout_pool))
        if n_holdout < args.num_holdout:
            print(
                f"[data] NOTE: test.parquet gets {n_holdout} rows, not "
                f"{args.num_holdout} — that is the whole usable pool."
            )
        if n_holdout:
            holdout_data = holdout_pool.shuffle(seed=args.seed + 2).select(
                range(n_holdout)
            )
            ceiling = sum(
                1 for r in holdout_data
                if r["pre_solve_rate_transfer_problem"] > 0.9
            )
            if ceiling:
                print(
                    f"[data] NOTE: {ceiling}/{n_holdout} test rows have a "
                    f"transfer solve rate > 0.9. Normalised Δ divides by the "
                    f"headroom (1 − pre), so these amplify small gains; pass "
                    f"--holdout-solve-rate-max 0.9 to drop them."
                )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_split(split, path: Path, split_name: str):
        rows = [
            _row(
                r["problem"],
                r["answer"],
                r["pre_solve_rate"],
                r["transfer_problem"],
                r["transfer_answer"],
                r["pre_solve_rate_transfer_problem"],
                split_name,
            )
            for r in split
        ]
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        mean_pre = sum(x["extra_info"]["pre_solve_rate"] for x in rows) / len(rows)
        mean_tp = sum(
            x["extra_info"]["pre_solve_rate_transfer_problem"] for x in rows
        ) / len(rows)
        print(
            f"[data] wrote {len(df)} rows → {path}  "
            f"(mean pre_solve_rate={mean_pre:.3f}, transfer={mean_tp:.3f})"
        )

    write_split(train_data, out_dir / "train.parquet", "train")
    write_split(val_data, out_dir / "val.parquet", "val")
    if holdout_data is not None:
        write_split(holdout_data, out_dir / "test.parquet", "test")


if __name__ == "__main__":
    main()
