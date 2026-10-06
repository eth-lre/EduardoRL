# The Assistance Dilemma: Learning to Teach via Multi-Turn RL (EduardoRL)

Code, data and evaluation for *The Assistance Dilemma: Learning to Teach via
Multi-Turn Reinforcement Learning*
([arXiv:2610.06446](https://arxiv.org/abs/2610.06446)).

Eduardo tutors are trained with multi-turn RL (DPPO, on top of
[veRL](https://github.com/volcengine/verl)). The student is a frozen
Llama-3.1-8B-Instruct. The reward is the student's normalized learning gain on a
**masked near-transfer post-test**: the student attempts an unseen variant of the
tutored problem while seeing only its own turns (tutor turns are replaced by
`(hidden)`). Two binary reward gates, **factual correctness** and **no
solution handover**, are judged by Qwen3.6-27B.

> [!IMPORTANT]
> **Hardware assumptions.** All training, serving and evaluation scripts assume a
> **SLURM cluster with NVIDIA GH200 (aarch64) GPUs**: jobs are submitted with
> `sbatch`, run inside a pyxis/enroot container built from the GH200 image, and
> the model servers are found via `squeue`. Running elsewhere means adapting the
> launchers in `eduardo/scripts/` and `eval/*_slurm.sh` and the container image.

## Models

The trained Eduardo tutors will be published on the Hugging Face Hub:

| model | base        | link |
|---|-------------|---|
| Eduardo-27B | Qwen3.8-27B | [dmacjam/eduardo-27b](https://huggingface.co/dmacjam/eduardo-27b) |

## Repository layout

```
eduardo/                    training environment + RL recipe (package name: eduardo)
  reward_function.py        masked near-transfer reward, binary gates, efficiency decay
  eduardo_interaction.py    frozen-student interaction (veRL Interaction)
  eduardo_agent_loop.py     multi-turn agent loop that keeps the live transcript
  metrics/                  correctness (post-test), judges, length/thinking decay
  configs/                  Hydra configs (user.yaml = the paper recipe) + all prompts
  create_transfer_dataset.py  builds near-transfer pairs + measured solve rates
  process_dataset.py        HF near-transfer dataset -> veRL train/val/test parquet
  scripts/                  SLURM launchers: servers, training, ablations, merging
eval/
  run_middleturn_eval.py    MathDial / Eedi middle-turn evaluation (Ped-RM, win rate, judge)
  eval_slurm.sh             SLURM wrapper for run_middleturn_eval.py
  run_env_test_eval.py      training-environment evaluation (Δ_transfer, Δ_same, leak rate)
  transfer_eval_slurm.sh    SLURM wrapper for run_env_test_eval.py
  pedagogical_rm.py         MathTutorBench pedagogical reward-model scorer
  prompts/                  evaluation prompts and judge rubrics
Dockerfile, requirements-gh200.txt   the training / serving image
```

## Near-transfer dataset

The dataset is on the Hugging Face Hub:
[`dmacjam/eduardoRL-transfer-dataset`](https://huggingface.co/datasets/dmacjam/eduardoRL-transfer-dataset).
Each source problem comes from Big-Math. Every split has these columns:

| column | meaning |
|---|---|
| `problem`, `answer` | source problem (tutored) and its answer |
| `transfer_problem`, `transfer_answer` | near-transfer variant (post-test only); empty if no variant passed validation |
| `pre_solve_rate` | frozen student's cold solve rate on the source (k=16) |
| `pre_solve_rate_transfer_problem` | frozen student's cold solve rate on the variant (k=16) |
| `*_n_scored` | number of graded samples behind each rate |
| `llama8b_solve_rate` | solve rate inherited from Big-Math (not used for the reward) |

`train` has 10,000 source problems (9,876 with a variant). `test` has 500
(494 with a variant; this is the 494-problem environment test set). `all_test`
has 1,945 problems (1,919 with a variant).

```python
from datasets import load_dataset
ds = load_dataset("dmacjam/eduardoRL-transfer-dataset")
```

Turn it into veRL parquet files (the training script does this automatically on
first launch). The script downloads the dataset from the Hub; pass
`--data-dir <dir>` to read local `train.jsonl` / `test.jsonl` files instead:

```bash
python -m eduardo.process_dataset --num-train 10000 --num-test 100 --seed 42 \
    --out-dir $WORK_DIR/data/eduardo
```

To regenerate the dataset from scratch (Gemini rewriter, blind-solve
validation, difficulty gate), see `eduardo/create_transfer_dataset.py` and
`eduardo/scripts/transfer_dataset_slurm.sh`.

## Setup

The recipe was run on a SLURM cluster with 4× GH200 nodes. Build the image from
`Dockerfile`, which pins veRL v0.7.1 on top of the NGC vLLM container, and
import it for [pyxis/enroot](https://github.com/NVIDIA/pyxis):

```bash
docker build -t eduardo .
enroot import -o $WORK_DIR/eduardo.sqsh dockerd://eduardo
```

Environment variables used by the scripts:

| variable | meaning |
|---|---|
| `WORK_DIR` | scratch dir for data, checkpoints, caches (default `$SCRATCH/eduardo`) |
| `EDUARDO_RECIPE_DIR` | path to `eduardo/` in your checkout (default `$HOME/eduardo/eduardo`) |
| `CONTAINER_IMAGE` | image for `srun --container-image` (default `$WORK_DIR/eduardo.sqsh`); `CONTAINER_ARGS=""` runs on the host |
| `SERVING_API_KEY` | bearer token for the vLLM endpoints (any string when vLLM runs without `--api-key`) |
| `GEMINI_API_KEY` | Gemini-3.1-Pro evaluation judge (never part of the reward) |
| `WANDB_API_KEY`, `WANDB_ENTITY` | W&B logging |

The SLURM scripts write logs to `logs/` relative to the submit directory. Create
it (`mkdir -p logs`) before you submit, and pass `--account=<your account>` to
`sbatch` if your cluster requires one.

## Training

1. Start the frozen student and the training judge. Each is a persistent vLLM job
   named `serve_<served-name>`, and the training job looks it up with `squeue`:

   ```bash
   ./eduardo/scripts/launch_student_server.sh   # Llama-3.1-8B-Instruct, 4 DP engines
   ./eduardo/scripts/launch_judge_server.sh     # Qwen3.6-27B, 2 DP x TP 2
   ```

2. Train. `configs/user.yaml` is the paper recipe, and every model size uses the
   same hyperparameters:

   ```bash
   cd eduardo
   sbatch --export=ALL,EXPERIMENT=eduardo-4b,MODEL_PATH=Qwen/Qwen3.5-4B   scripts/train_slurm.sh eduardo_base
   sbatch --export=ALL,EXPERIMENT=eduardo-9b,MODEL_PATH=Qwen/Qwen3.5-9B   scripts/train_slurm.sh eduardo_base
   sbatch --export=ALL,EXPERIMENT=eduardo-14b,MODEL_PATH=Qwen/Qwen3-14B   scripts/train_slurm.sh eduardo_base
   sbatch --export=ALL,EXPERIMENT=eduardo-27b,MODEL_PATH=Qwen/Qwen3.8-27B scripts/train_slurm.sh eduardo_base
   ```

3. Merge the FSDP checkpoint into a Hugging Face model, written to
   `$WORK_DIR/eval_checkpoints/<experiment>/step_<step>`:

   ```bash
   sbatch scripts/merge_checkpoint.sh eduardo-27b 75
   ```

### Leave-one-out ablation (4B)

`eduardo/scripts/run_ablations.sh` submits three arms from the full recipe,
each with one Hydra override: `ablation-4b-nomasking`
(`omit_teacher_turns=false`), `ablation-4b-nogates` (`judge_hard_gate=false`,
`judge_penalty=0`) and `ablation-4b-notransfer` (`test_on_transfer=false`).
Merge and evaluate each arm with the commands below.

```bash
MODEL_PATH=Qwen/Qwen3.5-4B ./eduardo/scripts/run_ablations.sh [--dry-run]
```

## Evaluation

**MathDial / Eedi middle-turn generation.** The tutor writes the middle teacher
turn of each human dialog. The scripts report the MathTutorBench pedagogical RM
score, the win rate against the human turn, and 1–5 Gemini-3.1-Pro pedagogy and
factuality scores. This benchmark needs no student server.

```bash
sbatch --export=ALL,EVAL_MODEL=$WORK_DIR/eval_checkpoints/eduardo-27b/step_75 \
    eval/eval_slurm.sh --gemini --enable-thinking --benchmark mathdial
sbatch --export=ALL,EVAL_MODEL=$WORK_DIR/eval_checkpoints/eduardo-27b/step_75 \
    eval/eval_slurm.sh --gemini --enable-thinking --benchmark eedi
```

**Training environment.** The tutor holds full dialogs with the frozen student
on the 494 held-out problems. The scripts report Δ_transfer, Δ_same and the leak
rate, using the same masked protocol as training.

```bash
sbatch --export=ALL,EVAL_MODEL=$WORK_DIR/eval_checkpoints/eduardo-27b/step_75 \
    eval/transfer_eval_slurm.sh --gemini --enable-thinking
```

**External benchmarks.** MathTutorBench and TutorMoments run with their own
code. To serve a trained tutor as an OpenAI-compatible endpoint, run
`MODEL_PATH=... SERVED_NAME=eduardo-27B ./eduardo/scripts/launch_trained_model.sh`.

## Citation

```bibtex
@article{macina2026assistance,
  title   = {The Assistance Dilemma: Learning to Teach via Multi-Turn Reinforcement Learning},
  author  = {Macina, Jakub and Kapur, Manu and Sachan, Mrinmaya},
  journal = {arXiv preprint arXiv:2610.06446},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.06446}
}
```

## License

This repository is released under the
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
license (CC BY-SA 4.0). See [`LICENSE`](LICENSE).
