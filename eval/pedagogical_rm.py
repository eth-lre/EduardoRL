"""Pedagogical reward-model scoring for tutor conversations.

Follows mathtutorbench's ``reward_model/compute_scaffolding_score.py``:
every teacher turn is scored by a sequence-classification RM (default
``eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel``) given the dialog history
up to that turn, reading the raw logit ``outputs.logits[0][0]``.

Input format: a list of conversation records, each a dict with
    - ``problem``: str
    - ``messages``: list of {"role": "teacher"|"student", "content": str}

Output: one list of floats per conversation (one score per teacher turn).
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "Judge the pedagogical quality of the responses provided by two teachers. Focus on the quality of the "
    "scaffolding guidance, correctness, and actionability of the feedback through nudges, questions "
    "and hints. Do not give high scores for revealing the full answer."
)


def format_conversation(
    problem: str,
    reference_solution: str,
    history: list[dict[str, str]],
    response: str,
) -> list[dict[str, str]]:
    """Build the RM chat: system prompt, problem, dialog history, and the
    teacher response to score (as the final assistant message).
    """
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Problem: " + problem + "\nReference Solution: " + reference_solution,
        },
    ]
    for entry in history:
        role = "assistant" if entry["role"] in ("teacher", "Teacher", "Tutor") else "user"
        conversation.append({"role": role, "content": entry["content"]})
    conversation.append({"role": "assistant", "content": response})
    return conversation


def get_reward_inputs(
    problem: str,
    messages: list[dict[str, str]],
    only_last_message: bool = False,
) -> list[list[dict[str, str]]]:
    """One RM prompt per teacher turn: history so far + that turn as response."""
    prompts = []
    for i, message in enumerate(messages):
        if message["role"] != "teacher":
            continue
        prompts.append(
            format_conversation(
                problem=problem,
                reference_solution="",
                history=messages[:i],
                response=message["content"],
            )
        )
    if only_last_message and prompts:
        prompts = [prompts[-1]]
    return prompts


def score_conversations(
    records: list[dict[str, Any]],
    reward_model_path: str,
    only_last_message: bool = False,
) -> list[list[float]]:
    """Score every teacher turn of every conversation with the RM
    (raw logit ``outputs.logits[0][0]``, as in mathtutorbench).
    """
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    model = (
        AutoModelForSequenceClassification.from_pretrained(reward_model_path)
        .to(device)
        .eval()
    )

    prompt_sets = [
        get_reward_inputs(r["problem"], r["messages"], only_last_message) for r in records
    ]
    total = sum(len(p) for p in prompt_sets)

    scores: list[list[float]] = []
    with torch.no_grad(), tqdm(total=total, desc="RM scoring") as pbar:
        for p_set in prompt_sets:
            conv_scores = []
            for conversation in p_set:
                inputs = tokenizer.apply_chat_template(
                    conversation, tokenize=True, return_tensors="pt"
                )
                input_ids = inputs if isinstance(inputs, torch.Tensor) else inputs["input_ids"]
                input_ids = input_ids.to(device)
                outputs = model(input_ids)
                conv_scores.append(outputs.logits[0][0].item())
                pbar.update(1)
            scores.append(conv_scores)
    return scores
