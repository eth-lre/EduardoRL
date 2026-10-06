"""Length-penalty metric: -penalty if any teacher turn hit the token limit."""

from __future__ import annotations

from typing import Optional


def compute(
    messages: list[dict],
    max_tokens_per_turn: int,
    penalty: float,
    tokenizer: Optional[object] = None,
) -> float:
    """Return -penalty if any teacher (assistant) message was truncated.

    A message counts as truncated if its token length is >= max_tokens_per_turn - 1.
    Without a tokenizer, a ~4 chars/token heuristic is used.
    """
    teacher_msgs = [
        (m.get("content") or "") for m in messages if m.get("role") == "assistant"
    ]
    if not teacher_msgs:
        return 0.0

    if tokenizer is not None:
        try:
            encode = getattr(tokenizer, "encode", None)
            if encode is None:
                raise AttributeError("tokenizer has no .encode")
            counts = [len(encode(m, add_special_tokens=False)) for m in teacher_msgs]
        except TypeError:
            counts = [len(tokenizer.encode(m)) for m in teacher_msgs]
        truncated = any(c >= max_tokens_per_turn - 1 for c in counts)
    else:
        truncated = any(len(m) / 4 >= max_tokens_per_turn - 1 for m in teacher_msgs)

    return -penalty if truncated else 0.0
