"""Eduardo agent loop: a ``ToolAgentLoop`` subclass that exposes the full
multi-turn transcript to the reward function.

The stock loop only returns concatenated token ids, which the reward manager
decodes with ``skip_special_tokens=True``, erasing turn boundaries. We copy
``agent_data.messages`` into ``extra_fields`` after each state transition; veRL
forwards these to the reward function as ``extra_info["messages"]``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentState,
    ToolAgentLoop,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shallow per-message copy: isolates dicts from later mutation without
    cloning large payloads."""
    return [dict(m) for m in messages]


@register("eduardo_agent")
class EduardoAgentLoop(ToolAgentLoop):
    """ToolAgentLoop that persists the live transcript into ``extra_fields``.

    Each teacher turn is hard-capped at ``max_tokens_per_turn +
    max_thinking_tokens_per_turn`` so it covers both the ``<think>`` block and
    the visible reply; the reward's length decays shape behaviour below this cap.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        reward_kwargs = self.config.get("custom_reward_function", {}).get(
            "reward_kwargs", {}
        )
        self.max_tokens_per_turn = int(reward_kwargs.get("max_tokens_per_turn", 400))
        # Thinking budget defaults to the visible budget (as in reward_function.py).
        self.max_thinking_tokens_per_turn = int(
            reward_kwargs.get(
                "max_thinking_tokens_per_turn", self.max_tokens_per_turn
            )
        )

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        turn_params = {
            **sampling_params,
            "max_tokens": self.max_tokens_per_turn
            + self.max_thinking_tokens_per_turn,
        }
        next_state = await super()._handle_generating_state(
            agent_data, turn_params, ignore_termination=ignore_termination
        )
        self._stash_transcript(agent_data)
        return next_state

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        next_state = await super()._handle_interacting_state(agent_data)
        self._stash_transcript(agent_data)
        return next_state

    @staticmethod
    def _stash_transcript(agent_data: AgentData) -> None:
        """Snapshot the conversation and turn counts into ``extra_fields``.

        Copied because ``agent_data.messages`` keeps mutating after the handler returns.
        """
        try:
            agent_data.extra_fields["messages"] = _copy_messages(agent_data.messages)
            agent_data.extra_fields["num_assistant_turns"] = int(
                agent_data.assistant_turns
            )
            agent_data.extra_fields["num_user_turns"] = int(agent_data.user_turns)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "EduardoAgentLoop: failed to stash transcript into extra_fields: %s",
                exc,
            )
