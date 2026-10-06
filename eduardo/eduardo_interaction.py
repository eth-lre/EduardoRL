"""Eduardo student-simulator Interaction for veRL's ``ToolAgentLoop``.

Called after every teacher turn. Terminates if the teacher emitted
``<end_of_conversation>`` after at least ``min_teacher_turns_before_eoc`` prior
turns (earlier tags are ignored); otherwise flips roles, strips teacher control
tokens, and queries the frozen student model for its next reply.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

try:
    from verl.interactions.base import BaseInteraction  # type: ignore
except Exception:  # pragma: no cover - shim for dev/lint without verl installed
    class BaseInteraction:
        def __init__(self, config: dict[str, Any], name: str = "eduardo"):
            self.config = config
            self.name = name

from .utils import (
    call_openai_with_retries,
    get_async_openai_client,
    get_concurrency_semaphore,
    load_prompt,
    strip_teacher_controls,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_STUDENT_SYSTEM_PROMPT = load_prompt("student_prompt")
_END_OF_CONVERSATION = "<end_of_conversation>"


class EduardoInteraction(BaseInteraction):
    """Student simulator, stateful per ``instance_id`` for concurrent rollouts."""

    def __init__(self, config: dict[str, Any], name: str = "eduardo"):
        try:
            super().__init__(config, name=name)
        except TypeError:
            # Older veRL: BaseInteraction(config) only
            super().__init__(config)
        self._api_base: str = config.get(
            "student_api_base",
            os.environ.get("STUDENT_API_BASE", "http://localhost:8080/v1"),
        )
        self._api_key_env: str = config.get("student_api_key_env", "SERVING_API_KEY")
        self._model: str = config["student_model"]
        self._temperature: float = float(config.get("temperature", 0.6))
        self._max_tokens: int = int(config.get("max_tokens", 500))
        self._retry_kwargs: dict[str, Any] = {
            "num_retries": int(config.get("num_retries", 3)),
            "backoff_base": float(config.get("backoff_base", 2.0)),
            "backoff_cap": float(config.get("backoff_cap", 30.0)),
            "jitter": bool(config.get("jitter", True)),
        }
        # Shares the "student" semaphore with reward_function.py so one budget
        # bounds both interaction turns and final-attempt calls.
        self._student_concurrency: int = int(config.get("student_concurrency", 64))
        # Teacher turns that must precede an <end_of_conversation> for it to be
        # honoured; earlier tags are ignored. 0 allows ending at any time.
        self._min_teacher_turns_before_eoc: int = int(
            config.get("min_teacher_turns_before_eoc", 0)
        )
        self._instances: dict[str, dict[str, Any]] = {}

    async def start_interaction(
        self, instance_id: str | None = None, **kwargs: Any
    ) -> str:
        instance_id = instance_id or str(uuid.uuid4())
        problem = kwargs.get("problem") or kwargs.get("extra_info", {}).get("problem", "")
        if not problem:
            raise ValueError(
                "EduardoInteraction.start_interaction requires `problem` in kwargs"
            )
        self._instances[instance_id] = {
            "problem": problem,
            "system": _STUDENT_SYSTEM_PROMPT.format(problem=problem),
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict],
        **kwargs: Any,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        state = self._instances.get(instance_id)
        if state is None:
            raise KeyError(
                f"EduardoInteraction: unknown instance_id={instance_id!r}; "
                f"did you call start_interaction?"
            )

        last_teacher = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"), None
        )
        if last_teacher and _END_OF_CONVERSATION in (last_teacher.get("content") or ""):
            n_prior = sum(1 for m in messages if m.get("role") == "assistant") - 1
            if n_prior >= self._min_teacher_turns_before_eoc:
                return True, "", 0.0, {"eoc_used": True}
            logger.debug(
                "EduardoInteraction: ignoring early <end_of_conversation> — "
                f"{n_prior} prior teacher turns < "
                f"{self._min_teacher_turns_before_eoc}"
            )

        student_msgs = self._flip_perspective(state["system"], messages)
        client = get_async_openai_client(self._api_base, self._api_key_env)
        semaphore = get_concurrency_semaphore("student", self._student_concurrency)

        async def _call():
            resp = await client.chat.completions.create(
                model=self._model,
                messages=student_msgs,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            return resp.choices[0].message.content or ""

        try:
            text = await call_openai_with_retries(
                _call,
                label="student_interaction",
                use_fallback=False,
                semaphore=semaphore,
                **self._retry_kwargs,
            )
        except Exception as e:
            # Terminate the rollout on a hard student failure; the reward
            # function will still see a well-formed (shorter) transcript.
            logger.error(f"EduardoInteraction: student call failed hard: {e}")
            return True, "", 0.0, {"error": str(e), "student_failed": True}

        return False, text.strip(), 0.0, {}

    async def finalize_interaction(self, instance_id: str) -> None:
        self._instances.pop(instance_id, None)

    @staticmethod
    def _flip_perspective(
        student_system: str, messages: list[dict]
    ) -> list[dict]:
        """Convert the teacher-view transcript to the student's view (roles
        swapped), stripping control tokens and thinking from teacher turns."""
        flipped: list[dict] = [{"role": "system", "content": student_system}]
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "system":
                continue
            if role == "assistant":
                flipped.append(
                    {"role": "user", "content": strip_teacher_controls(content)}
                )
            elif role == "user":
                flipped.append({"role": "assistant", "content": content})
        return flipped
