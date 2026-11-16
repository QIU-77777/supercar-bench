"""Per-session state and tool execution for the MCP server."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Type

from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
    ContextState,
    context_state,
    context_state_hash,
)
from car_bench.envs.car_voice_assistant.context.fixed_context import (
    FixedContext,
    fixed_context,
)
from car_bench.envs.policy_evaluator import policy_errors_during_runtime
from car_bench.envs.car_voice_assistant.tasks.task_config import (
    TaskConfig,
    task_config,
)
from car_bench.envs.tool import Tool
from car_bench.envs.tool_execution_error_evaluator import (
    tool_execution_errors_during_runtime,
)
from car_bench.types import USER_AS_A_TOOL_ACTION_NAMES

REMOVED_ARGUMENT_NOTE = (
    " Note that this tool argument is currently removed so that this tool can "
    "not be used. Do not try to call this tool with this argument as it will "
    "result in an error."
)


class Session:
    """One evaluation session: isolated state plus one tool allowlist."""

    def __init__(
        self,
        session_id: str,
        tools_map: Dict[str, Type[Tool]],
        context_init_config: Optional[Dict[str, Any]] = None,
        tool_profile: Optional[List[str]] = None,
        removed_part: Optional[List[str]] = None,
        calendar_id: Optional[str] = None,
    ) -> None:
        self.id = session_id
        self.tools_map = tools_map
        self.lock = threading.RLock()
        self.removed_part = list(removed_part) if removed_part else None
        self.tool_profile: Optional[set] = (
            set(tool_profile) if tool_profile is not None else None
        )

        self.state = ContextState()
        self.fixed = FixedContext()
        # calendar_id is Optional at runtime; only pass it when the task has one.
        self.task_config = (
            TaskConfig(calendar_id=calendar_id) if calendar_id else TaskConfig()
        )
        config = dict(context_init_config or {})
        # Install both contextvars around initialization: ContextState
        # .update_state reads fixed_context for the waypoint policy check,
        # mirroring the client-side _init_context_state in run.py.
        token_fixed = fixed_context.set(self.fixed)
        token_state = context_state.set(self.state)
        try:
            # Fixed context first: the waypoint check compares against it.
            self.fixed.update_state(**config)
            self.state.update_state(**config)
        finally:
            fixed_context.reset(token_fixed)
            context_state.reset(token_state)

        # Tools append to these contextvars; the server owns the lists so the
        # client can merge them into its own scoring context later.
        self.policy_errors: List[str] = []
        self.tool_execution_errors: List[str] = []
        self.call_count = 0

    # -- tools -----------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        from mcp_server.schema import mcp_tool_schema

        with self.lock:
            if self.tool_profile is None:
                names = sorted(self.tools_map)
            else:
                names = sorted(n for n in self.tools_map if n in self.tool_profile)
            return [mcp_tool_schema(self.tools_map[n].get_info()) for n in names]

    def set_tool_profile(self, names: List[str]) -> None:
        with self.lock:
            self.tool_profile = set(names)

    def _removed_parameter_note(self, name: str) -> str:
        if not self.removed_part:
            return ""
        removed_parameters = [
            part
            for part in self.removed_part
            if "." in part and part.split(".")[0] == name
        ]
        return REMOVED_ARGUMENT_NOTE if removed_parameters else ""

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one tool call and return the MCP tools/call result."""
        with self.lock:
            self.call_count += 1
            if name not in self.tools_map or (
                self.tool_profile is not None and name not in self.tool_profile
            ):
                return self._result(f"Unknown action {name}", is_error=True)

            token_state = context_state.set(self.state)
            token_fixed = fixed_context.set(self.fixed)
            token_task = task_config.set(self.task_config)
            token_policy = policy_errors_during_runtime.set(self.policy_errors)
            token_tool_errors = tool_execution_errors_during_runtime.set(
                self.tool_execution_errors
            )
            try:
                observation = self.tools_map[name].invoke(
                    data={}, **dict(arguments or {})
                )
                return self._result(observation, is_error=False)
            except Exception as exc:  # tool failure is data, not a crash
                self.tool_execution_errors.append(
                    f"{name}: {type(exc).__name__}: {exc}"
                )
                note = self._removed_parameter_note(name)
                return self._result(f"Error: {exc}{note}", is_error=True)
            finally:
                context_state.reset(token_state)
                fixed_context.reset(token_fixed)
                task_config.reset(token_task)
                policy_errors_during_runtime.reset(token_policy)
                tool_execution_errors_during_runtime.reset(token_tool_errors)

    @staticmethod
    def _result(observation: Any, is_error: bool) -> Dict[str, Any]:
        import json

        text = observation if isinstance(observation, str) else json.dumps(
            observation, ensure_ascii=False
        )
        structured: Any = None
        if isinstance(observation, str):
            try:
                structured = json.loads(observation)
            except (TypeError, ValueError):
                structured = None
        elif isinstance(observation, (dict, list)):
            structured = observation
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": structured,
            "isError": is_error,
        }

    # -- snapshot --------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.id,
                "call_count": self.call_count,
                "state_hash": context_state_hash(self.state),
                "context_state": self.state.model_dump(mode="json"),
                "fixed_context": self.fixed.model_dump(mode="json"),
                "policy_errors": list(self.policy_errors),
                "tool_execution_errors": list(self.tool_execution_errors),
            }


def is_user_action(name: str) -> bool:
    """True for action names that address the user rather than a tool."""
    return name in USER_AS_A_TOOL_ACTION_NAMES
