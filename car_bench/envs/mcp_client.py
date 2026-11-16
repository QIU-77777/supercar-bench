"""HTTP client for the supercar-bench MCP tool server.

Used by ``car_bench.envs.base.Env`` when ``mcp_server_url`` is configured:
agent tool calls travel over MCP instead of in-process invocation, while the
ground-truth replay inside the reward calculation keeps running in-process.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class MCPToolError(RuntimeError):
    """Transport or protocol level failure against the MCP server."""


class MCPToolClient:
    def __init__(self, base_url: str, timeout_sec: float = 60.0) -> None:
        if not base_url:
            raise ValueError("mcp_server_url must not be empty")
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._next_id = 0

    # -- transport -------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = (
            json.dumps(body, ensure_ascii=False).encode("utf-8")
            if body is not None
            else None
        )
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise MCPToolError(f"MCP {method} {path} failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise MCPToolError(f"MCP server unreachable at {self.base_url}: {exc}") from exc
        try:
            value = json.loads(raw) if raw else {}
        except (TypeError, ValueError) as exc:
            raise MCPToolError(f"MCP {method} {path} returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise MCPToolError(f"MCP {method} {path} returned a non-object body")
        return value

    def _rpc(self, session_id: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._next_id += 1
        response = self._request(
            "POST",
            f"/mcp/{session_id}",
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params},
        )
        if "error" in response:
            raise MCPToolError(f"MCP rpc error: {response['error']}")
        result = response.get("result")
        return result if isinstance(result, dict) else {}

    # -- API --------------------------------------------------------------

    def create_session(
        self,
        context_init_config: Optional[Dict[str, Any]] = None,
        tool_profile: Optional[List[str]] = None,
        removed_part: Optional[List[str]] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a session; returns the initial snapshot including state_hash."""
        return self._request(
            "POST",
            "/sessions",
            {
                "context_init_config": context_init_config or {},
                "tool_profile": tool_profile,
                "removed_part": removed_part,
                "calendar_id": calendar_id,
            },
        )

    def list_tools(self, session_id: str) -> List[Dict[str, Any]]:
        return self._rpc(session_id, "tools/list", {}).get("tools", [])

    def call_tool(self, session_id: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._rpc(
            session_id, "tools/call", {"name": name, "arguments": arguments or {}}
        )

    def call_tool_text(self, session_id: str, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call and return only the observation text."""
        result = self.call_tool(session_id, name, arguments)
        content = result.get("content") or []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
        return ""

    def snapshot(self, session_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/sessions/{session_id}/snapshot")

    def set_tool_profile(self, session_id: str, names: List[str]) -> Dict[str, Any]:
        return self._request(
            "POST", f"/sessions/{session_id}/tool_profile", {"tool_profile": list(names)}
        )

    def delete_session(self, session_id: str) -> None:
        try:
            self._request("DELETE", f"/sessions/{session_id}")
        except MCPToolError:
            pass  # best effort; a missing session is already gone
