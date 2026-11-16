"""HTTP + JSON-RPC front end for the MCP tool server."""

from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Type
from urllib.parse import urlsplit

from car_bench.envs.tool import Tool
from mcp_server.session import Session


class SessionStore:
    """Thread-safe registry of live sessions."""

    def __init__(self, tools: Optional[List[Type[Tool]]] = None) -> None:
        if tools is None:
            from car_bench.envs.car_voice_assistant.tools import ALL_TOOLS

            tools = list(ALL_TOOLS)
        self.tools_map: Dict[str, Type[Tool]] = {
            tool.get_info()["function"]["name"]: tool for tool in tools
        }
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(
        self,
        context_init_config: Optional[Dict[str, Any]] = None,
        tool_profile: Optional[List[str]] = None,
        removed_part: Optional[List[str]] = None,
        calendar_id: Optional[str] = None,
    ) -> Session:
        session_id = uuid.uuid4().hex
        session = Session(
            session_id=session_id,
            tools_map=self.tools_map,
            context_init_config=context_init_config,
            tool_profile=tool_profile,
            removed_part=removed_part,
            calendar_id=calendar_id,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session: {session_id}")
        return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


class MCPRequestHandler(BaseHTTPRequestHandler):
    """Routes the MCP HTTP surface.

    POST /sessions                        create a session
    POST /mcp/<session_id>                JSON-RPC: tools/list, tools/call
    GET  /sessions/<session_id>/snapshot  state snapshot for scoring
    POST /sessions/<session_id>/tool_profile  restrict the tool allowlist
    DELETE /sessions/<session_id>         drop a session
    GET  /health                          liveness
    """

    server_version = "supercar-bench-mcp/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def store(self) -> SessionStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return  # keep test output clean

    # -- helpers ---------------------------------------------------------

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            value = json.loads(raw.decode("utf-8")) if raw else {}
        except (TypeError, ValueError):
            self._send_json(400, {"error": "request body is not valid JSON"})
            return None
        if not isinstance(value, dict):
            self._send_json(400, {"error": "request body must be an object"})
            return None
        return value

    def _session_id(self, prefix: str) -> Optional[str]:
        path = urlsplit(self.path).path
        if not path.startswith(prefix):
            self._send_json(404, {"error": f"expected path prefix {prefix}"})
            return None
        session_id = path[len(prefix) :].strip("/")
        if not session_id:
            self._send_json(404, {"error": "missing session id"})
            return None
        return session_id

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok", "sessions": len(self.store)})
            return
        if path.startswith("/sessions/") and path.endswith("/snapshot"):
            session_id = self._session_id("/sessions/")
            if session_id is None:
                return
            session_id = session_id[: -len("/snapshot")]
            try:
                session = self.store.get(session_id)
            except KeyError:
                self._send_json(404, {"error": "unknown session"})
                return
            self._send_json(200, session.snapshot())
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        body = self._read_json()
        if body is None:
            return

        if path == "/sessions":
            session = self.store.create(
                context_init_config=body.get("context_init_config"),
                tool_profile=body.get("tool_profile"),
                removed_part=body.get("removed_part"),
                calendar_id=body.get("calendar_id"),
            )
            self._send_json(200, session.snapshot())
            return

        if path.startswith("/mcp/"):
            session_id = self._session_id("/mcp/")
            if session_id is None:
                return
            try:
                session = self.store.get(session_id)
            except KeyError:
                self._send_json(404, {"error": "unknown session"})
                return
            self._handle_rpc(session, body)
            return

        if path.startswith("/sessions/") and path.endswith("/tool_profile"):
            session_id = self._session_id("/sessions/")
            if session_id is None:
                return
            session_id = session_id[: -len("/tool_profile")]
            try:
                session = self.store.get(session_id)
            except KeyError:
                self._send_json(404, {"error": "unknown session"})
                return
            profile = body.get("tool_profile")
            if not isinstance(profile, list) or not all(
                isinstance(name, str) for name in profile
            ):
                self._send_json(400, {"error": "tool_profile must be a string list"})
                return
            session.set_tool_profile(profile)
            self._send_json(200, {"ok": True, "tool_count": len(profile)})
            return

        self._send_json(404, {"error": "not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path.startswith("/sessions/"):
            session_id = path[len("/sessions/") :].strip("/")
            if self.store.delete(session_id):
                self._send_json(200, {"ok": True})
            else:
                self._send_json(404, {"error": "unknown session"})
            return
        self._send_json(404, {"error": "not found"})

    # -- JSON-RPC --------------------------------------------------------

    def _handle_rpc(self, session: Session, request: Dict[str, Any]) -> None:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        def reply(result: Dict[str, Any]) -> None:
            self._send_json(
                200, {"jsonrpc": "2.0", "id": request_id, "result": result}
            )

        def fail(code: int, message: str) -> None:
            self._send_json(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": code, "message": message},
                },
            )

        if method == "tools/list":
            reply({"tools": session.list_tools()})
            return
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str) or not name:
                fail(-32602, "tools/call requires a string name")
                return
            if not isinstance(arguments, dict):
                fail(-32602, "tools/call arguments must be an object")
                return
            reply(session.call_tool(name, arguments))
            return
        fail(-32601, f"unknown method: {method}")


class MCPHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple, store: SessionStore) -> None:
        super().__init__(address, MCPRequestHandler)
        self.store = store
        self.daemon_threads = True


def create_server(
    host: str = "127.0.0.1",
    port: int = 8090,
    tools: Optional[List[Type[Tool]]] = None,
) -> MCPHTTPServer:
    """Build (not start) an MCP server bound to host:port."""
    return MCPHTTPServer((host, port), SessionStore(tools))
