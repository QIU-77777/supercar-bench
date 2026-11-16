"""End-to-end self-check for the MCP tool transport.

Runs without model credentials:

* a local fake OpenAI-compatible endpoint plays the simulated user and always
  ends the conversation with the STOP control keyword;
* a scripted agent replays the task's ground-truth actions, so a correct
  transport must score reward 1.0;
* the same run is repeated with the in-process transport, and both rewards
  must match.

Also checks the MCP server surface directly: tools/list shape, state
mutation through tools/call, snapshot hashing, allowlist enforcement and
unknown-tool handling.

Usage:
    python scripts/selfcheck_mcp.py [--task-index 0] [--keep-going]

Requires network access to HuggingFace (the task set is downloaded on first
use). Mock data comes from CAR_BENCH_MOCK_DATA_DIR when that variable is set,
otherwise a tiny local fixture is generated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------
# Fake OpenAI-compatible endpoint for the simulated user
# --------------------------------------------------------------------------

USER_REPLY = {
    "reasoning_if_continue_or_stop": "self-check fake user",
    "user_message": "Thanks, that is all.",
    "conversation_control_keyword": "STOP",
}


class FakeUserHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = {
            "id": "chatcmpl-selfcheck",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-user",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(USER_REPLY)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_fake_user(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), FakeUserHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# --------------------------------------------------------------------------
# Scripted agent: replays ground-truth actions, then answers
# --------------------------------------------------------------------------


class ScriptedAgent:
    """Agent that performs the task's ground-truth actions one per message."""

    def __init__(self, actions: List[Any]) -> None:
        self.actions = list(actions)
        self.step = 0

    def get_init_state(self, system_prompt: str, initial_observation: str):
        from car_bench.types import AgentState

        return AgentState(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_observation},
            ],
            start_time_s=time.time(),
        )

    def generate_next_message(self, state, tools_info):
        if self.step < len(self.actions):
            action = self.actions[self.step]
            self.step += 1
            if action.name == "respond":
                message = {"role": "assistant", "content": action.kwargs.get("content", "Done.")}
            else:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"selfcheck-call-{self.step}",
                            "type": "function",
                            "function": {
                                "name": action.name,
                                "arguments": json.dumps(action.kwargs),
                            },
                        }
                    ],
                }
            state.messages.append(message)
            return message, state
        message = {"role": "assistant", "content": "All actions are complete."}
        state.messages.append(message)
        return message, state


# --------------------------------------------------------------------------
# Mock data fixture
# --------------------------------------------------------------------------


def ensure_mock_data_dir() -> str:
    existing = os.environ.get("CAR_BENCH_MOCK_DATA_DIR")
    if existing:
        return existing
    root = Path(tempfile.mkdtemp(prefix="supercar-selfcheck-data-"))
    for folder in ("navigation", "productivity_and_communication"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for name in (
        "locations.jsonl",
        "pois.jsonl",
        "weather.jsonl",
        "routes_location_location.jsonl",
        "routes_location_poi.jsonl",
        "routes_poi_location.jsonl",
        "routes_metadata.jsonl",
        "routes_index.jsonl",
        "contacts.jsonl",
        "calendars.jsonl",
    ):
        (root / "navigation" if name != "contacts.jsonl" and name != "calendars.jsonl" else root / "productivity_and_communication", )
        pass
    nav = root / "navigation"
    prod = root / "productivity_and_communication"
    for name in ("locations.jsonl", "pois.jsonl", "weather.jsonl",
                 "routes_location_location.jsonl", "routes_location_poi.jsonl",
                 "routes_poi_location.jsonl", "routes_metadata.jsonl", "routes_index.jsonl"):
        (nav / name).write_text("", encoding="utf-8")
    for name in ("contacts.jsonl", "calendars.jsonl"):
        (prod / name).write_text("", encoding="utf-8")
    os.environ["CAR_BENCH_MOCK_DATA_DIR"] = str(root)
    return str(root)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


class Checker:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.passes: List[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passes.append(name)
            print(f"  ✅ {name}")
        else:
            self.failures.append(f"{name}: {detail}")
            print(f"  ❌ {name} — {detail}")

    def done(self) -> int:
        print(f"\n通过 {len(self.passes)} 项，失败 {len(self.failures)} 项")
        for failure in self.failures:
            print(f"  - {failure}")
        return 1 if self.failures else 0


def http_json(url: str, method: str = "GET", body: Dict[str, Any] | None = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def check_server_surface(checker: Checker, server_url: str) -> None:
    print("\n[1] MCP 服务端接口")
    health = http_json(server_url + "/health")
    checker.check("health 返回 ok", health.get("status") == "ok", str(health))

    created = http_json(
        server_url + "/sessions",
        "POST",
        {"context_init_config": {"sunroof_position": 0}, "tool_profile": None},
    )
    session_id = created.get("session_id")
    checker.check("创建会话返回 session_id", bool(session_id), str(created)[:200])
    initial_hash = created.get("state_hash")
    checker.check("创建时返回 state_hash", bool(initial_hash), str(created)[:200])

    tools = http_json(
        server_url + f"/mcp/{session_id}", "POST",
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )["result"]["tools"]
    from car_bench.envs.car_voice_assistant.tools import ALL_TOOLS

    names = {tool["name"] for tool in tools}
    checker.check(
        "tools/list 返回全部工具",
        len(tools) == len(ALL_TOOLS),
        f"期望 {len(ALL_TOOLS)}，实际 {len(tools)}",
    )
    checker.check(
        "工具定义使用 inputSchema",
        all("inputSchema" in tool and "name" in tool for tool in tools),
        "存在缺少 inputSchema 的工具",
    )

    call = http_json(
        server_url + f"/mcp/{session_id}", "POST",
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "open_close_sunroof", "arguments": {"percentage": 50}}},
    )["result"]
    checker.check("tools/call 返回文本内容", call.get("content", [{}])[0].get("text", "").startswith("{"), str(call)[:200])
    checker.check("tools/call 标记 isError=false", call.get("isError") is False, str(call)[:200])

    snapshot = http_json(server_url + f"/sessions/{session_id}/snapshot")
    checker.check(
        "快照反映状态变化",
        snapshot["context_state"]["sunroof_position"] == 50,
        f"sunroof_position={snapshot['context_state'].get('sunroof_position')}",
    )
    checker.check("状态变化后哈希改变", snapshot["state_hash"] != initial_hash, "哈希未变")

    bad = http_json(
        server_url + f"/mcp/{session_id}", "POST",
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "definitely_not_a_tool", "arguments": {}}},
    )["result"]
    checker.check(
        "未知工具返回 Unknown action",
        "Unknown action" in bad.get("content", [{}])[0].get("text", ""),
        str(bad)[:200],
    )

    profile = http_json(
        server_url + f"/sessions/{session_id}/tool_profile", "POST",
        {"tool_profile": ["get_sunroof_and_sunshade_position"]},
    )
    checker.check("设置工具白名单成功", profile.get("ok") is True, str(profile))
    denied = http_json(
        server_url + f"/mcp/{session_id}", "POST",
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "open_close_sunroof", "arguments": {"percentage": 10}}},
    )["result"]
    checker.check(
        "白名单外工具被拒绝",
        "Unknown action" in denied.get("content", [{}])[0].get("text", ""),
        str(denied)[:200],
    )

    deleted = http_json(server_url + f"/sessions/{session_id}", "DELETE")
    checker.check("删除会话成功", deleted.get("ok") is True, str(deleted))
    try:
        http_json(server_url + f"/sessions/{session_id}/snapshot")
        checker.check("删除后快照不可读", False, "仍返回 200")
    except urllib.error.HTTPError as exc:
        checker.check("删除后快照不可读", exc.code == 404, f"状态码 {exc.code}")


def run_one_task(
    server_url: str,
    user_base_url: str,
    task_index: int,
    task_type: str = "base",
    task_split: str = "train",
) -> Dict[str, Any]:
    from car_bench.envs import get_env

    env = get_env(
        "car_voice_assistant",
        user_strategy="llm",
        user_model="fake-user",
        user_provider="openai",
        policy_evaluator_strategy="llm",
        policy_evaluator_model="fake-user",
        policy_evaluator_provider="openai",
        task_type=task_type,
        task_split=task_split,
        task_index=task_index,
        evaluate_policy=False,
        score_tool_execution_errors=False,
        score_policy_errors=False,
        mcp_server_url=server_url,
    )
    result = _run_with_initialized_context(env, task_index)
    return {"result": result, "task_id": env.tasks[task_index].task_id}


def _run_with_initialized_context(env, task_index: int):
    """Execute a task with the ambient context state initialized.

    Mirrors run.py: the state pair stays installed for the whole run because
    the in-process transport (used by the ground-truth replay) reads it.
    """
    from car_bench.envs.car_voice_assistant.context.dynamic_context_state import (
        ContextState,
        context_state,
    )
    from car_bench.envs.car_voice_assistant.context.fixed_context import (
        FixedContext,
        fixed_context,
    )
    from car_bench.orchestrator import AgentOrchestrator

    task = env.tasks[task_index]
    state, fixed = ContextState(), FixedContext()
    token_state, token_fixed = context_state.set(state), fixed_context.set(fixed)
    try:
        state.update_state(**task.context_init_config)
        fixed.update_state(**task.context_init_config)
        agent = ScriptedAgent(task.actions)
        return AgentOrchestrator(agent).execute(env=env, task_index=task_index)
    finally:
        context_state.reset(token_state)
        fixed_context.reset(token_fixed)


def check_e2e(checker: Checker, server_url: str, user_base_url: str, task_index: int) -> None:
    print(f"\n[2] 端到端：MCP 传输（任务 {task_index}）")
    outcome = run_one_task(server_url, user_base_url, task_index)
    result = outcome["result"]
    info = result.info
    checker.check("run 返回 reward", isinstance(result.reward, float), str(result.reward))
    checker.check(
        "ground-truth 重放得到满分",
        abs(result.reward - 1.0) < 1e-6,
        f"reward={result.reward} info={json.dumps(info, ensure_ascii=False)[:400]}",
    )
    reward_info = (info.get("reward_info") or {}).get("info") or {}
    checker.check(
        "状态评分分量齐全",
        reward_info.get("r_actions_final") == 1.0
        and reward_info.get("r_actions_intermediate") == 1.0,
        f"r_actions_final={reward_info.get('r_actions_final')} "
        f"r_actions_intermediate={reward_info.get('r_actions_intermediate')}",
    )
    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    checker.check("对话包含工具结果", len(tool_messages) > 0, "没有 role=tool 消息")

    print(f"\n[3] 端到端：进程内传输（同一任务）")
    from car_bench.envs import get_env

    env = get_env(
        "car_voice_assistant",
        user_strategy="llm",
        user_model="fake-user",
        user_provider="openai",
        policy_evaluator_strategy="llm",
        policy_evaluator_model="fake-user",
        policy_evaluator_provider="openai",
        task_type="base",
        task_split="train",
        task_index=task_index,
        evaluate_policy=False,
        score_tool_execution_errors=False,
        score_policy_errors=False,
    )
    result_inproc = _run_with_initialized_context(env, task_index)
    checker.check(
        "两种传输得分一致",
        abs(result_inproc.reward - result.reward) < 1e-6,
        f"mcp={result.reward} in-process={result_inproc.reward}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="supercar-bench MCP self-check")
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args()

    ensure_mock_data_dir()

    # The fake user endpoint doubles as the policy evaluator endpoint.
    os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:18911/v1"
    os.environ["OPENAI_API_KEY"] = "self-check"

    print("启动模拟用户端点与 MCP 服务端 ...")
    user_server = start_fake_user(18911)

    from mcp_server.server import create_server

    mcp_server = create_server("127.0.0.1", 18912)
    threading.Thread(target=mcp_server.serve_forever, daemon=True).start()
    server_url = "http://127.0.0.1:18912"

    checker = Checker()
    try:
        check_server_surface(checker, server_url)
        check_e2e(checker, server_url, "http://127.0.0.1:18911/v1", args.task_index)
    finally:
        mcp_server.shutdown()
        mcp_server.server_close()
        user_server.shutdown()
        user_server.server_close()
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
