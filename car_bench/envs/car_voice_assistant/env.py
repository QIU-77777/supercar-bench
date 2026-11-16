import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from car_bench.envs.base import Env
from car_bench.envs.car_voice_assistant.mock_data import load_data
from car_bench.envs.car_voice_assistant.tools import (
    ALL_TOOLS,
)
from car_bench.envs.car_voice_assistant.wiki import WIKI
from car_bench.envs.policy_evaluator import PolicyEvaluatorStrategy
from car_bench.envs.user.user import UserStrategy
from car_bench.types import Action, Task


# Default HuggingFace dataset repo ID
HF_DATASET_REPO_ID = "johanneskirmayr/car-bench-dataset"


def _repo_root() -> Path:
    """Repository root (three levels above this file)."""
    return Path(__file__).resolve().parents[3]


def _local_tasks_file(task_type: str, task_split: str) -> Optional[Path]:
    """Path to the vendored task file for this type/split, if present.

    Tasks live in ``data/tasks/<type>_<split>.jsonl`` after running
    ``scripts/download_dataset.py``. The file is small (about 1 MB for all
    six splits), so it is committed to the repository.
    """
    path = _repo_root() / "data" / "tasks" / f"{task_type}_{task_split}.jsonl"
    return path if path.is_file() else None


def _parse_task_row(d: Dict[str, Any]) -> Task:
    """Build a Task from one raw dataset row.

    ``context_init_config``, ``actions`` and ``removed_part`` are stored as
    JSON strings in the dataset and are parsed back here.
    """
    d = dict(d)
    d["context_init_config"] = json.loads(d["context_init_config"])
    d["actions"] = [Action(**a) for a in json.loads(d["actions"])]
    if d.get("removed_part") is not None:
        d["removed_part"] = json.loads(d["removed_part"])
    return Task(**d)


def _load_tasks(
    task_type: str,
    task_split: str,
    repo_id: str = HF_DATASET_REPO_ID,
) -> List[Task]:
    """Load tasks, preferring the vendored copy under data/tasks/.

    Falls back to the HuggingFace dataset when the local file is absent.
    """
    local = _local_tasks_file(task_type, task_split)
    if local is not None:
        tasks = []
        for line in local.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            tasks.append(_parse_task_row(json.loads(line)))
        print(f"Loaded {len(tasks)} tasks from local file: {local}")
        return tasks

    from datasets import load_dataset

    config_name = f"tasks_{task_type}"
    print(f"Loading tasks from HuggingFace: {repo_id} / {config_name} / {task_split}")
    ds = load_dataset(repo_id, config_name, split=task_split)

    tasks = []
    for row in ds:
        tasks.append(_parse_task_row(dict(row)))

    print(f"Loaded {len(tasks)} tasks from HuggingFace ({config_name}/{task_split})")
    return tasks


class MockCarVoiceAssistantDomainEnv(Env):
    def __init__(
        self,
        user_strategy: Union[str, UserStrategy] = UserStrategy.LLM,
        policy_evaluator_strategy: Union[
            str, UserStrategy
        ] = PolicyEvaluatorStrategy.LLM,
        user_model: str = "gpt-4.1-mini",
        policy_evaluator_model: str = "gpt-4.1-mini",
        user_provider: Optional[str] = None,
        policy_evaluator_provider: Optional[str] = None,
        user_thinking: bool = False,
        task_type: str = "base",
        task_split: str = "train",
        task_index: Optional[int] = None,
        evaluate_policy: Optional[bool] = False,
        score_tool_execution_errors: Optional[bool] = False,
        score_policy_errors: Optional[bool] = False,
        use_user_as_a_tool_tools: bool = False,
        mcp_server_url: Optional[str] = None,
        user_request_timeout_sec: float = 120.0,
        user_retry_backoff_sec: float = 5.0,
    ):
        # Load tasks from HuggingFace
        tasks = _load_tasks(task_type, task_split)

        all_tools_plus_user_as_a_tool = None
        wiki_with_user_as_a_tool = None
        if use_user_as_a_tool_tools:
            all_tools_plus_user_as_a_tool = ALL_TOOLS_PLUS_USER_AS_A_TOOL
            wiki_with_user_as_a_tool = WIKI

        super().__init__(
            data_load_func=load_data,
            tools=all_tools_plus_user_as_a_tool or ALL_TOOLS,
            tasks=tasks,
            wiki=wiki_with_user_as_a_tool or WIKI,
            user_strategy=user_strategy,
            user_model=user_model,
            user_thinking=user_thinking,
            policy_evaluator_strategy=policy_evaluator_strategy,
            policy_evaluator_model=policy_evaluator_model,
            user_provider=user_provider,
            policy_evaluator_provider=policy_evaluator_provider,
            task_index=task_index,
            evaluate_policy=evaluate_policy,
            score_tool_execution_errors=score_tool_execution_errors,
            score_policy_errors=score_policy_errors,
            mcp_server_url=mcp_server_url,
            user_request_timeout_sec=user_request_timeout_sec,
            user_retry_backoff_sec=user_retry_backoff_sec,
        )
        self.terminate_tools = ["call_phone_by_number"]
