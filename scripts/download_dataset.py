"""Download the benchmark dataset into the repository's data/ directory.

Two parts:

* ``data/tasks/`` — the task files (about 1 MB total). Small enough to commit,
  so they are versioned with the code and evaluations run offline.
* ``data/mock_data/`` — the fixed reference data (about 686 MB). Not
  committable: two files exceed GitHub's 100 MB per-file limit, and the data
  is regenerable. It stays local and is ignored by git.

Usage:
    python scripts/download_dataset.py            # tasks + mock data
    python scripts/download_dataset.py --tasks-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TASKS_DIR = DATA_DIR / "tasks"
MOCK_DATA_DIR = DATA_DIR / "mock_data"

DATASET_REPO = "johanneskirmayr/car-bench-dataset"

TASK_FILES = [
    "base_train",
    "base_test",
    "disambiguation_train",
    "disambiguation_test",
    "hallucination_train",
    "hallucination_test",
]


def download_tasks() -> None:
    from huggingface_hub import hf_hub_download

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    for name in TASK_FILES:
        target = TASKS_DIR / f"{name}.jsonl"
        if target.is_file():
            print(f"已存在，跳过: {target}")
            continue
        cached = hf_hub_download(
            DATASET_REPO, f"tasks/{name}.jsonl", repo_type="dataset"
        )
        target.write_bytes(Path(cached).read_bytes())
        print(f"已下载: {target} ({target.stat().st_size / 1024:.0f} KB)")


def download_mock_data() -> None:
    from huggingface_hub import snapshot_download

    MOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"下载 mock_data 到 {MOCK_DATA_DIR}（约 686 MB，首次较慢）...")
    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        allow_patterns=["mock_data/**"],
        local_dir=str(DATA_DIR),
    )
    total = sum(f.stat().st_size for f in MOCK_DATA_DIR.rglob("*.jsonl"))
    print(f"mock_data 就绪：{total / 1024 / 1024:.0f} MB，{len(list(MOCK_DATA_DIR.rglob('*.jsonl')))} 个文件")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载评测数据集到 data/ 目录")
    parser.add_argument("--tasks-only", action="store_true", help="只下载任务文件")
    args = parser.parse_args()

    download_tasks()
    if not args.tasks_only:
        download_mock_data()

    print("\n数据目录结构：")
    print(f"  任务:   {TASKS_DIR}  (纳入 git)")
    print(f"  资料:   {MOCK_DATA_DIR}  (本地保留，git 忽略)")
    print("\n加载顺序：CAR_BENCH_MOCK_DATA_DIR > data/mock_data > HuggingFace 下载")
    return 0


if __name__ == "__main__":
    sys.exit(main())
