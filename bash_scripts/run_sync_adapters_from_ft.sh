#!/usr/bin/env bash
set -euo pipefail

# Copy FT best_checkpoint adapter weights into weights/adapters/<task>_b8/.
# Usage: bash bash_scripts/run_sync_adapters_from_ft.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FT_CHECKPOINT_ROOT="${FT_CHECKPOINT_ROOT:-/network/rit/lab/wang_lab/zjguo/FT/checkpoint/ft}"
ADAPTER_ROOT="${PROJECT_ROOT}/weights/adapters"
MANIFEST="${ADAPTER_ROOT}/manifest.json"

mkdir -p "${ADAPTER_ROOT}"

python3 - <<PY
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

project_root = Path("${PROJECT_ROOT}")
ft_root = Path("${FT_CHECKPOINT_ROOT}")
adapter_root = Path("${ADAPTER_ROOT}")
manifest_path = Path("${MANIFEST}")

required = ["adapter_config.pt", "adapter_model.bin", "classifier_head.bin"]

manifest = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "ft_checkpoint_root": str(ft_root),
    "tasks": {},
    "skipped": [],
}

if manifest_path.exists():
    try:
        manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest["tasks"] = {}
        manifest["skipped"] = []
    except json.JSONDecodeError:
        pass

for task_dir in sorted(ft_root.glob("*_b8")):
    task = task_dir.name
    best_dir = task_dir / "best_checkpoint"
    out_dir = adapter_root / task
    if not best_dir.is_dir():
        manifest["skipped"].append({"task": task, "reason": "missing best_checkpoint"})
        continue
    missing = [name for name in required if not (best_dir / name).is_file()]
    if missing:
        manifest["skipped"].append({"task": task, "reason": f"missing files: {missing}"})
        continue

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(best_dir / name, out_dir / name)

    manifest["tasks"][task] = {
        "task": task[:-3] if task.endswith("_b8") else task,
        "adapter_dir": str(out_dir),
        "source_best_checkpoint": str(best_dir),
        "files": required,
    }
    print(f"[OK] synced {task} <- {best_dir}")

manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Wrote manifest: {manifest_path}")
if manifest["skipped"]:
    print("Skipped:")
    for row in manifest["skipped"]:
        print(f"  - {row['task']}: {row['reason']}")
PY
