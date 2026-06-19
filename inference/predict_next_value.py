#!/usr/bin/env python3
"""
Predict task-specific value/label at a user-specified prediction time.

Input:
- base CLMBR backbone (from HF)
- local adapter weights (selected by task)
- one patient timeline JSON
- prediction_time

Output:
- task head prediction from classifier_head (required for this workflow)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from diablo.block_linear import BlockLinear
from inference.task_catalog import describe_task, enrich_task_head_prediction, normalize_task_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict task value/label at prediction_time.")
    p.add_argument(
        "--task",
        type=str,
        default="",
        help="Task name or adapter folder name. Examples: guo_los, guo_los_b8.",
    )
    p.add_argument(
        "--list_tasks",
        action="store_true",
        help="List all available local adapter tasks under weights/adapters and exit.",
    )
    p.add_argument(
        "--describe_task",
        action="store_true",
        help="Print task metadata (labels, problem type) and exit.",
    )
    p.add_argument(
        "--adapter_dir",
        type=str,
        default="",
        help="Optional adapter directory override (defaults to weights/adapters/<task>_b8).",
    )
    p.add_argument(
        "--base_model_name",
        type=str,
        default="StanfordShahLab/clmbr-t-base",
        help="Base pretrained model name/path.",
    )
    p.add_argument("--hf_token", type=str, default="", help="Optional Hugging Face token.")
    p.add_argument("--hf_token_env", type=str, default="HF_TOKEN")
    p.add_argument("--patient_json", type=str, default="", help="Path to one patient JSON file.")
    p.add_argument(
        "--prediction_time",
        type=str,
        default="",
        help="Prediction time (e.g. 2021-01-01 12:00:00 or ISO format).",
    )
    p.add_argument("--output_json", type=str, default="", help="Optional path to save JSON output.")
    return p.parse_args()


def _parse_time(s: str) -> datetime:
    s = s.strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(s)


def _load_patient(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        patient = json.load(f)
    if "patient_id" not in patient or "events" not in patient:
        raise ValueError("patient_json must contain keys: patient_id, events")
    return patient


def _truncate_patient_at_time(patient: dict[str, Any], prediction_time: datetime) -> dict[str, Any]:
    events = []
    for e in patient["events"]:
        t = _parse_time(str(e["time"])) if not isinstance(e["time"], datetime) else e["time"]
        if t <= prediction_time:
            events.append({"time": t, "measurements": e.get("measurements", [])})
    if len(events) == 0:
        raise ValueError("No events <= prediction_time; cannot run prediction.")
    return {
        "patient_id": int(patient["patient_id"]),
        "static_measurements": patient.get("static_measurements", []),
        "events": events,
    }


def _resolve_adapter_dir(project_root: Path, task: str, adapter_dir: str) -> Path:
    if adapter_dir:
        p = Path(adapter_dir)
        return p if p.is_absolute() else (project_root / p)
    if not task:
        raise ValueError("Please provide --task or --adapter_dir.")
    root = project_root / "weights" / "adapters"
    t = task.strip()
    candidates = [root / t] if t.endswith("_b8") else [root / f"{t}_b8", root / t]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No adapter dir found for task={t}. Tried: {[str(x) for x in candidates]}")


def _load_adapter_config(adapter_dir: Path) -> dict[str, object]:
    adapter_cfg_path = adapter_dir / "adapter_config.pt"
    if not adapter_cfg_path.exists():
        return {}
    return torch.load(adapter_cfg_path, map_location="cpu")


def _load_manifest(project_root: Path) -> dict[str, object]:
    manifest_path = project_root / "weights" / "adapters" / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_available_tasks(project_root: Path) -> list[dict[str, object]]:
    root = project_root / "weights" / "adapters"
    if not root.exists():
        return []
    manifest = _load_manifest(project_root)
    synced_tasks = set((manifest.get("tasks") or {}).keys())
    rows: list[dict[str, object]] = []
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        task_name = p.name[:-3] if p.name.endswith("_b8") else p.name
        cfg = _load_adapter_config(p)
        info = describe_task(task_name, cfg if isinstance(cfg, dict) else None)
        weight_status = "synced_from_ft" if p.name in synced_tasks else "local_only"
        rows.append(
            {
                "task": normalize_task_name(task_name),
                "adapter_dir": str(p),
                "display_name": info["display_name"],
                "problem_type": info["problem_type"],
                "num_labels": info["num_labels"],
                "weight_status": weight_status,
            }
        )
    return rows


def inject_kwargs_from_adapter_config(cfg: dict[str, object]) -> dict[str, object]:
    return {
        "dropout": float(cfg.get("adapter_dropout", 0.05)),
        "num_blocks": int(cfg["adapter_num_blocks"]),
        "target_modules": list(cfg["adapter_target_modules"]),
    }


def _is_target_module(name: str, target_modules: list[str]) -> bool:
    return name.split(".")[-1] in target_modules


def _get_parent_module(model: torch.nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_blockdiag_adapters(
    model: torch.nn.Module, dropout: float, num_blocks: int, target_modules: list[str]
) -> dict[str, int]:
    stats = {"matched": 0, "replaced": 0}
    for param in model.parameters():
        param.requires_grad = False
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and _is_target_module(name, target_modules)
    ]
    for name, module in candidates:
        stats["matched"] += 1
        parent, attr_name = _get_parent_module(model, name)
        wrapped = BlockLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            num_blocks=num_blocks,
            bias=module.bias is not None,
            drop_out=dropout,
        )
        wrapped.linear.weight.data = module.weight.data.clone()
        if module.bias is not None:
            wrapped.linear.bias.data = module.bias.data.clone()
        wrapped.to(module.weight.device).to(module.weight.dtype)
        wrapped.linear.weight.requires_grad_(False)
        if wrapped.linear.bias is not None:
            wrapped.linear.bias.requires_grad_(False)
        wrapped.block_A.requires_grad_(True)
        setattr(parent, attr_name, wrapped)
        stats["replaced"] += 1
    return stats


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    if args.list_tasks:
        tasks = _list_available_tasks(project_root)
        print("Available tasks:" if tasks else "No adapter tasks found.")
        for row in tasks:
            status = row["weight_status"]
            status_note = "" if status == "synced_from_ft" else " [local only; re-run sync when FT best_checkpoint ready]"
            print(
                f"- {row['task']}: {row['display_name']} "
                f"({row['problem_type']}, num_labels={row['num_labels']}){status_note}"
            )
        return

    adapter_dir = _resolve_adapter_dir(project_root, args.task, args.adapter_dir).resolve()
    adapter_cfg_preview = _load_adapter_config(adapter_dir)
    if args.describe_task:
        info = describe_task(args.task or adapter_dir.name, adapter_cfg_preview)
        print(json.dumps(info, indent=2))
        return

    if not args.patient_json or not args.prediction_time:
        raise ValueError(
            "--patient_json and --prediction_time are required unless --list_tasks or --describe_task is set."
        )
    patient_json = Path(args.patient_json).resolve()
    prediction_time = _parse_time(args.prediction_time)

    adapter_cfg_path = adapter_dir / "adapter_config.pt"
    adapter_pt_path = adapter_dir / "adapter_model.pt"
    adapter_bin_path = adapter_dir / "adapter_model.bin"
    classifier_pt_path = adapter_dir / "classifier_head.pt"
    classifier_bin_path = adapter_dir / "classifier_head.bin"
    if adapter_pt_path.exists():
        adapter_path = adapter_pt_path
    elif adapter_bin_path.exists():
        adapter_path = adapter_bin_path
    else:
        raise FileNotFoundError(f"Missing adapter_model.pt/bin in {adapter_dir}")
    adapter_cfg = adapter_cfg_preview if adapter_cfg_preview else torch.load(adapter_cfg_path, map_location="cpu")
    adapter_state = torch.load(adapter_path, map_location="cpu")
    has_classifier = classifier_pt_path.exists() or classifier_bin_path.exists()
    if not has_classifier:
        raise FileNotFoundError(
            f"No classifier head found in {adapter_dir}. "
            "Expected classifier_head.pt or classifier_head.bin."
        )

    hf_token = args.hf_token.strip() or os.environ.get(args.hf_token_env, "").strip()
    from_pretrained_kwargs = {"token": hf_token} if hf_token else {}

    import femr.models.processor
    import femr.models.tokenizer
    import femr.models.transformer

    tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(
        args.base_model_name, **from_pretrained_kwargs
    )
    model = femr.models.transformer.FEMRModel.from_pretrained(args.base_model_name, **from_pretrained_kwargs)
    inject_kwargs = inject_kwargs_from_adapter_config(adapter_cfg)
    inject_blockdiag_adapters(model, **inject_kwargs)
    model.load_state_dict(adapter_state, strict=False)
    model.eval()
    batch_processor = femr.models.processor.FEMRBatchProcessor(tokenizer, task=None)

    patient_raw = _load_patient(patient_json)
    patient = _truncate_patient_at_time(patient_raw, prediction_time)

    raw_batch = batch_processor.convert_patient(patient, tensor_type="pt")
    batch = batch_processor.collate([raw_batch])

    with torch.no_grad():
        _, result = model(**batch)
        representations = result["representations"]
        target_idx = int(len(result["timestamps"]) - 1)

        hidden = representations[target_idx]

    task_head_output = None
    if has_classifier:
        cls_path = classifier_pt_path if classifier_pt_path.exists() else classifier_bin_path
        cls_state = torch.load(cls_path, map_location="cpu")
        if not isinstance(cls_state, dict):
            raise ValueError(f"Unsupported classifier checkpoint format: {type(cls_state)}")
        if "state_dict" in cls_state and isinstance(cls_state["state_dict"], dict):
            cls_state = cls_state["state_dict"]
        if "classifier.weight" not in cls_state:
            raise KeyError(f"classifier.weight not found in classifier checkpoint: {cls_path}")
        out_features = int(cls_state["classifier.weight"].shape[0])
        in_features = int(cls_state["classifier.weight"].shape[1])
        classifier = nn.Linear(in_features, out_features)
        classifier.load_state_dict(
            {
                "weight": cls_state["classifier.weight"],
                "bias": cls_state.get("classifier.bias", torch.zeros(out_features)),
            }
        )
        classifier.eval()

        with torch.no_grad():
            cls_logits = classifier(hidden.unsqueeze(0)).squeeze(0)

        problem_type = str(adapter_cfg.get("problem_type", "single_label_classification"))
        if problem_type == "multi_label_classification":
            probs_cls = torch.sigmoid(cls_logits)
            task_head_output = {
                "problem_type": problem_type,
                "logits": [float(x) for x in cls_logits.tolist()],
                "probabilities": [float(x) for x in probs_cls.tolist()],
                "predicted_labels": [int(float(x) >= 0.5) for x in probs_cls.tolist()],
            }
        else:
            probs_cls = torch.softmax(cls_logits, dim=-1)
            pred_idx = int(torch.argmax(probs_cls).item())
            task_head_output = {
                "problem_type": problem_type,
                "logits": [float(x) for x in cls_logits.tolist()],
                "probabilities": [float(x) for x in probs_cls.tolist()],
                "predicted_label": pred_idx,
            }

    task_name = normalize_task_name(args.task if args.task else adapter_dir.name)
    task_info = describe_task(task_name, adapter_cfg if isinstance(adapter_cfg, dict) else None)
    if task_head_output is not None:
        task_head_output = enrich_task_head_prediction(task_name, adapter_cfg, task_head_output)

    output = {
        "task": task_name,
        "display_name": task_info["display_name"],
        "adapter_dir": str(adapter_dir),
        "patient_id": int(patient["patient_id"]),
        "prediction_time": prediction_time.isoformat(sep=" "),
        "num_events_used": len(patient["events"]),
        "task_head_prediction": task_head_output,
    }

    print(json.dumps(output, indent=2))
    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Saved prediction output: {out_path}")


if __name__ == "__main__":
    main()

