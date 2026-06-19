"""Human-readable task metadata for inference output."""

from __future__ import annotations

from typing import Any

CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

LAB_CLASS_NAMES = [
    "normal",
    "abnormal_mild",
    "abnormal_moderate",
    "abnormal_severe",
]

TASK_DISPLAY_NAMES: dict[str, str] = {
    "chexpert": "CheXpert findings (14 labels)",
    "guo_los": "Prolonged length of stay",
    "guo_readmission": "30-day readmission",
    "guo_icu": "ICU transfer",
    "new_hypertension": "Hypertension onset",
    "new_hyperlipidemia": "Hyperlipidemia onset",
    "new_pancan": "Pancreatic cancer onset",
    "new_acutemi": "Acute MI onset",
    "new_celiac": "Celiac disease onset",
    "new_lupus": "Lupus onset",
    "lab_thrombocytopenia": "Thrombocytopenia",
    "lab_hyperkalemia": "Hyperkalemia",
    "lab_hyponatremia": "Hyponatremia",
    "lab_anemia": "Anemia",
    "lab_hypoglycemia": "Hypoglycemia",
}

BINARY_POSITIVE_LABEL = "positive"
BINARY_NEGATIVE_LABEL = "negative"


def normalize_task_name(task_name: str) -> str:
    name = task_name.strip()
    if name.endswith("_b8"):
        name = name[:-3]
    return name


def label_names_for_task(task_name: str, num_labels: int, problem_type: str) -> list[str]:
    task = normalize_task_name(task_name)
    if task == "chexpert" and num_labels == len(CHEXPERT_LABELS):
        return list(CHEXPERT_LABELS)
    if task.startswith("lab_") and num_labels == len(LAB_CLASS_NAMES):
        return list(LAB_CLASS_NAMES)
    if problem_type == "single_label_classification" and num_labels == 2:
        return [BINARY_NEGATIVE_LABEL, BINARY_POSITIVE_LABEL]
    return [f"class_{i}" for i in range(num_labels)]


def describe_task(task_name: str, adapter_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    task = normalize_task_name(task_name)
    cfg = adapter_cfg or {}
    num_labels = int(cfg.get("num_labels", 0))
    problem_type = str(cfg.get("problem_type", ""))
    if num_labels <= 0 or not problem_type:
        num_labels, problem_type = _default_head_spec(task)
    return {
        "task": task,
        "display_name": TASK_DISPLAY_NAMES.get(task, task),
        "problem_type": problem_type,
        "num_labels": num_labels,
        "label_names": label_names_for_task(task, num_labels, problem_type),
        "notes": _task_notes(task, problem_type),
    }


def _default_head_spec(task_name: str) -> tuple[int, str]:
    if task_name == "chexpert":
        return len(CHEXPERT_LABELS), "multi_label_classification"
    if task_name.startswith("lab_"):
        return len(LAB_CLASS_NAMES), "single_label_classification"
    return 2, "single_label_classification"


def _task_notes(task_name: str, problem_type: str) -> str:
    if task_name == "chexpert":
        return "Multi-label chest X-ray findings; each label is independent (sigmoid threshold 0.5)."
    if task_name.startswith("lab_"):
        return (
            "4-class lab severity (0=normal, 1-3=abnormal). "
            "For binary risk, use abnormal_probability = 1 - P(normal)."
        )
    if problem_type == "single_label_classification":
        return "Binary outcome; positive_probability is P(class=1)."
    return ""


def enrich_task_head_prediction(
    task_name: str,
    adapter_cfg: dict[str, Any],
    task_head_output: dict[str, Any],
) -> dict[str, Any]:
    meta = describe_task(task_name, adapter_cfg)
    enriched = dict(task_head_output)
    enriched["display_name"] = meta["display_name"]
    enriched["label_names"] = meta["label_names"]

    probs = enriched.get("probabilities") or []
    problem_type = str(enriched.get("problem_type", meta["problem_type"]))

    if problem_type == "multi_label_classification":
        labels = meta["label_names"]
        enriched["predicted_findings"] = [
            {"label": labels[i], "probability": probs[i], "predicted": int(probs[i] >= 0.5)}
            for i in range(min(len(labels), len(probs)))
        ]
        return enriched

    if normalize_task_name(task_name).startswith("lab_") and probs:
        pred_idx = int(enriched.get("predicted_label", 0))
        enriched["predicted_class_name"] = meta["label_names"][pred_idx] if pred_idx < len(meta["label_names"]) else f"class_{pred_idx}"
        enriched["abnormal_probability"] = float(1.0 - probs[0])
        enriched["is_abnormal"] = pred_idx >= 1
        return enriched

    if problem_type == "single_label_classification" and len(probs) >= 2:
        enriched["positive_probability"] = float(probs[1])
        enriched["predicted_class_name"] = meta["label_names"][int(enriched.get("predicted_label", 0))]
        return enriched

    pred_idx = int(enriched.get("predicted_label", 0))
    if pred_idx < len(meta["label_names"]):
        enriched["predicted_class_name"] = meta["label_names"][pred_idx]
    return enriched
