#!/usr/bin/env python3
"""
Build patient-level CLMBR timelines from user dataset parquet and save to .pt.

Expected raw rows include at least:
- subject_id
- time
- code
Optional:
- numeric_value
- text_value
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLMBR patient timeline cache (.pt)")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset",
        help="Path to user dataset root or parquet file. If dir, expects data/data.parquet.",
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=20000,
        help="Max number of patients to process.",
    )
    parser.add_argument(
        "--output_pt_path",
        type=str,
        default="dataset_pt/patients.pt",
        help="Where to write processed .pt patient cache.",
    )
    return parser.parse_args()


def resolve_data_parquet_path(dataset_path: str) -> Path:
    p = Path(dataset_path).resolve()
    if p.is_dir():
        candidate = p / "data" / "data.parquet"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Cannot find data parquet at {candidate}. "
                "Expected dataset/data/data.parquet style layout."
            )
        return candidate
    if not p.exists():
        raise FileNotFoundError(f"dataset_path not found: {p}")
    return p


def build_measurement_from_row(row: Dict) -> Dict:
    measurement = {"code": row["code"]}
    numeric_value = row.get("numeric_value")
    text_value = row.get("text_value")
    if numeric_value is not None and not (
        isinstance(numeric_value, float) and math.isnan(numeric_value)
    ):
        measurement["numeric_value"] = float(numeric_value)
    if text_value is not None and text_value != "":
        measurement["text_value"] = str(text_value)
    return measurement


def ensure_birth_event(events: List[Dict]) -> List[Dict]:
    birth_code = "SNOMED/184099003"
    has_birth = any(
        m.get("code") == birth_code for e in events for m in e.get("measurements", [])
    )
    if not has_birth and len(events) > 0:
        events = [{"time": events[0]["time"], "measurements": [{"code": birth_code}]}] + events
    return events


def to_patient(patient_id: int, events: List[Dict]) -> Dict:
    events = ensure_birth_event(events)
    return {"patient_id": int(patient_id), "static_measurements": [], "events": events}


def build_patients(dataset_path: Path, max_patients: int) -> List[Dict]:
    stream = load_dataset("parquet", data_files=str(dataset_path), split="train", streaming=True)

    patients: List[Dict] = []
    current_subject_id = None
    current_events: List[Dict] = []
    current_event = None

    progress = tqdm(stream, desc="Building patient timelines", unit="rows")
    for row in progress:
        sid = int(row["subject_id"])
        if current_subject_id is None:
            current_subject_id = sid

        if sid != current_subject_id:
            if current_events:
                patients.append({"patient": to_patient(current_subject_id, current_events)})
                if max_patients is not None and len(patients) >= max_patients:
                    break
            current_subject_id = sid
            current_events = []
            current_event = None

        event_time = row["time"]
        if current_event is None or current_event["time"] != event_time:
            current_event = {"time": event_time, "measurements": []}
            current_events.append(current_event)
        current_event["measurements"].append(build_measurement_from_row(row))

        if len(patients) > 0 and len(patients) % 500 == 0:
            progress.set_postfix({"patients": len(patients)})

    if (
        current_subject_id is not None
        and current_events
        and (max_patients is None or len(patients) < max_patients)
    ):
        patients.append({"patient": to_patient(current_subject_id, current_events)})

    return patients


def main() -> None:
    args = parse_args()
    parquet_path = resolve_data_parquet_path(args.dataset_path)
    patients = build_patients(parquet_path, args.max_patients)
    if len(patients) == 0:
        raise ValueError("No patients were built. Check dataset path and schema.")

    out_path = Path(args.output_pt_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "patients": patients,
        "meta": {
            "dataset_path": str(parquet_path),
            "max_patients": args.max_patients,
            "num_patients": len(patients),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    torch.save(payload, out_path)
    print(f"Saved patient timeline cache: {out_path}")
    print(f"Saved patients: {len(patients):,}")

    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload["meta"], f, indent=2)
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

