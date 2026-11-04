"""Schema checks for published campaign artefacts — no invented metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_results_json_has_required_top_level_keys():
    data = json.loads((ROOT / "results.json").read_text())
    assert "environment" in data
    assert "experiments" in data
    env = data["environment"]
    assert env.get("vllm") or env.get("vllm_version")
    assert env.get("gpu_count", 0) >= 1
    experiments = data["experiments"]
    assert isinstance(experiments, (dict, list)) and len(experiments) >= 1


def test_csv_rows_align_with_documented_columns():
    path = ROOT / "benchmark_results.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "CSV is empty"
    required = {"run_id", "model_revision", "concurrency", "tokens_per_second"}
    cols = set(rows[0].keys())
    assert required <= cols, f"missing {required - cols}; have {cols}"


def test_csv_notes_flag_single_run():
    path = ROOT / "benchmark_results.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    notes = " ".join((r.get("notes") or "").lower() for r in rows)
    assert "single-run" in notes or "single run" in notes
