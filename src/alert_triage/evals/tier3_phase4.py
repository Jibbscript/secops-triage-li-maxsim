from __future__ import annotations

import argparse
import json
from pathlib import Path

from alert_triage.triage import compute_disposition_metrics, load_analyst_labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the phase-4 tier-3 disposition harness.")
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--tier2-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def run_phase4_disposition(fixture_dir: Path, tier2_json: Path, out_json: Path) -> dict[str, object]:
    analyst_labels = load_analyst_labels(fixture_dir / "analyst_labels.json")
    tier2_report = json.loads(tier2_json.read_text())
    samples = tier2_report["samples"]
    query_ids = [str(sample["query_id"]) for sample in samples]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate query_id values found in tier2 report")
    predicted = {
        str(sample["query_id"]): str(sample["disposition"])
        for sample in samples
    }
    metrics = compute_disposition_metrics(analyst_labels, predicted)
    report = {
        "experiment_id": "tier3-phase4",
        "retriever": "binary-then-fp16-rerank",
        "labels_dataset": "tests/fixtures/phase1_tier1/analyst_labels.json",
        "sample_count": len(analyst_labels),
        "query_count": len(analyst_labels),
        "metrics": {
            "weighted_f1": metrics["weighted_f1"],
            "cohens_kappa": metrics["cohens_kappa"],
        },
        "per_label": metrics["per_label"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    args = _parse_args()
    run_phase4_disposition(args.fixture_dir, args.tier2_json, args.out_json)


if __name__ == "__main__":
    main()
