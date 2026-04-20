from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence


def load_analyst_labels(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    return {str(query_id): str(label) for query_id, label in payload.items()}


def _f1(expected_label: str, expected: Sequence[str], predicted: Sequence[str]) -> tuple[float, int]:
    support = sum(label == expected_label for label in expected)
    true_positive = sum(
        gold == expected_label and pred == expected_label
        for gold, pred in zip(expected, predicted, strict=True)
    )
    false_positive = sum(
        gold != expected_label and pred == expected_label
        for gold, pred in zip(expected, predicted, strict=True)
    )
    false_negative = sum(
        gold == expected_label and pred != expected_label
        for gold, pred in zip(expected, predicted, strict=True)
    )
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    if precision + recall == 0:
        return 0.0, support
    return (2.0 * precision * recall / (precision + recall)), support


def _weighted_f1(expected: Sequence[str], predicted: Sequence[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    total = len(expected)
    if total == 0:
        return 0.0

    weighted_sum = 0.0
    for label in labels:
        f1, support = _f1(label, expected, predicted)
        weighted_sum += (f1 * support) / total
    return weighted_sum


def _cohens_kappa(expected: Sequence[str], predicted: Sequence[str]) -> float:
    total = len(expected)
    if total == 0:
        return 0.0

    observed = sum(gold == pred for gold, pred in zip(expected, predicted, strict=True)) / total
    gold_counts = Counter(expected)
    pred_counts = Counter(predicted)
    chance = sum((gold_counts[label] / total) * (pred_counts[label] / total) for label in set(gold_counts) | set(pred_counts))
    if chance == 1.0:
        return 1.0
    return (observed - chance) / (1.0 - chance)


def compute_disposition_metrics(expected_by_query: dict[str, str], predicted_by_query: dict[str, str]) -> dict[str, object]:
    expected_ids = set(expected_by_query)
    predicted_ids = set(predicted_by_query)
    if expected_ids != predicted_ids:
        missing = sorted(expected_ids - predicted_ids)
        extra = sorted(predicted_ids - expected_ids)
        raise ValueError(f"query-id mismatch: missing={missing} extra={extra}")

    ordered_ids = list(expected_by_query)
    expected = [expected_by_query[query_id] for query_id in ordered_ids]
    predicted = [predicted_by_query[query_id] for query_id in ordered_ids]
    labels = sorted(set(expected) | set(predicted))
    confusion = {
        gold: {pred: 0 for pred in labels}
        for gold in labels
    }
    for gold, pred in zip(expected, predicted, strict=True):
        confusion[gold][pred] += 1

    per_label = {}
    for label in labels:
        f1, support = _f1(label, expected, predicted)
        per_label[label] = {"support": support, "f1": f1}

    return {
        "weighted_f1": _weighted_f1(expected, predicted),
        "cohens_kappa": _cohens_kappa(expected, predicted),
        "per_label": per_label,
        "confusion_matrix": confusion,
    }
