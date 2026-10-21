#!/usr/bin/env python3
"""Validate algebraization classes and explicit loss profiles."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research" / "tensor_frontier" / "algebraization-contracts.toml"

CLASSES = {"A", "S", "D"}
LOSS_VALUES = {"preserved", "lost", "conditional", "not_applicable"}
TARGET_BY_CLASS = {
    "A": "preserved",
    "S": "changed",
    "D": "not_replaced",
}
REQUIRED_FIELDS = {
    "id",
    "class",
    "source_vocabulary",
    "replacement",
    "scope",
    "target_predicate_status",
    "loss",
}


def validate(data: dict) -> list[str]:
    errors: list[str] = []

    if type(data.get("schema")) is not int or data.get("schema") != 1:
        errors.append("schema must be integer 1")

    dimensions = data.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions or not all(
        isinstance(item, str) and item for item in dimensions
    ):
        errors.append("dimensions must be a nonempty string list")
        dimensions = []
    elif len(set(dimensions)) != len(dimensions):
        errors.append("dimensions must not contain duplicates")

    records = data.get("algebraization")
    if not isinstance(records, list) or not records:
        errors.append("at least one [[algebraization]] record is required")
        return errors

    seen_ids: set[str] = set()
    expected_dimensions = set(dimensions)

    for index, record in enumerate(records):
        prefix = f"algebraization[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix} must be a table")
            continue

        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(missing)}")

        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"{prefix}.id must be a nonempty string")
        elif record_id in seen_ids:
            errors.append(f"duplicate algebraization id: {record_id}")
        else:
            seen_ids.add(record_id)

        klass = record.get("class")
        if klass not in CLASSES:
            errors.append(f"{prefix}.class is invalid: {klass!r}")

        for field in ("source_vocabulary", "replacement", "scope"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.{field} must be a nonempty string")

        target = record.get("target_predicate_status")
        expected_target = TARGET_BY_CLASS.get(klass)
        if expected_target is not None and target != expected_target:
            errors.append(
                f"{prefix}.target_predicate_status must be "
                f"{expected_target!r} for class {klass}"
            )

        loss = record.get("loss")
        if not isinstance(loss, dict):
            errors.append(f"{prefix}.loss must be a table")
            continue

        loss_keys = set(loss)
        if loss_keys != expected_dimensions:
            missing_dims = sorted(expected_dimensions - loss_keys)
            extra_dims = sorted(loss_keys - expected_dimensions)
            if missing_dims:
                errors.append(
                    f"{prefix}.loss missing dimensions: {', '.join(missing_dims)}"
                )
            if extra_dims:
                errors.append(
                    f"{prefix}.loss has unknown dimensions: {', '.join(extra_dims)}"
                )

        for dimension, status in loss.items():
            if status not in LOSS_VALUES:
                errors.append(
                    f"{prefix}.loss.{dimension} has invalid status: {status!r}"
                )

    return errors


def main() -> int:
    with REGISTRY.open("rb") as handle:
        data = tomllib.load(handle)

    errors = validate(data)
    if errors:
        print("algebraization contracts: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"algebraization contracts: PASS ({len(data['algebraization'])} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
