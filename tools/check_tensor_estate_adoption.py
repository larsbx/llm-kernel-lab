#!/usr/bin/env python3
"""Validate the tensor-frontier cross-estate authority routing."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research" / "tensor_frontier" / "estate-adoption.toml"

VALID_GRADES = {"E0", "E1", "E2", "E3", "E4", "E5"}
EVIDENCE_ONLY_ROLES = {
    "domain_owner",
    "reusable_exact_kernel_owner",
    "independent_oracle",
    "semantic_contract_owner",
}
REQUIRED_AUTHORITIES = {
    "discovery",
    "reusable_exact_kernels",
    "independent_oracle",
    "semantic_contracts",
    "claim_owner",
    "proof_authority",
}


def validate(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("schema") != 1:
        errors.append("schema must be 1")

    authority = data.get("authority")
    if not isinstance(authority, dict):
        errors.append("authority table is required")
    else:
        missing = sorted(REQUIRED_AUTHORITIES - set(authority))
        if missing:
            errors.append("authority missing keys: " + ", ".join(missing))
        if authority.get("claim_owner") != "domain_repository":
            errors.append("claim_owner must remain domain_repository")
        if authority.get("proof_authority") != "domain_checker_or_lean":
            errors.append("proof_authority must remain domain_checker_or_lean")

    routes = data.get("route")
    if not isinstance(routes, list) or not routes:
        errors.append("at least one [[route]] is required")
        return errors

    seen_ids: set[str] = set()
    priorities: set[int] = set()
    for index, route in enumerate(routes):
        prefix = f"route[{index}]"
        if not isinstance(route, dict):
            errors.append(f"{prefix} must be a table")
            continue

        route_id = route.get("id")
        if not isinstance(route_id, str) or not route_id:
            errors.append(f"{prefix}.id must be a nonempty string")
        elif route_id in seen_ids:
            errors.append(f"duplicate route id: {route_id}")
        else:
            seen_ids.add(route_id)

        repo = route.get("repository")
        if not isinstance(repo, str) or "/" not in repo:
            errors.append(f"{prefix}.repository must be owner/name")

        role = route.get("role")
        if role not in EVIDENCE_ONLY_ROLES:
            errors.append(f"{prefix}.role is unknown: {role!r}")

        grade = route.get("allowed_max_grade")
        if grade not in VALID_GRADES:
            errors.append(f"{prefix}.allowed_max_grade is invalid: {grade!r}")

        if route.get("may_accept_domain_claim") is not False:
            errors.append(
                f"{prefix}.may_accept_domain_claim must be false; "
                "domain acceptance is never delegated by this manifest"
            )

        priority = route.get("priority")
        if type(priority) is not int or priority < 1:
            errors.append(f"{prefix}.priority must be a positive integer")
        elif priority in priorities:
            errors.append(f"duplicate priority: {priority}")
        else:
            priorities.add(priority)

        for key in ("imports", "exports"):
            values = route.get(key)
            if not isinstance(values, list) or not values or not all(
                isinstance(v, str) and v for v in values
            ):
                errors.append(f"{prefix}.{key} must be a nonempty string list")

    return errors


def main() -> int:
    with MANIFEST.open("rb") as handle:
        data = tomllib.load(handle)
    errors = validate(data)
    if errors:
        print("tensor estate adoption: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"tensor estate adoption: PASS ({len(data['route'])} routes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
