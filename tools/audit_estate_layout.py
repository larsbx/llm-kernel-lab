#!/usr/bin/env python3
"""Validate the estate repository architecture manifest.

This audit is intentionally domain-agnostic. Domain theorem status and certificate
acceptance remain consumer responsibilities.
"""

from __future__ import annotations

import glob
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "estate.toml"

ALLOWED_PLANE_AUTHORITIES = {
    "governance",
    "canonical_executable",
    "claim_state",
    "non_authoritative_reference",
    "non_authoritative_oracle",
    "non_authoritative_experiment",
    "contract",
    "evidence",
    "pinned_external",
    "repository_tooling",
    "exposition",
    "publication",
    "example",
}
ALLOWED_LANGUAGE_AUTHORITIES = {"canonical", "supporting"}


def fail(message: str) -> None:
    raise AssertionError(message)


def load(path: Path = MANIFEST) -> dict:
    if not path.is_file():
        fail(f"missing estate manifest: {path.relative_to(ROOT)}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def matches(pattern: str) -> list[Path]:
    return [Path(p) for p in glob.glob(str(ROOT / pattern), recursive=True)]


def validate(data: dict) -> None:
    if data.get("version") != 1:
        fail("estate.toml version must be 1")
    if data.get("template") != "estate-repository-v1":
        fail("estate.toml template must be estate-repository-v1")

    repository = data.get("repository", {})
    repository_id = repository.get("id", "")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repository_id):
        fail("repository.id must be OWNER/REPOSITORY")
    if repository.get("layout_status") not in {"transitional", "canonical"}:
        fail("repository.layout_status must be transitional or canonical")

    principles = data.get("principles", {})
    if principles.get("ordering") != ["authority", "domain", "language"]:
        fail("principles.ordering must be authority, domain, language")
    if principles.get("cross_language_disagreement") != "fail_closed":
        fail("cross-language disagreement must fail closed")
    if principles.get("empty_silos") != "forbidden":
        fail("empty silos must be forbidden")

    planes = data.get("plane", [])
    if not planes:
        fail("at least one authority plane is required")

    ids: set[str] = set()
    targets: set[str] = set()
    for plane in planes:
        plane_id = plane.get("id", "")
        target = plane.get("target", "")
        authority = plane.get("authority", "")
        if not plane_id:
            fail("plane.id is required")
        if plane_id in ids:
            fail(f"duplicate plane id: {plane_id}")
        ids.add(plane_id)
        if not target:
            fail(f"plane {plane_id}: target is required")
        if target in targets:
            fail(f"duplicate plane target: {target}")
        targets.add(target)
        if authority not in ALLOWED_PLANE_AUTHORITIES:
            fail(f"plane {plane_id}: unknown authority {authority!r}")

        current = plane.get("current", [])
        current_globs = plane.get("current_globs", [])
        if plane.get("required", False) and not (current or current_globs):
            fail(f"plane {plane_id}: required plane needs a current mapping")

        for rel in current:
            if not (ROOT / rel).exists():
                fail(f"plane {plane_id}: missing current path {rel}")
        for pattern in current_globs:
            if not matches(pattern):
                fail(f"plane {plane_id}: current_globs pattern matches nothing: {pattern}")

    if "kernel" not in ids:
        fail("kernel plane is required for this template")
    if "policy" not in ids:
        fail("policy plane is required for this template")

    languages = data.get("language", [])
    names: set[str] = set()
    canonical = []
    for language in languages:
        name = language.get("name", "")
        authority = language.get("authority", "")
        if not name:
            fail("language.name is required")
        if name in names:
            fail(f"duplicate language: {name}")
        names.add(name)
        if authority not in ALLOWED_LANGUAGE_AUTHORITIES:
            fail(f"language {name}: invalid authority {authority!r}")
        if authority == "canonical":
            canonical.append(language)
        if language.get("acceptance_authority") and authority != "canonical":
            fail(f"language {name}: supporting language cannot have acceptance authority")

    if len(canonical) != 1:
        fail("exactly one canonical language is required")
    if "kernel" not in canonical[0].get("roles", []):
        fail("canonical language must own the kernel role")

    for required in ("ARCHITECTURE.md", "docs/architecture/estate-repository-template-v1.md"):
        if not (ROOT / required).is_file():
            fail(f"missing architecture entrypoint: {required}")

    workspace = ROOT / "pixi.toml"
    if workspace.is_file():
        wdata = tomllib.loads(workspace.read_text(encoding="utf-8"))
        expected_name = repository_id.split("/", 1)[1]
        if wdata.get("workspace", {}).get("name") != expected_name:
            fail(
                "pixi workspace identity disagrees with estate.toml: "
                f"expected {expected_name!r}"
            )

    polyglot = ROOT / "polyglot.manifest.toml"
    if polyglot.is_file():
        pdata = tomllib.loads(polyglot.read_text(encoding="utf-8"))
        if pdata.get("repository") != repository_id:
            fail("polyglot.manifest.toml repository disagrees with estate.toml")
        estate_link = pdata.get("estate", {}).get("manifest")
        if estate_link != "estate.toml":
            fail("polyglot.manifest.toml must link to estate.toml")

        if (ROOT / "oracles/julia").exists():
            supporting = pdata.get("authority", {}).get("supporting_languages", [])
            if "Julia" not in supporting:
                fail("Julia oracle lane exists but polyglot supporting_languages omits Julia")


def main() -> int:
    try:
        validate(load())
    except (AssertionError, tomllib.TOMLDecodeError) as exc:
        print(f"estate-layout audit failed: {exc}", file=sys.stderr)
        return 1
    print("estate-layout audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
