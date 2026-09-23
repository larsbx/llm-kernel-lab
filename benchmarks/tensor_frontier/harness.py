#!/usr/bin/env python3
"""Minimal experiment registry front end.

The first scaffold intentionally lists contracts before it runs accelerator
kernels.  Kernel implementations enter only after differential predicates are
fixed.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

MANIFEST = Path(__file__).with_name("manifest.toml")


def load_manifest() -> dict:
    with MANIFEST.open("rb") as handle:
        return tomllib.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print registered experiments as JSON")
    args = parser.parse_args()
    manifest = load_manifest()
    if args.list:
        print(json.dumps(manifest["experiment"], indent=2, sort_keys=True))
        return 0
    parser.error("no action selected; start with --list")


if __name__ == "__main__":
    raise SystemExit(main())
