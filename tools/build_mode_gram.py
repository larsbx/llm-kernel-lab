#!/usr/bin/env python3
"""Build the Linux-64 Mojo/LP64 CBLAS experiment and record its provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "mojo/tensor_frontier/mode_gram.mojo"
DEFAULT_OUTPUT = ROOT / ".build/tensor_frontier/libmode_gram.so"


def build(output=DEFAULT_OUTPUT, mojo=None, blas=None, *, source=SOURCE, dependencies=()):
    mojo = mojo or os.environ.get("MOJO", "mojo")
    compiler = shutil.which(mojo)
    if compiler is None:
        raise RuntimeError(f"Mojo compiler unavailable: {mojo}")
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Must expose cblas_dgemm with LP64 int dimensions, NOT ILP64 *_64 symbols.
    blas = blas or os.environ.get("MODE_GRAM_BLAS", "libblas.so.3")
    link_arg = str(Path(blas).resolve()) if "/" in blas else "-l:" + blas
    command = [compiler, "build", str(source), "--emit", "shared-lib", "-O3",
               "-Xlinker", link_arg, "-o", str(output)]
    subprocess.run(command, check=True)
    metadata = {
        "command": command,
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True).strip(),
        "blas_link_argument": link_arg,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sources_sha256": {str(Path(p).resolve().relative_to(ROOT)):
            hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (source, *dependencies)},
        "binary_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "implementation_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "dynamic_dependencies": subprocess.check_output(["ldd", str(output)], text=True),
    }
    output.with_suffix(".build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mojo")
    parser.add_argument("--blas", help="LP64 CBLAS soname or absolute library path")
    args = parser.parse_args()
    print(build(args.output, args.mojo, args.blas))
