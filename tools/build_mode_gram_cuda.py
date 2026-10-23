#!/usr/bin/env python3
"""Offline CUDA-target compilation with source, binary and toolchain receipts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'mojo/tensor_frontier/mode_gram_cuda.mojo'
DEFAULT_OUTPUT = ROOT / '.build/tensor_frontier/libmode_gram_cuda.so'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output=DEFAULT_OUTPUT, mojo=None, target='nvidia:sm_80'):
    compiler = shutil.which(mojo or os.environ.get('MOJO', 'mojo'))
    if not compiler:
        raise RuntimeError('Mojo compiler unavailable')
    if not re.fullmatch(r'nvidia:sm_[0-9]+[a-z]?', target):
        raise ValueError('an explicit NVIDIA architecture target is required')
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, 'build', str(SOURCE), '--emit', 'shared-lib',
               '--target-accelerator', target, '-O3', '-o', str(output)]
    subprocess.run(command, check=True)
    receipt = dict(command=command, target=target,
        compiler_version=subprocess.check_output([compiler, '--version'], text=True).strip(),
        binary_sha256=sha(output),
        sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                        (SOURCE, Path(__file__).resolve())},
        implementation_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        dynamic_dependencies=subprocess.check_output(['ldd',str(output)],text=True),
        validation='offline target compilation only; no GPU execution')
    output.with_suffix('.build.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return output


def checked_build(path):
    path = Path(path).resolve()
    receipt = json.loads(path.with_suffix('.build.json').read_text())
    expected = {str(p.relative_to(ROOT)) for p in (SOURCE, Path(__file__).resolve())}
    if set(receipt.get('sources_sha256', {})) != expected:
        raise ValueError('incomplete build provenance')
    if receipt['binary_sha256'] != sha(path):
        raise ValueError('binary hash mismatch')
    for source, digest in receipt['sources_sha256'].items():
        if digest != sha(ROOT/source):
            raise ValueError('source hash mismatch; rebuild')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--mojo')
    parser.add_argument('--target', default='nvidia:sm_80')
    args = parser.parse_args()
    print(build(args.output, args.mojo, args.target))
