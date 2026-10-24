#!/usr/bin/env python3
"""Build an opt-in benchmark adapter to the exact ILP64 OpenBLAS loaded by NumPy.

Never link Mojo's LP64 arguments directly to ILP64 symbols. This adapter widens
all dimensions, records both binaries, and adds one C call to each BLAS call.
"""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.tensor_frontier.profile_mode_gram import Blas, numpy_openblas, sha


def build():
    backend = Blas(numpy_openblas(), 'numpy-ilp64')
    path = Path(backend.metadata['path'])
    source = ROOT / 'benchmarks/tensor_frontier/numpy_blas_bridge.c'
    output = ROOT / '.build/tensor_frontier/libnumpy_cblas_lp64.so'
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['gcc', '-O3', '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared',
               str(source), str(path), '-Wl,-z,defs', '-Wl,-rpath,' + str(path.parent),
               '-o', str(output)]
    subprocess.run(command, check=True)
    metadata = {'command': command, 'compiler': subprocess.check_output(['gcc', '--version'], text=True),
                'source_sha256': sha(source), 'binary_sha256': sha(output),
                'backend': backend.metadata,
                'purpose': 'benchmark-only ABI adapter; LP64 dimensions widened to ILP64',
                'dynamic_dependencies': subprocess.check_output(['ldd', str(output)], text=True)}
    output.with_suffix('.build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return output


if __name__ == '__main__':
    print(build())
