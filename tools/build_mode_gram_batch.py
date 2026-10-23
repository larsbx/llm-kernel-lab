#!/usr/bin/env python3
"""Build the guarded, allocation-free native batch experiment."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_mode_gram import build as build_native

SOURCE = ROOT / 'mojo/tensor_frontier/mode_gram_batch.mojo'
DEFAULT_OUTPUT = ROOT / '.build/tensor_frontier/libmode_gram_batch.so'


def build(output=DEFAULT_OUTPUT, mojo=None, blas=None):
    return build_native(output, mojo, blas, source=SOURCE,
                        dependencies=(ROOT / 'mojo/tensor_frontier/mode_gram.mojo',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--mojo')
    parser.add_argument('--blas')
    args = parser.parse_args()
    print(build(args.output, args.mojo, args.blas))
