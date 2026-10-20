#!/usr/bin/env python3
"""Reproducible CPU comparison; E0 timings and E1 cross-language observations.

Run as `python -m benchmarks.tensor_frontier.benchmark_mode_gram` from the repo.
Both public APIs receive the same resident C-contiguous FP64 tensor. Timing
includes validation, output allocation, FFI (Mojo), packing and GEMM; excludes
compilation, imports, input generation and exact-reference computation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np

from tensor_lab.mojo_mode_gram import DEFAULT_LIBRARY, MojoModeGram
from tensor_lab.numpy_baseline import mode_gram_allclose, mode_gram_numpy
from tensor_lab.tensors import mode_gram

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "benchmarks/tensor_frontier/mode_gram_cases.json"


def cases(include_medium=True):
    result = json.loads(CORPUS.read_text())["cases"]
    if include_medium:
        for shape in [(16, 24, 8), (24, 24, 24)]:
            # Stable dyadic values: no PRNG or seed dependence; exact in FP64.
            data = ((np.arange(np.prod(shape), dtype=np.int64) * 17) % 65 - 32) / 8
            result.append({"id": "dyadic_" + "x".join(map(str, shape)),
                           "tensor": data.reshape(shape).tolist()})
    return result


def residual(reference, actual):
    expected = np.asarray(reference, dtype=np.float64)
    if not mode_gram_allclose(reference, actual):
        raise ValueError("FP64 differential tolerance failed; refusing benchmark evidence")
    error = np.abs(actual - expected)
    return float(np.max(error)), float(np.max(error / (1e-12 + 1e-12 * np.abs(expected))))


def measure(function, tensor, mode, reference, warmup, iterations):
    for _ in range(warmup):
        residual(reference, function(tensor, mode))
    samples = []
    max_residual = 0.0
    max_ratio = 0.0
    for _ in range(iterations):
        start = time.perf_counter_ns()
        actual = function(tensor, mode)
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed)
        # Check every observed result, outside the timed region.
        error, ratio = residual(reference, actual)
        max_residual = max(max_residual, error)
        max_ratio = max(max_ratio, ratio)
    return samples, max_residual, max_ratio


def benchmark(library, warmup, iterations, rounds, include_medium=True):
    if warmup < 0 or iterations <= 0 or rounds <= 0:
        raise ValueError("warmup must be nonnegative; iterations and rounds must be positive")
    library = Path(library).resolve()
    build = json.loads(library.with_suffix(".build.json").read_text())
    if hashlib.sha256(library.read_bytes()).hexdigest() != build["binary_sha256"]:
        raise ValueError("library no longer matches recorded build")
    source = ROOT / "mojo/tensor_frontier/mode_gram.mojo"
    if hashlib.sha256(source.read_bytes()).hexdigest() != build["source_sha256"]:
        raise ValueError("source differs from recorded build; rebuild before benchmarking")
    kernel = MojoModeGram(library)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    cpuinfo = Path("/proc/cpuinfo").read_text()
    cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                if line.startswith("model name")), platform.machine())
    device = {"kind": "cpu", "model": cpu, "platform": platform.platform(),
              "affinity": sorted(os.sched_getaffinity(0))}
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        device["cgroup_cpu_max"] = quota.read_text().strip()
    records = []
    for case in cases(include_medium):
        tensor = np.asarray(case["tensor"], dtype=np.float64)
        for mode in range(3):
            reference = mode_gram(case["tensor"], mode)
            functions = [("mojo_cblas", kernel), ("numpy", mode_gram_numpy)]
            for repeat in range(rounds):
                # Alternate order to reduce systematic first/second bias.
                for name, function in functions[::1 if repeat % 2 == 0 else -1]:
                    samples, error, ratio = measure(function, tensor, mode, reference, warmup, iterations)
                    median = statistics.median(samples) / 1e9
                    # Common logical traffic floor, NOT hardware-counter traffic:
                    # one input read + one output write; caches/BLAS packing and
                    # validation reads are excluded from this effective metric.
                    effective_bytes = tensor.nbytes + tensor.shape[mode] ** 2 * 8
                    records.append({
                        "case": case["id"], "shape": list(tensor.shape), "mode": mode,
                        "dtype": "float64", "implementation": name,
                        "implementation_revision": revision, "device": device,
                        "round": repeat, "warmup_count": warmup, "measured_iterations": iterations,
                        "wall_time_seconds": sum(samples) / 1e9,
                        "median_wall_time_seconds": median, "samples_ns": samples,
                        "effective_bytes_moved_per_iteration": effective_bytes,
                        "effective_bytes_moved": effective_bytes * iterations,
                        "effective_bytes_per_second": effective_bytes / median,
                        "max_numerical_residual": error, "max_tolerance_ratio": ratio,
                        "performance_evidence": "E0", "numerical_evidence": "E1",
                    })
    return {
        "schema": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_revision": revision,
        "working_tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "build": build, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "numpy_configuration": np.show_config(mode="dicts"),
        "thread_environment": {key: os.environ.get(key) for key in
                               ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "reference": "tensor_lab.tensors.mode_gram (exact rational)",
        "comparison": "abs(observed-reference) <= 1e-12 + 1e-12*abs(reference)",
        "timing_scope": "resident-array public API: validation, allocation, packing, FFI and GEMM",
        "effective_bytes_definition": "8*(tensor element count + output element count); logical floor, not measured DRAM traffic",
        "authority": "E0 performance; E1 replicated numerical observations; no theorem authority",
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--corpus-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new evidence path")
    report = benchmark(args.library, args.warmup, args.iterations, args.rounds, not args.corpus_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(report['records'])} measured records to {args.output}")
