#!/usr/bin/env python3
"""Run the PADZE reproducibility validation gates.

This is the compact entry point for reviewers who want to verify the PADZE upgrade
claims without running the broader DNNaic training/reproduction pipeline.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CORE_TESTS = [
    "tests/test_adze_rarefaction.py",
    "tests/test_adze_moments.py",
    "tests/test_adze_io_features.py",
    "tests/test_classical_rolling.py",
    "tests/test_vcf_structure_equivalence.py",
    "tests/test_vcf_allele_coding_invariance.py",
    "tests/test_multi_population_pihat.py",
    "tests/test_cpp_parity.py",
    "tests/test_original_adze_example.py",
    "tests/test_hgdp_competition_outputs.py",
]


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _runnable(path: Path) -> bool:
    if not path.exists() or not os.access(path, os.X_OK):
        return False
    try:
        subprocess.run([str(path)], cwd=ROOT, capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _find_runnable_adze(user_path: str | None) -> Path | None:
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    candidates.extend([
        ROOT / "src" / "external" / "adze" / "src" / "adze-1.0",
        ROOT / "ADZEOriginal" / "src" / "adze-1.0",
        ROOT / "ADZEOriginal" / "ADZE-1.0" / "adze-1.0",
    ])
    for candidate in candidates:
        path = candidate.expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if _runnable(path):
            return path
    return None


def _has_h952_outputs() -> bool:
    base = ROOT / "data" / "competition"
    needed = [
        base / "compare_tools.py",
        base / "adze1" / "out_H952_r",
        base / "adze1" / "out_H952_p",
        base / "padze" / "out_H952_r",
        base / "padze" / "out_H952_p",
    ]
    return all(path.exists() for path in needed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adze-bin", default=None,
                    help="optional original/rebuilt ADZE binary for benchmark smoke")
    ap.add_argument("--benchmark", choices=["auto", "always", "never"], default="auto",
                    help="run the PADZE vs ADZE benchmark smoke (default: auto)")
    ap.add_argument("--h952", choices=["auto", "always", "never"], default="auto",
                    help="run local H952 competition comparison (default: auto)")
    args = ap.parse_args()

    _run([sys.executable, "-m", "pytest", "-q", *CORE_TESTS])

    if args.h952 != "never":
        if _has_h952_outputs():
            _run([sys.executable, "data/competition/compare_tools.py"])
        elif args.h952 == "always":
            raise SystemExit("H952 comparison requested but data/competition outputs are absent")
        else:
            print("SKIP H952 comparison: data/competition outputs are absent")

    if args.benchmark != "never":
        binary = _find_runnable_adze(args.adze_bin)
        if binary is None:
            if args.benchmark == "always":
                raise SystemExit("benchmark requested but no runnable ADZE binary was found")
            print("SKIP benchmark smoke: no runnable ADZE binary found")
        else:
            _run([
                sys.executable,
                "scripts/benchmarks/benchmark_padze_vs_original.py",
                "--adze-bin", str(binary),
                "--populations", "3",
                "--samples-per-pop", "4",
                "--loci", "20",
                "--alleles", "4",
                "--max-g", "6",
                "--repeats", "1",
                "--full",
            ])

    print("PADZE validation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
