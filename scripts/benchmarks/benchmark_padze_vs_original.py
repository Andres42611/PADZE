#!/usr/bin/env python3
"""Benchmark PADZE against original ADZE on equivalent inputs.

Original ADZE only accepts STRUCTURE-like input, so this benchmark generates matched
STRUCTURE and VCF datasets from the same genotypes:

* original ADZE C++: STRUCTURE -> R/P/C outputs
* PADZE legacy path: STRUCTURE -> classical result
* PADZE upgrade path: VCF + popmap -> ADZE-compatible R/P/C outputs

The comparison verifies that the Python and C++ summary statistics match within the
precision printed by ADZE, then reports median wall times over repeated runs. The C++ timing
includes process startup and text output. The Python timings are in-process and include
input parsing, classical computation, and ADZE-style output writing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from padze import classical_features, read_structure, read_vcf  # noqa: E402
from padze.cli import _write_adze_outputs  # noqa: E402


def _find_adze_binary(user_path: str | None) -> tuple[Path, str]:
    candidates = []
    requested = Path(user_path).expanduser().resolve() if user_path else None
    if user_path:
        candidates.append(requested)
    candidates.extend([
        ROOT / "ADZEOriginal" / "src" / "adze-1.0",
        ROOT / "ADZEOriginal" / "ADZE-1.0" / "adze-1.0",
        ROOT / "src" / "external" / "adze" / "src" / "adze-1.0",
    ])
    for cand in candidates:
        cand = cand.expanduser().resolve()
        if not cand.exists() or not os.access(cand, os.X_OK):
            continue
        try:
            probe = subprocess.run([str(cand)], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError) as exc:
            last = f"{cand}: {exc}"
            continue
        text = (probe.stdout + probe.stderr)
        if "Allelic Diversity Analyzer" in text or "paramfile" in text:
            note = ("requested" if requested is not None and cand == requested
                    else "auto-selected runnable ADZE binary")
            return cand, note
        last = f"{cand}: did not look like ADZE"
    raise SystemExit(f"no runnable ADZE binary found; last probe: {locals().get('last', 'none')}")


def _generate_dataset(tmp: Path, *, populations: int, samples_per_pop: int,
                      loci: int, alleles: int, seed: int) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(seed)
    pops = [f"P{i + 1}" for i in range(populations)]
    samples = {p: [f"{p}_S{s + 1}" for s in range(samples_per_pop)] for p in pops}
    # genotypes[p, sample, locus, hap] -> allele index 0..alleles-1
    genotypes = {
        p: rng.integers(0, alleles, size=(samples_per_pop, loci, 2), dtype=np.int64)
        for p in pops
    }

    stru = tmp / "data.stru"
    with stru.open("w") as fh:
        fh.write(" ".join(f"L{i + 1}" for i in range(loci)) + "\n")
        for p in pops:
            for si, sample in enumerate(samples[p]):
                for hap in range(2):
                    vals = [str(int(genotypes[p][si, li, hap]) + 1) for li in range(loci)]
                    fh.write(f"{sample} {p} " + " ".join(vals) + "\n")

    vcf = tmp / "data.vcf"
    sample_order = [s for p in pops for s in samples[p]]
    alts = ",".join(f"A{i}" for i in range(1, alleles))
    with vcf.open("w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        fh.write("\t".join(sample_order) + "\n")
        for li in range(loci):
            row = ["chr1", str(li + 1), ".", "A0", alts, ".", "PASS", ".", "GT"]
            for p in pops:
                for si in range(samples_per_pop):
                    a, b = genotypes[p][si, li]
                    row.append(f"{int(a)}/{int(b)}")
            fh.write("\t".join(row) + "\n")

    popmap = tmp / "data.popmap"
    with popmap.open("w") as fh:
        for p in pops:
            for s in samples[p]:
                fh.write(f"{s} {p}\n")
    return stru, vcf, popmap


def _write_paramfile(path: Path, *, max_g: int, data_lines: int, loci: int, full: bool) -> None:
    full_flag = 1 if full else 0
    text = f"""MAX_G {max_g}
DATA_LINES {data_lines}
LOCI {loci}
NON_DATA_ROWS 1
NON_DATA_COLS 2
GROUP_BY_COL 2
DATA_FILE data.stru
R_OUT cpp_r
P_OUT cpp_p
COMB 1
K_RANGE 2
C_OUT cpp_c
MISSING -9
TOLERANCE 1
FULL_R {full_flag}
FULL_P {full_flag}
FULL_C {full_flag}
PRINT_PROGRESS 0
"""
    path.write_text(text)


def _parse_summary(path: Path, label_count: int, kind: str) -> dict[tuple[str, tuple[str, ...], int], tuple[float, float, float]]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        labels = tuple(parts[:label_count])
        g = int(parts[label_count])
        out[(kind, labels, g)] = tuple(float(x) for x in parts[label_count + 2:label_count + 5])
    return out


def _parse_full(path: Path, label_count: int, kind: str, n_loci: int) -> dict[tuple[str, tuple[str, ...], int], tuple[float, ...]]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "NUM_LOCI" in line or not line.strip():
            continue
        parts = line.split()
        if len(parts) < label_count + n_loci + 5:
            continue
        labels = tuple(parts[:label_count])
        g = int(parts[label_count])
        vals = tuple(float(x) for x in parts[label_count + 2:label_count + 2 + n_loci])
        out[(kind, labels, g)] = vals
    return out


def _python_summary(res) -> dict[tuple[str, tuple[str, ...], int], tuple[float, float, float]]:
    out = {}
    for (key, g), vals in res.summary.items():
        kind, suffix = key.split("_", 1)
        if kind in ("alpha", "pi"):
            labels = (res.populations[int(suffix) - 1],)
            out_kind = "r" if kind == "alpha" else "p"
        else:
            labels = tuple(res.populations[int(ch) - 1] for ch in suffix)
            out_kind = f"c_{len(labels)}"
        out[(out_kind, labels, g)] = vals
    return out


def _run_cpp(binary: Path, cwd: Path, *, max_g: int, populations: int,
             samples_per_pop: int, loci: int, full: bool) -> tuple[float, dict, dict]:
    _write_paramfile(cwd / "paramfile.txt", max_g=max_g,
                     data_lines=populations * samples_per_pop * 2,
                     loci=loci, full=full)
    t0 = time.perf_counter()
    proc = subprocess.run([str(binary), "paramfile.txt"], cwd=cwd,
                          capture_output=True, text=True, timeout=120)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"ADZE failed: {proc.stderr[-400:] or proc.stdout[-400:]}")
    summary = {}
    summary.update(_parse_summary(cwd / "cpp_r", 1, "r"))
    summary.update(_parse_summary(cwd / "cpp_p", 1, "p"))
    summary.update(_parse_summary(cwd / "cpp_c_2", 2, "c_2"))
    full_data = {}
    if full:
        full_data.update(_parse_full(cwd / "cpp_r_fulldata", 1, "r", loci))
        full_data.update(_parse_full(cwd / "cpp_p_fulldata", 1, "p", loci))
        full_data.update(_parse_full(cwd / "cpp_c_2_fulldata", 2, "c_2", loci))
    return dt, summary, full_data


def _run_python_structure(stru: Path, tmp: Path, *, max_g: int, full: bool) -> tuple[float, dict, dict]:
    t0 = time.perf_counter()
    loci = read_structure(str(stru), header_rows=1, meta_columns=2, pop_column=1,
                          label_column=0, ploidy=1)
    res = classical_features(loci, max_g=max_g, pihat_sizes=(2,), keep_per_locus=full)
    prefix = tmp / "py_structure"
    _write_adze_outputs(res, str(prefix), pihat_sizes=(2,), full=full)
    dt = time.perf_counter() - t0
    full_data = {}
    if full:
        n_loci = len(loci.count_matrices)
        full_data.update(_parse_full(Path(f"{prefix}_r_fulldata"), 1, "r", n_loci))
        full_data.update(_parse_full(Path(f"{prefix}_p_fulldata"), 1, "p", n_loci))
        full_data.update(_parse_full(Path(f"{prefix}_c_2_fulldata"), 2, "c_2", n_loci))
    return dt, _python_summary(res), full_data


def _run_python_vcf(vcf: Path, popmap: Path, tmp: Path, *, max_g: int, full: bool) -> tuple[float, dict, dict]:
    t0 = time.perf_counter()
    loci = read_vcf(str(vcf), str(popmap))
    res = classical_features(loci, max_g=max_g, pihat_sizes=(2,), keep_per_locus=full)
    prefix = tmp / "py_vcf"
    _write_adze_outputs(res, str(prefix), pihat_sizes=(2,), full=full)
    dt = time.perf_counter() - t0
    full_data = {}
    if full:
        n_loci = len(loci.count_matrices)
        full_data.update(_parse_full(Path(f"{prefix}_r_fulldata"), 1, "r", n_loci))
        full_data.update(_parse_full(Path(f"{prefix}_p_fulldata"), 1, "p", n_loci))
        full_data.update(_parse_full(Path(f"{prefix}_c_2_fulldata"), 2, "c_2", n_loci))
    return dt, _python_summary(res), full_data


def _max_abs_diff(a: dict, b: dict) -> tuple[float, dict]:
    keys_a = set(a)
    keys_b = set(b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    if only_a or only_b:
        raise AssertionError(
            f"summary key mismatch: only_cpp={only_a[:5]} only_py={only_b[:5]}")
    keys = sorted(keys_a)
    if not keys:
        return float("nan"), {"rows": 0}
    diff = max(abs(x - y) for k in keys for x, y in zip(a[k], b[k]))
    return diff, {"rows": len(keys)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adze-bin", default=None, help="path to original/rebuilt ADZE binary")
    ap.add_argument("--populations", type=int, default=3)
    ap.add_argument("--samples-per-pop", type=int, default=20)
    ap.add_argument("--loci", type=int, default=300)
    ap.add_argument("--alleles", type=int, default=6)
    ap.add_argument("--max-g", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--full", action="store_true", help="write FULL_* per-locus files too")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    binary, binary_note = _find_adze_binary(args.adze_bin)
    with tempfile.TemporaryDirectory(prefix="adze_bench_") as td:
        tmp = Path(td)
        stru, vcf, popmap = _generate_dataset(
            tmp,
            populations=args.populations,
            samples_per_pop=args.samples_per_pop,
            loci=args.loci,
            alleles=args.alleles,
            seed=args.seed,
        )
        timings = {"cpp_adze_structure": [], "python_structure": [], "python_vcf": []}
        diffs = {"python_structure_vs_cpp": [], "python_vcf_vs_cpp": []}
        full_diffs = {"python_structure_full_vs_cpp": [], "python_vcf_full_vs_cpp": []}
        rows_s = rows_v = {"rows": 0}
        full_rows_s = full_rows_v = {"rows": 0}
        for r in range(args.repeats):
            run_dir = tmp / f"run_{r}"
            run_dir.mkdir()
            shutil.copy(stru, run_dir / "data.stru")
            cpp_dt, cpp_summary, cpp_full = _run_cpp(
                binary, run_dir, max_g=args.max_g, populations=args.populations,
                samples_per_pop=args.samples_per_pop, loci=args.loci, full=args.full)
            py_s_dt, py_s, py_s_full = _run_python_structure(
                run_dir / "data.stru", run_dir, max_g=args.max_g, full=args.full)
            py_v_dt, py_v, py_v_full = _run_python_vcf(
                vcf, popmap, run_dir, max_g=args.max_g, full=args.full)
            timings["cpp_adze_structure"].append(cpp_dt)
            timings["python_structure"].append(py_s_dt)
            timings["python_vcf"].append(py_v_dt)
            diff_s, rows_s = _max_abs_diff(cpp_summary, py_s)
            diff_v, rows_v = _max_abs_diff(cpp_summary, py_v)
            diffs["python_structure_vs_cpp"].append(diff_s)
            diffs["python_vcf_vs_cpp"].append(diff_v)
            if rows_s["rows"] != rows_v["rows"]:
                raise AssertionError(f"structure/vcf row-count mismatch: {rows_s} vs {rows_v}")
            if args.full:
                full_diff_s, full_rows_s = _max_abs_diff(cpp_full, py_s_full)
                full_diff_v, full_rows_v = _max_abs_diff(cpp_full, py_v_full)
                full_diffs["python_structure_full_vs_cpp"].append(full_diff_s)
                full_diffs["python_vcf_full_vs_cpp"].append(full_diff_v)
                if full_rows_s["rows"] != full_rows_v["rows"]:
                    raise AssertionError(
                        f"structure/vcf FULL row-count mismatch: {full_rows_s} vs {full_rows_v}")

    med = {k: median(v) for k, v in timings.items()}
    speedup_structure = med["cpp_adze_structure"] / med["python_structure"]
    speedup_vcf = med["cpp_adze_structure"] / med["python_vcf"]
    report = {
        "adze_binary": str(binary),
        "adze_binary_note": binary_note,
        "parameters": vars(args),
        "median_seconds": med,
        "speedup_vs_cpp": {
            "python_structure": speedup_structure,
            "python_vcf": speedup_vcf,
        },
        "max_abs_diff": {k: max(v) for k, v in diffs.items()},
        "summary_rows": rows_s["rows"],
        "full_rows": full_rows_s["rows"] if args.full else 0,
        "notes": [
            "C++ timing includes process startup and text output.",
            "Python timings are in-process and include parsing, classical computation, and ADZE-style output writing.",
            "Original ADZE cannot ingest VCF; python_vcf is the upgrade path.",
        ],
    }
    if args.full:
        report["max_abs_full_diff"] = {k: max(v) for k, v in full_diffs.items()}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    if any(value > 1e-4 for value in report["max_abs_diff"].values()):
        return 1
    if args.full and any(value > 1e-4 for value in report["max_abs_full_diff"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
