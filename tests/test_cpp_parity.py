"""Parity test: PADZE vs the vendored C++ ADZE on identical input.

This is the strongest correctness check available: it runs the actual C++ ADZE
reference tree and PADZE on the *same* STRUCTURE dataset and asserts the
allelic richness (alpha), private allelic richness (pi), and combination-private richness
(pihat) -- with their mean / variance / SE across loci -- agree to a tight tolerance.

It is self-contained and degrades gracefully:
  * If ``ADZE_BIN`` points to a runnable ADZE binary, it is used.
  * Else, if a built ADZE binary is found in ``ADZEOriginal/src`` or the older
    ``src/external/adze/src`` location, it is used.
  * Else, if a C++ compiler and GSL (``gsl-config``) are available, the binary is built
    on the fly into the source tree.
  * Else the test SKIPS with instructions (so it never breaks an environment without C++).

Build manually with::

    cd ADZEOriginal/src
    clang++ -O2 -std=c++11 $(gsl-config --cflags) ADZE_*.cpp $(gsl-config --libs) -lm \\
        -o adze_py_parity

Run standalone::

    /Users/ard/Desktop/genenv/bin/python tests/test_cpp_parity.py
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
ADZE_SRC_CANDIDATES = (
    os.path.join(REPO, "ADZEOriginal", "src"),
    os.path.join(REPO, "src", "external", "adze", "src"),
)
sys.path.insert(0, os.path.join(REPO, "src"))

from padze import classical_features, read_structure  # noqa: E402


class SkipParity(Exception):
    """Raised to skip the parity test when the C++ toolchain is unavailable."""


def _skip_parity(reason):
    try:
        import pytest  # type: ignore
    except Exception:
        print(f"SKIP test_cpp_parity: {reason}")
        return
    pytest.skip(str(reason))


def _adze_src_dir():
    for candidate in ADZE_SRC_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    looked = ", ".join(os.path.relpath(p, REPO) for p in ADZE_SRC_CANDIDATES)
    raise SkipParity(f"C++ ADZE source tree not found; looked in {looked}")


def _find_or_build_binary():
    src_dir = _adze_src_dir()
    candidates = []
    if os.environ.get("ADZE_BIN"):
        candidates.append(os.path.abspath(os.environ["ADZE_BIN"]))
    candidates.extend(os.path.join(src_dir, name)
                      for name in ("adze_py_parity", "adze_arm64", "adze-1.0"))
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            # Verify it actually runs on this arch (the shipped i386 binary may not).
            # Run in a throwaway cwd: ADZE writes a default paramfile.txt when given no args.
            try:
                with tempfile.TemporaryDirectory() as probe:
                    subprocess.run([cand], cwd=probe, capture_output=True, timeout=10)
                return cand
            except Exception:
                continue
    # Try to build.
    if shutil.which("gsl-config") is None:
        raise SkipParity("gsl-config not found; cannot build C++ ADZE")
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise SkipParity("no C++ compiler found")
    cflags = subprocess.check_output(["gsl-config", "--cflags"], text=True).split()
    libs = subprocess.check_output(["gsl-config", "--libs"], text=True).split()
    srcs = [f for f in os.listdir(src_dir) if f.endswith(".cpp")]
    if not srcs:
        raise SkipParity(f"no C++ source files found in {src_dir}")
    out = os.path.join(src_dir, "adze_py_parity")
    cmd = [compiler, "-O2", "-std=c++11", *cflags, *srcs, *libs, "-lm", "-o", out]
    proc = subprocess.run(cmd, cwd=src_dir, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0 or not os.path.isfile(out):
        raise SkipParity(f"C++ build failed: {proc.stderr[-400:]}")
    return out


def _make_structure(path, *, seed=123, n_pops=3, n_loci=8, copies=4, max_alleles=3,
                    miss_frac=0.0):
    rng = np.random.default_rng(seed)
    pop_names = [f"pop{chr(ord('A') + i)}" for i in range(n_pops)]
    geno = {}
    for pi, p in enumerate(pop_names):
        arr = rng.integers(1, max_alleles + 1, size=(copies, n_loci))
        if miss_frac:
            arr = np.where(rng.random((copies, n_loci)) < miss_frac, -9, arr)
            for li in range(n_loci):
                if int((arr[:, li] != -9).sum()) < 2:
                    fill = np.flatnonzero(arr[:, li] == -9)[:2]
                    for r in fill:
                        arr[r, li] = int(rng.integers(1, max_alleles + 1))
        geno[p] = arr
    lines = [" ".join(f"L{l + 1}" for l in range(n_loci))]   # locus-name header row
    for pi, p in enumerate(pop_names):
        for h in range(copies):
            row = [f"ind{pi}_{h}", p]
            row.extend(str(int(geno[p][h, li])) for li in range(n_loci))
            lines.append(" ".join(row))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return pop_names, n_loci, n_pops * copies


def _write_paramfile(path, *, data_file, max_g, data_lines, n_loci):
    text = f"""MAX_G {max_g}
DATA_LINES {data_lines}
LOCI {n_loci}
NON_DATA_ROWS 1
NON_DATA_COLS 2
GROUP_BY_COL 2
DATA_FILE {data_file}
R_OUT rich.out
P_OUT priv.out
COMB 1
K_RANGE 2
C_OUT comb.out
MISSING -9
TOLERANCE 1
FULL_R 1
FULL_P 1
FULL_C 1
PRINT_PROGRESS 0
"""
    with open(path, "w") as f:
        f.write(text)


def _parse_cpp(path):
    """Parse 'name... g numLoci avg var se' lines into {(name,g): (avg,var,se)}."""
    out = {}
    if not os.path.isfile(path):
        return out
    for line in open(path):
        t = line.split()
        if len(t) < 6:
            continue
        try:
            g = int(t[-5]); avg = float(t[-3]); var = float(t[-2]); se = float(t[-1])
        except ValueError:
            continue
        name = "_".join(t[:-5])
        out[(name, g)] = (avg, var, se)
    return out


def _parse_cpp_full(path, n_loci):
    """Parse a FULL_R/P/C '_fulldata' file -> {(name,g): [per-locus values]}.

    Line format: 'name... g numLoci v0 v1 ... v_{L-1} mean var se'. The header line
    (contains 'NUM_LOCI') is skipped.
    """
    out = {}
    if not os.path.isfile(path):
        return out
    for line in open(path):
        if "NUM_LOCI" in line or not line.strip():
            continue
        t = line.split()
        if len(t) < n_loci + 5:
            continue
        try:
            vals = [float(x) for x in t[-(n_loci + 3):-3]]
            numloci = int(t[-(n_loci + 4)])
            g = int(t[-(n_loci + 5)])
        except ValueError:
            continue
        name = "_".join(t[:-(n_loci + 5)])
        out[(name, g)] = vals
    return out


def run_parity(max_g=4, tol=1e-4, seed=123, miss_frac=0.0):
    binary = _find_or_build_binary()
    tmp = tempfile.mkdtemp(prefix="adze_parity_")
    data_path = os.path.join(tmp, "data.stru")
    pop_names, n_loci, data_lines = _make_structure(data_path, seed=seed,
                                                    miss_frac=miss_frac)
    _write_paramfile(os.path.join(tmp, "paramfile.txt"),
                     data_file="data.stru", max_g=max_g,
                     data_lines=data_lines, n_loci=n_loci)
    proc = subprocess.run([binary, "paramfile.txt"], cwd=tmp,
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise AssertionError(f"C++ ADZE failed: {proc.stderr[-400:]}")

    rich = _parse_cpp(os.path.join(tmp, "rich.out"))
    priv = _parse_cpp(os.path.join(tmp, "priv.out"))
    comb = _parse_cpp(os.path.join(tmp, "comb.out_2"))
    assert rich and priv and comb, "C++ produced no output to compare"

    loci = read_structure(data_path, ploidy=1, one_row_per_individual=False,
                          header_rows=1)
    assert loci.populations == sorted(pop_names)
    res = classical_features(loci, max_g=max_g, pihat_sizes=(2,))
    pmap = {p: i + 1 for i, p in enumerate(sorted(pop_names))}

    def stat_at(key, g):
        return res.summary[(key, g)]

    max_abs = 0.0
    n_cmp = 0
    expected_summary_keys = set()
    for cpp, keyfn, label in (
        (rich, lambda n: f"alpha_{pmap[n]}", "alpha"),
        (priv, lambda n: f"pi_{pmap[n]}", "pi"),
        (comb, lambda n: "pihat_" + "".join(str(pmap[p]) for p in n.split("_")), "pihat"),
    ):
        for (name, g), (avg, var, se) in cpp.items():
            key = keyfn(name)
            expected_summary_keys.add((key, g))
            pm, pv, ps = stat_at(key, g)
            for a, b in ((pm, avg), (pv, var), (ps, se)):
                max_abs = max(max_abs, abs(a - b))
                n_cmp += 1
                assert abs(a - b) < tol, (
                    f"{label} {name} g={g}: py={a} cpp={b} (|d|={abs(a-b):.2e})")
    assert set(res.summary) == expected_summary_keys

    # ---- per-locus (FULL_R/P/C) equivalence ----
    n_loci = len(loci.count_matrices)
    rich_f = _parse_cpp_full(os.path.join(tmp, "rich.out_fulldata"), n_loci)
    priv_f = _parse_cpp_full(os.path.join(tmp, "priv.out_fulldata"), n_loci)
    comb_f = _parse_cpp_full(os.path.join(tmp, "comb.out_2_fulldata"), n_loci)
    n_full = 0
    expected_full_keys = set()
    for cpp_full, keyfn in (
        (rich_f, lambda n: f"alpha_{pmap[n]}"),
        (priv_f, lambda n: f"pi_{pmap[n]}"),
        (comb_f, lambda n: "pihat_" + "".join(str(pmap[p]) for p in n.split("_"))),
    ):
        for (name, g), vals in cpp_full.items():
            key = keyfn(name)
            expected_full_keys.add((key, g))
            py_vals = res.per_locus[(key, g)]
            assert len(vals) == len(py_vals), (key, g, len(vals), len(py_vals))
            for a, b in zip(py_vals, vals):
                max_abs = max(max_abs, abs(a - b))
                n_full += 1
                assert abs(a - b) < tol, (
                    f"FULL {key} g={g}: py={a} cpp={b} (|d|={abs(a-b):.2e})")
    assert set(res.per_locus) == expected_full_keys
    return max_abs, n_cmp + n_full, len(rich) + len(priv) + len(comb)


def test_cpp_parity():
    try:
        max_abs, n_cmp, n_rows = run_parity()
    except SkipParity as e:
        _skip_parity(e)
        return
    print(f"PASS test_cpp_parity: {n_cmp} mean/var/se comparisons over {n_rows} "
          f"C++ rows; max |py - cpp| = {max_abs:.3e}")


def test_cpp_parity_second_seed():
    try:
        max_abs, n_cmp, _ = run_parity(seed=2024, max_g=4)
    except SkipParity as e:
        _skip_parity(e)
        return
    print(f"PASS test_cpp_parity_second_seed: {n_cmp} comparisons; "
          f"max |py - cpp| = {max_abs:.3e}")


def test_cpp_parity_missing_data():
    try:
        max_abs, n_cmp, _ = run_parity(seed=77, max_g=4, miss_frac=0.25)
    except SkipParity as e:
        _skip_parity(e)
        return
    print(f"PASS test_cpp_parity_missing_data: {n_cmp} comparisons; "
          f"max |py - cpp| = {max_abs:.3e}")


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)}/{len(fns)} C++ parity tests completed")


if __name__ == "__main__":
    _main()
