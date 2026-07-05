"""Large randomized parity harness: PADZE classical mode vs the C++ ADZE binary.

Generates thousands of randomized valid inputs -- varying the number of populations (2-5),
the number of loci, the number of alleles, per-population sample sizes, the missing-data
fraction, the maximum rarefaction depth, and the genotypes -- and for each one runs both the
vendored C++ ADZE and PADZE's classical mode, then compares every emitted statistic:
the across-loci mean / variance / standard error of alpha, pi, and pihat, and every per-locus
value (the FULL_R / FULL_P / FULL_C output). It also checks that the two implementations emit
exactly the same set of (statistic, depth) rows (a structural check); any difference is a
failure.

The C++ prints six significant figures, so the achievable agreement is bounded by that
rounding; the pass threshold is 1e-5 in absolute value (statistics here are < 10, so the
six-figure rounding error is <= ~5e-6). A genuine formula mismatch produces differences that
are orders of magnitude larger, so it is caught.

Run (on the Linux box with the built binary)::

    PYTHONPATH=src python3 tests/test_cpp_parity_10k.py --n 10000 --workers 8

Environment variables ``ADZE_BIN`` (path to the built C++ binary) and ``ADZE_WORK`` (a fast
scratch dir, default /dev/shm/adze10k) override the defaults.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

try:
    import pytest
except Exception:  # pragma: no cover - direct harness execution does not require pytest
    pytest = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "src"))

from padze import classical_features, read_structure  # noqa: E402

BIN = os.path.abspath(os.environ.get(
    "ADZE_BIN", os.path.join(REPO, "src", "external", "adze", "src", "adze_bin")))
WORK = os.path.abspath(os.environ.get("ADZE_WORK", "/dev/shm/adze10k"))
TOL = 1e-5


# ---- input generation --------------------------------------------------------------------

def make_case(seed):
    """Return (structure_text, paramfile_text, meta) for a randomized valid dataset."""
    rng = np.random.default_rng(seed)
    P = int(rng.integers(2, 6))                 # 2..5 populations
    L = int(rng.integers(3, 25))                # 3..24 loci
    max_alleles = int(rng.integers(2, 6))       # 2..5 distinct allele codes
    miss_frac = float(rng.uniform(0.0, 0.35))   # 0..35% missing
    copies = [int(rng.integers(3, 11)) for _ in range(P)]   # gene copies per population

    # genotype matrices per population: (copies_p, L), integer allele codes or -9 (missing).
    geno = []
    for p in range(P):
        g = rng.integers(1, max_alleles + 1, size=(copies[p], L))
        miss = rng.random((copies[p], L)) < miss_frac
        g = np.where(miss, -9, g)
        # keep >= 2 non-missing gene copies per (population, locus): avoids all-missing loci
        # (undefined in the C++) and keeps depth 2 always defined.
        for l in range(L):
            col = g[:, l]
            if int(np.sum(col != -9)) < 2:
                take = rng.choice(copies[p], size=min(2, copies[p]), replace=False)
                for r in take:
                    g[r, l] = int(rng.integers(1, max_alleles + 1))
        geno.append(g)

    # true per-(population, locus) sample sizes and the max depth to request.
    Nj = np.array([[int(np.sum(geno[p][:, l] != -9)) for l in range(L)] for p in range(P)])
    max_g = int(Nj.max())

    # structure file: locus-name header row, then one row per gene copy grouped by population.
    lines = [" ".join(f"L{l + 1}" for l in range(L))]
    for p in range(P):
        for r in range(copies[p]):
            row = [f"i{p}_{r}", f"p{p}"]
            row += [str(int(geno[p][r, l])) for l in range(L)]
            lines.append(" ".join(row))
    structure_text = "\n".join(lines) + "\n"
    data_lines = sum(copies)

    sizes = tuple(range(2, P))                  # pihat combination sizes 2..P-1
    if sizes:
        comb = 1
        k_range = str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]}-{sizes[-1]}"
    else:
        comb = 0
        k_range = "2"

    paramfile_text = (
        f"MAX_G {max_g}\nDATA_LINES {data_lines}\nLOCI {L}\nNON_DATA_ROWS 1\n"
        f"NON_DATA_COLS 2\nGROUP_BY_COL 2\nDATA_FILE data.stru\nR_OUT rich.out\n"
        f"P_OUT priv.out\nCOMB {comb}\nK_RANGE {k_range}\nC_OUT comb.out\nMISSING -9\n"
        f"TOLERANCE 1\nFULL_R 1\nFULL_P 1\nFULL_C 1\nPRINT_PROGRESS 0\n")

    meta = dict(P=P, L=L, max_g=max_g, sizes=sizes, pop_names=[f"p{p}" for p in range(P)])
    return structure_text, paramfile_text, meta


# ---- C++ output parsing ------------------------------------------------------------------

def parse_summary(path):
    """'name... g numLoci avg var se' -> {(name, g): (avg, var, se)}."""
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
        out[("_".join(t[:-5]), g)] = (avg, var, se)
    return out


def parse_full(path, n_loci):
    """'name... g numLoci v0..v_{L-1} mean var se' -> {(name, g): [per-locus values]}."""
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
            g = int(t[-(n_loci + 5)])
        except ValueError:
            continue
        out[("_".join(t[:-(n_loci + 5)]), g)] = vals
    return out


# ---- one comparison ----------------------------------------------------------------------

def run_one(seed):
    structure_text, paramfile_text, meta = make_case(seed)
    P, L, max_g, sizes = meta["P"], meta["L"], meta["max_g"], meta["sizes"]
    pop_names = meta["pop_names"]
    pmap = {p: i + 1 for i, p in enumerate(sorted(pop_names))}

    tmp = os.path.join(WORK, f"s{seed}")
    os.makedirs(tmp, exist_ok=True)
    try:
        with open(os.path.join(tmp, "data.stru"), "w") as f:
            f.write(structure_text)
        with open(os.path.join(tmp, "paramfile.txt"), "w") as f:
            f.write(paramfile_text)
        proc = subprocess.run([BIN, "paramfile.txt"], cwd=tmp,
                              capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            return dict(seed=seed, ok=False, n=0, npass=0, maxabs=0.0,
                        fail=f"C++ exit {proc.returncode}: {proc.stderr[-200:]}")

        cpp_sum = {}
        cpp_full = {}
        # alpha, pi
        for fn, kind in (("rich.out", "alpha"), ("priv.out", "pi")):
            for (name, g), v in parse_summary(os.path.join(tmp, fn)).items():
                cpp_sum[(f"{kind}_{pmap[name]}", g)] = v
            for (name, g), vals in parse_full(
                    os.path.join(tmp, fn + "_fulldata"), L).items():
                cpp_full[(f"{kind}_{pmap[name]}", g)] = vals
        # pihat (one file per combination size)
        for k in sizes:
            for (name, g), v in parse_summary(os.path.join(tmp, f"comb.out_{k}")).items():
                key = "pihat_" + "".join(str(pmap[p]) for p in name.split("_"))
                cpp_sum[(key, g)] = v
            for (name, g), vals in parse_full(
                    os.path.join(tmp, f"comb.out_{k}_fulldata"), L).items():
                key = "pihat_" + "".join(str(pmap[p]) for p in name.split("_"))
                cpp_full[(key, g)] = vals

        # PADZE classical
        loci = read_structure(os.path.join(tmp, "data.stru"), ploidy=1,
                              one_row_per_individual=False, header_rows=1)
        res = classical_features(loci, max_g=max_g, pihat_sizes=sizes)

        # structural check: identical emitted (statistic, depth) rows.
        if set(res.summary) != set(cpp_sum):
            only_cpp = sorted(set(cpp_sum) - set(res.summary))[:4]
            only_py = sorted(set(res.summary) - set(cpp_sum))[:4]
            return dict(seed=seed, ok=False, n=0, npass=0, maxabs=0.0,
                        fail=f"key set mismatch: only_cpp={only_cpp} only_py={only_py}")

        n = npass = 0
        maxabs = 0.0
        worst = None
        for key, (avg, var, se) in cpp_sum.items():
            pm, pv, ps = res.summary[key]
            for a, b, tag in ((pm, avg, "mean"), (pv, var, "var"), (ps, se, "se")):
                d = abs(a - b)
                n += 1
                if d <= TOL:
                    npass += 1
                if d > maxabs:
                    maxabs = d
                    worst = (key, tag, a, b)
        for key, vals in cpp_full.items():
            pyvals = res.per_locus[key]
            if len(pyvals) != len(vals):
                return dict(seed=seed, ok=False, n=n, npass=npass, maxabs=maxabs,
                            fail=f"per-locus length {key}: py={len(pyvals)} cpp={len(vals)}")
            for a, b in zip(pyvals, vals):
                d = abs(a - b)
                n += 1
                if d <= TOL:
                    npass += 1
                if d > maxabs:
                    maxabs = d
                    worst = (key, "locus", a, b)

        ok = (npass == n)
        out = dict(seed=seed, ok=ok, n=n, npass=npass, maxabs=maxabs, fail=None)
        if not ok:
            out["fail"] = f"value mismatch worst={worst}"
        return out
    except Exception as e:  # noqa: BLE001
        return dict(seed=seed, ok=False, n=0, npass=0, maxabs=0.0, fail=f"exception: {e!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1)
    args = ap.parse_args()

    if not os.path.isfile(BIN):
        raise SystemExit(f"C++ binary not found at {BIN}; set ADZE_BIN")
    os.makedirs(WORK, exist_ok=True)

    seeds = list(range(args.seed0, args.seed0 + args.n))
    t0 = time.time()
    tot_cmp = tot_pass = 0
    max_abs = 0.0
    n_cases_ok = 0
    failures = []

    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, seeds, chunksize=16), 1):
            tot_cmp += r["n"]
            tot_pass += r["npass"]
            max_abs = max(max_abs, r["maxabs"])
            if r["ok"]:
                n_cases_ok += 1
            else:
                if len(failures) < 25:
                    failures.append((r["seed"], r["fail"]))
            if i % 1000 == 0:
                dt = time.time() - t0
                print(f"  {i}/{args.n} cases  ({i/dt:.0f}/s)  "
                      f"comparisons={tot_cmp:,}  max|d|={max_abs:.2e}", file=sys.stderr)

    dt = time.time() - t0
    print("\n==== PADZE vs C++ ADZE: 10k parity harness ====")
    print(f"cases run              : {args.n}")
    print(f"cases fully matching   : {n_cases_ok}")
    print(f"total comparisons      : {tot_cmp:,}")
    print(f"comparisons passing    : {tot_pass:,}")
    print(f"pass rate              : {100.0 * tot_pass / tot_cmp:.6f}%"
          if tot_cmp else "pass rate: n/a")
    print(f"max |py - cpp|         : {max_abs:.3e}   (threshold {TOL:g})")
    print(f"wall time              : {dt:.1f}s  ({args.n/dt:.0f} cases/s, {args.workers} workers)")
    if failures:
        print(f"\nFAILURES ({len(failures)} shown, seeds reproducible):")
        for seed, msg in failures:
            print(f"  seed {seed}: {msg}")
    else:
        print("\nno failures: every comparison within threshold and all row sets identical")
    return 0 if (n_cases_ok == args.n and not failures) else 1


def test_cpp_parity_10k_manual_harness_collects_explicitly():
    """Pytest-visible entrypoint for the manual randomized C++ parity harness."""
    if pytest is None:
        return
    enabled = os.environ.get("ADZE_RUN_10K_PARITY", "").lower() in {"1", "true", "yes"}
    if not enabled:
        pytest.skip("manual 10k C++ parity harness; set ADZE_RUN_10K_PARITY=1 and ADZE_BIN")

    n = os.environ.get("ADZE_10K_N", "10000")
    workers = os.environ.get("ADZE_10K_WORKERS", "8")
    seed0 = os.environ.get("ADZE_10K_SEED0", "1")
    timeout = int(os.environ.get("ADZE_10K_TIMEOUT", "3600"))
    proc = subprocess.run(
        [sys.executable, __file__, "--n", n, "--workers", workers, "--seed0", seed0],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


if pytest is not None:
    test_cpp_parity_10k_manual_harness_collects_explicitly = pytest.mark.manual(
        test_cpp_parity_10k_manual_harness_collects_explicitly)


if __name__ == "__main__":
    raise SystemExit(main())
