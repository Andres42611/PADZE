# PADZE Usage & Verification

**PADZE** (Pythonic Allelic Diversity Analyzer) is a NumPy implementation and
extension of the ADZE allelic-rarefaction statistics with direct VCF and
STRUCTURE input. This guide walks a user or reviewer through installing the
package, running its verified example commands, and confirming correctness with
the bundled test suite and validation driver.

Every command below was run from the repository's `GitHub/` directory on
2026-07-05 and the outputs shown are real. To run the CLI straight from a
checkout without installing, prefix commands with `PYTHONPATH=src` as shown.

- Package / import name: `padze`
- Console command: `padze`
- Version: `0.1.0`
- Example inputs: `examples/trio.vcf`, `examples/trio.popmap`

---

## 1. Install

Plain install (runtime dependencies only):

```bash
pip install .
```

Install with the test extras (adds `pytest` and `scipy` for the verification
suite):

```bash
pip install -e ".[test]"
```

Both expose the `padze` console command and the `padze` importable package.
No install is required to exercise the CLI from a checkout — set `PYTHONPATH`
to the `src/` directory instead:

```bash
PYTHONPATH=src python -m padze --help
```

Confirm the package imports and exposes its public API:

```bash
PYTHONPATH=src python -c "import padze; print(padze.__version__); print(padze.compute_features.__name__); print(padze.read_vcf.__name__)"
```

```text
0.1.0
compute_features
read_vcf
```

---

## 2. Usage — verified example commands

The three commands below use the packaged `examples/trio.vcf` (three
populations A, B, C; two diploid samples each; seven loci) and its
`examples/trio.popmap`.

### 2a. `info` — read input and print metadata

```bash
PYTHONPATH=src python -m padze info \
  --vcf examples/trio.vcf \
  --popmap examples/trio.popmap
```

```text
source: VCF:examples/trio.vcf
populations (3): A, B, C
samples/pop: A=2, B=2, C=2
loci: kept 7 / read 7
filters: drop all-missing loci
missing call fraction (kept): 0.0119
ADZE feature shape per locus: alpha_j, pi_j (j=1..P) and pihat over population combinations; summarized across loci as (mean, variance, se, skewness, kurtosis) per rarefaction depth.
max usable rarefaction depth: 3
```

This confirms the reader parses the VCF, maps samples to populations, applies
missingness filtering, and reports the usable rarefaction depth.

### 2b. `features` — the across-loci feature table (CSV)

```bash
PYTHONPATH=src python -m padze features \
  --vcf examples/trio.vcf \
  --popmap examples/trio.popmap \
  --out features.csv
```

```text
wrote 2 rows x 46 cols -> features.csv
```

The table has one row per rarefaction depth (here depths 2 and 3) and 46
columns: the depth `g` plus five moments — mean, variance, se, skewness,
kurtosis — for each of the nine per-locus statistics (`alpha_1..3`, `pi_1..3`,
and the pairwise `pihat_12`, `pihat_13`, `pihat_23`). Using `--out -` (the
default) writes the same CSV to stdout. The header and first data row look
like:

```text
g,alpha_1_mean,alpha_1_variance,alpha_1_se,alpha_1_skewness,alpha_1_kurtosis,alpha_2_mean,...,pihat_23_skewness,pihat_23_kurtosis
2,1.380952380952381,0.07142857142857144,0.10101525445522108,-0.9839173128183324,-0.8641975308641985,...
```

Output is deterministic: identical inputs and flags always produce identical
CSV.

### 2c. `features --adze-prefix ... --adze-full` — ADZE-compatible R/P/C files

```bash
PYTHONPATH=src python -m padze features \
  --vcf examples/trio.vcf \
  --popmap examples/trio.popmap \
  --adze-prefix out \
  --adze-full
```

```text
wrote ADZE-compatible files: out_r, out_r_fulldata, out_p, out_p_fulldata, out_c_2, out_c_2_fulldata
```

This writes six files in the original ADZE layout:

| File | Contents |
| --- | --- |
| `out_r` | allelic richness (`alpha`) summaries, one block per population |
| `out_r_fulldata` | FULL_R: per-locus `alpha` values plus AVG/VAR/STD_ERR |
| `out_p` | private allelic richness (`pi`) summaries |
| `out_p_fulldata` | FULL_P: per-locus `pi` values plus AVG/VAR/STD_ERR |
| `out_c_2` | pairwise combination-private richness (`pihat`) summaries |
| `out_c_2_fulldata` | FULL_C: per-locus `pihat` values plus AVG/VAR/STD_ERR |

Real content of `out_r` (columns: population, depth `g`, `NUM_LOCI`, mean,
variance, SE):

```text
A 2 7 1.380952380952381 0.07142857142857144 0.10101525445522108
A 3 7 1.5714285714285714 0.1607142857142857 0.1515228816828316

B 2 7 1.3095238095238098 0.08730158730158731 0.11167656571008165
B 3 7 1.4642857142857142 0.19642857142857142 0.16751484856512247
B 4 7 1.5714285714285714 0.2857142857142857 0.20203050891044214

C 2 7 1.4761904761904763 0.050264550264550255 0.08473871628596277
C 3 7 1.7142857142857142 0.11309523809523814 0.1271080744289442
C 4 7 1.8571428571428572 0.1428571428571429 0.14285714285714288
```

The matching FULL_R file carries the per-locus values behind those summaries
(header truncated for width):

```text
POP_GROUPING G NUM_LOCI chr1:100 chr1:200 chr1:300 chr1:400 chr1:500 chr1:600 chr1:700 AVG VAR STD_ERR
A 2 7 1.5 1.5 1.5 1.5 1 1.6666666666666667 1 1.380952380952381 0.07142857142857144 0.10101525445522108
A 3 7 1.75 1.75 1.75 1.75 1 2 1 1.5714285714285714 0.1607142857142857 0.1515228816828316
```

Drop `--adze-full` to write just the three summary files (`out_r`, `out_p`,
`out_c_2`).

---

## 3. Running the test suite

With the test extras installed (Section 1), run the full suite from `GitHub/`:

```bash
pytest
```

Observed result:

```text
82 passed, 11 skipped
```

The 11 skips are **optional extended gates** that require external artifacts: a
locally built C++ ADZE binary (with GSL) for on-the-fly parity, the 10k-locus
manual parity harness, and the large HGDP / competition datasets. The 82
passing tests fully exercise PADZE's own numerics, I/O, and CLI; the skipped
gates are additional cross-checks that light up automatically once those
external artifacts are present.

---

## 4. The validation driver

`scripts/repro/validate_padze.py` is the compact, reviewer-facing entry point.
It runs the correctness-critical test modules and then, when the optional local
artifacts exist, the H952 competition comparison and an ADZE-binary benchmark
smoke test.

```bash
python scripts/repro/validate_padze.py
```

Observed output:

```text
+ .../python -m pytest -q tests/test_adze_rarefaction.py tests/test_adze_moments.py tests/test_adze_io_features.py tests/test_classical_rolling.py tests/test_vcf_structure_equivalence.py tests/test_vcf_allele_coding_invariance.py tests/test_multi_population_pihat.py tests/test_cpp_parity.py tests/test_original_adze_example.py tests/test_hgdp_competition_outputs.py
........................................................................
.....ssssssssss
SKIP H952 comparison: data/competition outputs are absent
SKIP benchmark smoke: no runnable ADZE binary found
PADZE validation completed
```

The driver runs the core gates green and cleanly announces which optional
cross-checks it skipped and why. Its flags let you demand the extended gates
when their artifacts are staged:

```bash
python scripts/repro/validate_padze.py --help
```

```text
usage: validate_padze.py [-h] [--adze-bin ADZE_BIN]
                         [--benchmark {auto,always,never}]
                         [--h952 {auto,always,never}]
```

- `--adze-bin PATH` — point at an original/rebuilt ADZE binary for the
  benchmark smoke.
- `--benchmark {auto,always,never}` — `always` forces the PADZE-vs-ADZE
  benchmark and fails if no binary is found; `auto` (default) skips gracefully.
- `--h952 {auto,always,never}` — `always` forces the local H952 competition
  comparison against staged `data/competition/` outputs.

For a fully-loaded run against the original ADZE and the H952 comparison data:

```bash
python scripts/repro/validate_padze.py --h952 always --benchmark always --adze-bin /path/to/adze-1.0
```

---

## 5. What each core gate proves

| Gate (test module) | What it proves |
| --- | --- |
| `test_cpp_parity.py` | Runs the vendored **C++ ADZE reference and PADZE on the same STRUCTURE dataset** and asserts `alpha`, `pi`, and `pihat` — with their across-loci mean, variance, and SE — agree to a tight tolerance. The strongest correctness check available. |
| `test_original_adze_example.py` | Reproduces the concrete example distributed with ADZE 1.0 (`small_data.stru` → `small_r`, `small_p`, `small_c_2`, and the FULL_* files), matching every summary and per-locus value to the precision printed by the original program. |
| `test_adze_rarefaction.py` | Validates the rarefaction core — the absence-probability / hypergeometric machinery behind `alpha`, `pi`, `pihat`. |
| `test_adze_moments.py` | Confirms the across-loci moments: hand-computed mean/variance/SE/skewness/kurtosis, plus **parity with `scipy.stats.skew`/`kurtosis`** (both bias-corrected sample and biased population forms), and that streaming accumulation equals the batch computation. |
| `test_vcf_structure_equivalence.py` | Locks the **VCF ↔ STRUCTURE upgrade path**: the VCF and STRUCTURE readers yield identical aligned allele-count loci and identical classical statistics for the same diploid data, including missing calls and multiallelic loci. |
| `test_vcf_allele_coding_invariance.py` | The statistics are invariant to arbitrary relabeling of allele codes — the output depends on the partition of alleles, not their symbols. |
| `test_multi_population_pihat.py` | Exercises the combination-private richness (`pihat`) across multiple populations and combination sizes. |
| `test_classical_rolling.py` | Checks the classical (mean, variance, SE) exact-match mode and the rolling-genomic-window mode. |
| `test_adze_io_features.py` | End-to-end I/O and feature-table assembly, including the ADZE-compatible R/P/C and FULL_R/P/C writers exercised in Section 2c. |
| `test_hgdp_competition_outputs.py` | Compares PADZE against staged ADZE 1.0 outputs on real HGDP-scale data when present (optional). |

Together these gates establish that PADZE reproduces the original C++ ADZE
(rarefaction core; classical mean/variance/SE; FULL_R/P/C per-locus tables),
that its VCF and STRUCTURE front-ends are equivalent, and that its added
higher-moment statistics match `scipy` to machine precision.
