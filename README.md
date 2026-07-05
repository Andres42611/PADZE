# PADZE — Pythonic Allelic Diversity Analyzer

Rarefaction-based allelic diversity statistics for population genetics, as a modern, importable Python package with direct VCF input.

[![tests](https://github.com/Andres42611/PADZE/actions/workflows/tests.yml/badge.svg)](https://github.com/Andres42611/PADZE/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PADZE (`padze`) is a standalone Python successor to [ADZE 1.0](https://github.com/szpiech/ADZE). It computes allelic richness, private allelic richness, and combination-private richness across standardized sample depths using the rarefaction approach of Szpiech, Jakobsson & Rosenberg (2008), and adds direct VCF input, an importable Python API, a `padze` command-line tool, and higher-order across-loci moments.

## What it computes

For every rarefaction depth *g*, PADZE standardizes populations to a common sample size and reports:

- **Allelic richness** (`alpha`) — expected number of distinct alleles per population at depth *g*.
- **Private allelic richness** (`pi`) — expected number of alleles unique to a single population.
- **Combination-private richness** (`pihat`) — expected number of alleles private to a combination of populations.

Each statistic is summarized **across loci** with the classical **mean, variance, and standard error**, and — new in PADZE — the **skewness** and **excess kurtosis** of its per-locus distribution.

## Relationship to ADZE 1.0

PADZE is a genuine upgrade and successor to ADZE 1.0. It reimplements the rarefaction core in vectorized NumPy and extends it: reads VCF (including `.vcf.gz`) directly with a simple population map, exposes an importable Python API (`from padze import ...`), ships the `padze` command-line tool, and adds third- and fourth-moment (skewness, excess-kurtosis) summaries alongside the classical mean/variance/SE outputs. It retains the STRUCTURE input format for direct parity with ADZE 1.0, and its classical mode reproduces the original C++ ADZE output. PADZE is an independent implementation and is not endorsed by the original ADZE authors.

## Install

Install the released package from [PyPI](https://pypi.org/project/padze/):

```bash
pip install padze
```

This installs the `padze` command-line tool and the importable `padze` module. PADZE requires Python 3.10+ and depends only on NumPy at runtime.

To work from a checkout of this repository instead:

```bash
pip install .            # runtime install from source
pip install -e ".[test]" # editable install with the test suite
pip install -e ".[dev]"  # editable install with build/release tooling
```

## Quick start

The commands below use the example inputs in [`examples/`](examples/) and run from the repository root.

Inspect a VCF and its population map:

```bash
padze info --vcf examples/trio.vcf --popmap examples/trio.popmap
```

Compute the extended feature table (mean/variance/SE plus skewness/kurtosis, one row per depth):

```bash
padze features --vcf examples/trio.vcf --popmap examples/trio.popmap \
    --population-order A B C --out features.csv
```

Write ADZE-compatible `R`/`P`/`C` output files, including per-locus `FULL_R`/`FULL_P`/`FULL_C` tables:

```bash
padze features --vcf examples/trio.vcf --popmap examples/trio.popmap \
    --population-order A B C --adze-prefix out --adze-full
```

This writes `out_r`, `out_r_fulldata`, `out_p`, `out_p_fulldata`, `out_c_2`, and `out_c_2_fulldata`.

> Running from a checkout without installing? Prefix any command with `PYTHONPATH=src` and use the module form, e.g. `PYTHONPATH=src python -m padze info --vcf examples/trio.vcf --popmap examples/trio.popmap`.

## CLI overview

PADZE exposes two subcommands. Install the package to get the `padze` console command, or use `python -m padze`.

| Subcommand | Purpose |
| --- | --- |
| `padze info` | Read an input, print population/sample metadata and the maximum usable rarefaction depth. |
| `padze features` | Compute across-loci statistics and write CSV or ADZE-compatible output. |

Key `features` flags:

- `--population-order A B C` — fix the population order used for numbered `alpha`/`pi`/`pihat` outputs.
- `--classical` — emit the classical `(mean, variance, se)` table that reproduces the C++ ADZE output.
- `--adze-prefix PREFIX` — write ADZE-compatible `PREFIX_r`, `PREFIX_p`, and `PREFIX_c_k` files.
- `--adze-full` — with `--adze-prefix`, also write the per-locus `FULL_R`/`FULL_P`/`FULL_C`-style `*_fulldata` files.
- `--rolling-window W` — compute the classical statistics over sliding genomic windows of size `W` (`--step`, `--window-unit loci|bp`).
- `--max-depth`, `--pihat-sizes`, `--moments` — control the depth range, which combination sizes to score, and which moments to emit.

Run `padze features --help` for the full flag list, including VCF filtering (`--require-pass`, `--max-missing-fraction`, `--biallelic-only`) and STRUCTURE input options.

## Python API

```python
from padze import compute_features, read_vcf

loci = read_vcf("examples/trio.vcf", "examples/trio.popmap",
                population_order=["A", "B", "C"])
table = compute_features(loci)          # one row per rarefaction depth
matrix, columns = table.to_frame()      # NumPy array + column names
```

The public API also exposes `read_structure`, `classical_features`, `rolling_window_features`, and the rarefaction and moment primitives (`locus_statistics`, `moments_from_values`, `MomentAccumulator`). See the [manual](docs/MANUAL.md) for details.

## Outputs

- **ADZE-compatible `R`/`P`/`C` files** (`--adze-prefix`) — richness (`_r`), private richness (`_p`), and combination-private richness (`_c_k`) tables in the original ADZE text layout.
- **Per-locus `FULL_R`/`FULL_P`/`FULL_C` files** (`--adze-full`) — the full per-locus values behind each summary, as `*_fulldata`.
- **Deleted-loci report** — when missingness tolerances remove loci, a `*_p_deletedloci` companion lists them.
- **Extended feature table** (`--out`) — a CSV with mean, variance, standard error, skewness, and excess kurtosis for every statistic at every depth.

## Validation

PADZE ships with an automated test suite: **82 tests pass** (11 skipped for optional external tools). The rarefaction core, the classical mean/variance/SE outputs, and the per-locus `FULL_R`/`FULL_P`/`FULL_C` outputs are validated against the original C++ ADZE. The VCF path is validated against matched STRUCTURE encodings, allele-coding permutations, and missingness edge cases. See [`docs/usage-and-verification.md`](docs/usage-and-verification.md) for the commands and observed results.

```bash
pip install -e ".[test]"
pytest
```

## Documentation

- [`PADZE_Manual.pdf`](PADZE_Manual.pdf) — the PADZE manual (printable PDF, at the repository root).
- [`docs/MANUAL.md`](docs/MANUAL.md) — the same manual in Markdown for reading on GitHub.
- [`docs/adze-1.0-reference.md`](docs/adze-1.0-reference.md) — mapping to ADZE 1.0 concepts and formats.
- [`docs/usage-and-verification.md`](docs/usage-and-verification.md) — verified usage and validation commands.

## Citation

If you use PADZE, please cite both the package and the original ADZE method.

**PADZE:**

> del Castillo A, Shmalo Y. 2026. PADZE: Pythonic Allelic Diversity Analyzer (version 0.1.0). https://github.com/Andres42611/PADZE

**Original ADZE method:**

> Szpiech ZA, Jakobsson M, Rosenberg NA. 2008. ADZE: a rarefaction approach for counting alleles private to combinations of populations. *Bioinformatics* 24:2498–2504. doi:[10.1093/bioinformatics/btn478](https://doi.org/10.1093/bioinformatics/btn478)

## License

PADZE is released under the [MIT License](LICENSE).

## Authors

- **Andres del Castillo**
- **Yitzchak Shmalo** — Hebrew University of Jerusalem (yitzchak.shmalo@mail.huji.ac.il)
