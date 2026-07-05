# Changelog

All notable changes to PADZE (Pythonic Allelic Diversity Analyzer) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-05

First public release of `padze`, a standalone Python successor to and upgrade
over ADZE 1.0 (Szpiech, Jakobsson & Rosenberg 2008).

### Added

- Direct VCF input with a simple sample-to-population popmap — no STRUCTURE
  conversion required.
- Importable Python API (`from padze import compute_features, read_vcf, ...`).
- `padze` command-line interface with `info` and `features` subcommands.
- ADZE-compatible outputs generated directly from VCF: R/P/C files,
  FULL_R/FULL_P/FULL_C per-locus files, and the deleted-loci report.
- Classical exact-match mode reproducing the C++ ADZE mean/variance/
  standard-error output.
- Extended across-loci moments: skewness and excess kurtosis in addition to
  mean, variance, and standard error.
- Rolling-window mode for classical statistics over sliding genomic windows.
- STRUCTURE input retained for parity and legacy workflows.
- Validated against the original C++ ADZE (parity tests), backed by an
  automated suite of 82 passing tests.
