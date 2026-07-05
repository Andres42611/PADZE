# PADZE and ADZE 1.0 Reference

Date checked: 2026-07-05

**PADZE** (Pythonic Allelic Diversity Analyzer, package `padze`) is the modern,
standalone Python successor to ADZE 1.0. It reproduces the original analyses and
extends them with new statistics, direct VCF support, a Python API, and a
pip-installable distribution.

Original ADZE 1.0 resources — source, manual, and prebuilt binaries — live
upstream at https://github.com/szpiech/ADZE. PADZE does **not** bundle the
original ADZE manual PDF or binaries; use the upstream repository for those.

## What ADZE 1.0 Provides Publicly

The ADZE 1.0 GitHub repository (https://github.com/szpiech/ADZE) is small and
release-oriented. The public page contains a short README, `src/`,
`ADZE_Manual.pdf`, and platform archives for Linux, macOS, Windows, and x86_64
builds.

The upstream manual documents the elements scientific users need:

- software name, version, author/contact, and availability URL
- citation for the ADZE Bioinformatics paper
- formula/method description for allelic richness, private allelic richness, and
  combination-private richness
- installation and run instructions
- input file format expectations (STRUCTURE-format input)
- parameter-file semantics
- output file format definitions (R, P, and C files)
- small example and larger data example
- references

For the original manual, prebuilt binaries, and C++ source, go to
https://github.com/szpiech/ADZE.

## PADZE vs ADZE 1.0

PADZE keeps the proven ADZE 1.0 method and rarefaction estimators while
upgrading the tooling around them. Each row is a genuine improvement; PADZE also
retains STRUCTURE input and reproduces ADZE 1.0 outputs for parity.

| Area | ADZE 1.0 | PADZE (upgrade) |
| --- | --- | --- |
| **Input** | STRUCTURE-format files only | Reads VCF + population map directly, and retains STRUCTURE input |
| **Interface** | Single C++ command-line binary | Python API plus a `padze` CLI |
| **Statistics** | alpha (allelic richness), pi (private allelic richness), pihat (combination-private richness); mean, variance, standard error | Retains alpha / pi / pihat with mean / variance / SE, and adds skewness and excess kurtosis across loci |
| **Outputs** | R, P, C output files from STRUCTURE input | Generates ADZE-compatible R / P / C plus FULL output files directly from VCF |
| **Portability** | Per-platform prebuilt binaries | Pip-installable, cross-platform pure Python |

### Details

- **Input.** ADZE 1.0 requires STRUCTURE-format input. PADZE reads VCF together
  with a population map (popmap) directly, removing the manual conversion step,
  while still accepting STRUCTURE-format input for existing pipelines.
- **Interface.** ADZE 1.0 ships as a C++ command-line binary. PADZE exposes both
  a Python import/API for scripted analysis and a `padze` CLI for terminal use.
- **Statistics.** PADZE retains the ADZE 1.0 estimators — alpha (allelic
  richness), pi (private allelic richness), and pihat (combination-private
  richness) — with their mean, variance, and standard error, and adds skewness
  and excess kurtosis across loci for a fuller picture of the rarefaction
  distribution.
- **Outputs.** PADZE generates ADZE-compatible R (richness), P (private), and C
  (combination) files, plus consolidated FULL output, directly from VCF.
- **Portability.** PADZE installs with `pip` and runs anywhere Python runs, in
  place of maintaining per-platform prebuilt binaries.

## Parity and Reproducibility

PADZE is validated against ADZE 1.0: from the same input it reproduces the R, P,
and C outputs of the original tool, so existing analyses and published pipelines
carry over unchanged. This makes PADZE a drop-in upgrade path — same results
where they overlap, with new capabilities layered on top.

## Citation and Attribution

Cite the original ADZE method paper:

> Szpiech ZA, Jakobsson M, Rosenberg NA. 2008. "ADZE: a rarefaction approach for
> counting alleles private to combinations of populations." *Bioinformatics*
> 24:2498–2504. doi:10.1093/bioinformatics/btn478

PADZE is an independent implementation and is not endorsed by or affiliated with
the original ADZE authors.

## Licensing Notes

PADZE source and project documentation are distributed under PADZE's own license.
Original ADZE 1.0 files (source, manual, binaries) remain governed by their
upstream terms at https://github.com/szpiech/ADZE and are referenced there rather
than redistributed with PADZE.
