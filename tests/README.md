# tests/ — unit tests

> Last updated: 2026-07-04. Tests for PADZE
> ([../src/padze/](../src/padze/README.md)) plus the FashionMNIST digest.
> Run via `pytest` after installing `.[test]`. Several core formula/parity files also
> support direct `python tests/<file>.py` execution.

## Files
- `test_cpp_parity.py`: parity vs the real C++ ADZE binary (`|Δ| ≤ 3.3e-6`).
- `test_cpp_parity_10k.py`: marked manual 10k randomized C++ parity harness; pytest
  collects it and skips unless `ADZE_RUN_10K_PARITY=1` is set.
- `test_original_adze_example.py`: exact regression against original ADZE's packaged
  `small_*` example outputs, including FULL files and deleted-loci output.
- `test_adze_rarefaction.py`: rarefaction vs a naive C++-reference implementation.
- `test_adze_moments.py`: across-loci moments vs hand-computed + SciPy + edge cases.
- `test_adze_io_features.py`: VCF/STRUCTURE end-to-end feature extraction, metadata,
  population order, and ADZE tolerance behavior.
- `test_vcf_structure_equivalence.py`: matched VCF/STRUCTURE genotype encodings produce
  identical count matrices and classical statistics.
- `test_vcf_allele_coding_invariance.py`: REF/ALT swaps and multiallelic allele-index
  permutations leave ADZE summaries unchanged.
- `test_multi_population_pihat.py`: pihat combinations for five populations, including
  the all-population combination.
- `test_hgdp_competition_outputs.py`: H952 ADZE 1.0 vs PADZE competition artifacts
  when `data/competition/` is present.
- `test_fashion_mnist_adze_stats.py`: FashionMNIST ADZE-feature digest sanity checks;
  skips when optional ML packages are absent.
- `fixtures/`: `trio.vcf` + `trio.popmap` test inputs.
