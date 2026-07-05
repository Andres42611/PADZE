# src/padze/ — PADZE allelic-rarefaction stats (Python)

> Last updated: 2026-07-04. Vectorized, NumPy-only Python port of ADZE (Szpiech,
> Jakobsson & Rosenberg 2008) that computes allelic richness (α), private (π), and
> combination-private (π̂) richness by rarefaction over sample size *g* — and adds
> skewness + excess kurtosis (the 3rd/4th-moment gap the C++ omits).
> Verified against the C++ binary, original ADZE example outputs, matched VCF/STRUCTURE
> encodings, and missing-data parity cases. Design rationale lives in the module
> docstrings of the files below.

## Key modules
- [rarefaction.py](rarefaction.py): per-locus α/π/π̂ at all depths (`locus_statistics`).
- [moments.py](moments.py): across-loci mean/var/se + bias-corrected skew/kurtosis; streaming `MomentAccumulator`.
- [features.py](features.py): assemble the feature table (`compute_features` → row per depth `g`).
- [io.py](io.py): `read_vcf` (recommended input) and `read_structure` (C++ `.stru` for parity).
- [cli.py](cli.py) / [__main__.py](__main__.py): `python -m padze {info,features}`.

## Install / run / test
```bash
python -m pip install -e .
padze features --vcf tests/fixtures/trio.vcf --popmap tests/fixtures/trio.popmap --out features.csv
padze features --vcf tests/fixtures/trio.vcf --popmap tests/fixtures/trio.popmap --adze-prefix adze_out --adze-full
python tests/test_cpp_parity.py   # runs real C++ ADZE when a binary or GSL toolchain is available
```
Popmaps are two-column text files: `sample_id population`. Use `--population-order` when
the numeric output labels (`alpha_1`, `pihat_13`, etc.) must follow a specific topology.
`--adze-prefix` writes ADZE-style `R_OUT`/`P_OUT`/`C_OUT` files from VCF directly; add
`--adze-full` for FULL_R/FULL_P/FULL_C-style per-locus files.
For ADZE-compatible VCF runs, `--min-populations-genotyped` controls whether ragged loci
with fully missing populations are retained for classical row suppression. For extended
feature tables, `--depth-policy common|ragged` makes rectangular versus available-locus
moment summaries explicit.

Original ADZE-style STRUCTURE files can be read by passing the paramfile-equivalent layout:

```bash
padze info \
  --structure ADZEOriginal/ADZE-1.0/small_data.stru \
  --structure-header-rows 1 \
  --structure-meta-columns 5 \
  --structure-pop-column 5 \
  --max-missing-per-population 0.2
```

Other tests: `tests/test_adze_*.py`, `tests/test_classical_rolling.py`,
`tests/test_vcf_structure_equivalence.py`, `tests/test_vcf_allele_coding_invariance.py`,
`tests/test_multi_population_pihat.py`, and `tests/test_cpp_parity.py`.

## Optional / deep-dive (read the code, not a restatement)
- Formulas (hypergeometric Q, α/π/π̂, G1/G2 moments), VCF filters, missing-data handling,
  and the streaming-accumulator design are documented in the module docstrings of the files above.
