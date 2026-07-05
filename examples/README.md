# PADZE Minimal VCF Example

A tiny, self-contained trio dataset for a fast end-to-end tour of PADZE. The two
input files mirror `../tests/fixtures/trio.vcf` and `../tests/fixtures/trio.popmap`,
staged here as publication-facing examples separate from the test suite.

## Example inputs

- `trio.vcf` — 7 biallelic loci across 6 diploid samples.
- `trio.popmap` — two-column `sample_id population` map assigning the 6 samples to
  populations A, B, and C (2 samples each).

Run the commands below from the `GitHub/` directory. They use the `padze` console
command installed by `python -m pip install -e .`; from a source checkout you can run
the identical pipeline with `PYTHONPATH=src python -m padze` instead.

## Inspect input

```bash
padze info --vcf examples/trio.vcf --popmap examples/trio.popmap
```

Expected summary:

```text
populations (3): A, B, C
samples/pop: A=2, B=2, C=2
loci: kept 7 / read 7
max usable rarefaction depth: 3
```

## Extended PADZE feature table

```bash
padze features \
  --vcf examples/trio.vcf \
  --popmap examples/trio.popmap \
  --population-order A B C \
  --out features.csv
```

This writes one row per rarefaction depth with mean, variance, standard error,
skewness, and kurtosis for each statistic — the 3rd/4th-moment upgrade the C++
ADZE omits. `--population-order A B C` pins the numeric labels (`alpha_1`,
`pihat_13`, ...) to the population topology you choose.

## Classical ADZE-compatible summary

```bash
padze features \
  --vcf examples/trio.vcf \
  --popmap examples/trio.popmap \
  --max-depth 3 \
  --classical \
  --out classical.csv
```

This writes `statistic,g,n_loci,mean,variance,se` rows matching the classical ADZE
allelic richness (α), private (π), and combination-private (π̂) outputs.
