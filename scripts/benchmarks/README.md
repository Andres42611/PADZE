# scripts/benchmarks/ — benchmark runs + plotting

> Last updated: 2026-06-15. Comparison/benchmark drivers and plotters. The FashionMNIST
> ADZE-style digest is a non-genetic testbed (its paper lives in [../../papers/fashion_mnist_adze/](../../papers/fashion_mnist_adze/README.md));
> outputs land in [../../results/](../../results/README.md).

## Key files
- [benchmark_models.py](benchmark_models.py): DNNaic-vs-baseline regression/direction benchmark driver.
- [benchmark_padze_vs_original.py](benchmark_padze_vs_original.py): PADZE vs original ADZE timing + numeric parity benchmark on matched STRUCTURE/VCF data.
- [reg_barplot.R](reg_barplot.R): plot regression metrics from `../../results/metrics/*.json`.
- [fashion_mnist_adze_stats.py](fashion_mnist_adze_stats.py): FashionMNIST ADZE-style feature digest + FC DNN runs.
- [fashion_mnist_cnn_compare.py](fashion_mnist_cnn_compare.py): raw-CNN vs CNN+ADZE-rich comparison.
- [popgen_benchmarks.py](popgen_benchmarks.py): pop-gen benchmark helpers.
- [train_real_target_signature_split.py](train_real_target_signature_split.py): real-array target-signature split run.

> Note: the legacy notebook-export benchmark lives at `../../needs_review/benchmark_thesismodels.py` (pending conversion).
