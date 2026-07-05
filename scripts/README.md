# scripts/ — reproduction, benchmarking, and data utilities

Command-line helpers that reproduce, benchmark, and stage inputs for PADZE. Run them
from the `GitHub/` directory.

## Subdirectories

- [repro/](repro/) — reproducibility gates.
  - [validate_padze.py](repro/validate_padze.py): compact entry point that runs the core
    PADZE test suite, the optional H952 competition comparison, and an optional
    PADZE-vs-original-ADZE benchmark smoke. This is the fastest way for reviewers to
    verify the PADZE upgrade claims without running the full training pipeline.

    ```bash
    python scripts/repro/validate_padze.py
    ```

- [benchmarks/](benchmarks/README.md) — comparison/benchmark drivers and plotters.
  - [benchmark_padze_vs_original.py](benchmarks/benchmark_padze_vs_original.py): PADZE vs
    original ADZE timing + numeric parity benchmark on matched STRUCTURE/VCF data.
  - Plus the FashionMNIST ADZE-style digest and DNNaic regression benchmarks.

- [data/](data/README.md) — external-data acquisition.
  - [fetch_hgdp_rosenberg2005.py](data/fetch_hgdp_rosenberg2005.py): fetch the HGDP-CEPH
    microsatellite files used by the human ADZE-paper reproduction pipeline.
