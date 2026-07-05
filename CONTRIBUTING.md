# Contributing to PADZE

PADZE (Pythonic Allelic Diversity Analyzer) is the standalone Python successor to
ADZE 1.0. Contributions are welcome — this guide gets you set up and productive.

## Getting set up

Clone the repository and install in editable mode with the development extras:

```bash
git clone <repo-url>
cd padze
pip install -e ".[dev]"
```

This installs `padze` (import `padze`, console command `padze`) along with the
test and tooling dependencies. For a lighter install with just the test
dependencies, use `pip install -e ".[test]"`.

## Running the tests

Run the full suite from the repository root:

```bash
pytest
```

You should see **82 passing and 11 skipped**. The skips are optional
C++-parity and large-data gates that auto-skip cleanly when a C++ compiler + GSL
or external datasets are absent. This is expected and is not a failure — install
a compiler and GSL to opt into those checks.

## Project scope and guiding principle

PADZE is a NumPy port and extension of the ADZE allelic-rarefaction statistics.
**Correctness and numerical parity with ADZE 1.0 is the guiding principle for
any change to the core statistics.** New features are welcome, but they must not
alter established estimator math without a dedicated parity review.

## Adding tests and fixtures

Every change should come with tests:

- Prefer tiny, deterministic fixtures — see `tests/fixtures/` and `examples/`.
- For any new statistic, add a parity test against ADZE 1.0 or a
  reference-formula test that pins the expected values.

## Coding conventions

- Write NumPy-vectorized code; avoid Python-level loops over allele/sample data.
- Keep output deterministic: identical inputs and flags must always produce
  identical output.
- Keep the public API in `src/padze/__init__.py` stable. Additions are fine;
  breaking changes require a release note.

## Submitting changes

1. Create a branch for your work.
2. Ensure `pytest` is green (82 passed, 11 skipped).
3. Open a pull request with a clear description of what changed and why.
