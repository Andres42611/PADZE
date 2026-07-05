"""Tests for padze.moments: hand-checked values, scipy parity, edge cases.

Runnable two ways:
    /Users/ard/Desktop/genenv/bin/python tests/test_adze_moments.py      # standalone
    pytest tests/test_adze_moments.py                                    # if pytest present
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from padze.moments import (  # noqa: E402
    MomentAccumulator,
    moments_from_values,
    moments_matrix,
)

try:
    import scipy.stats as ss
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False


def test_hand_symmetric():
    # x = 1..5: mean 3, Bessel var 2.5, se sqrt(0.5), skew 0, excess kurt G2 = -1.2
    m = moments_from_values([1, 2, 3, 4, 5])
    assert math.isclose(m.mean, 3.0)
    assert math.isclose(m.variance, 2.5)
    assert math.isclose(m.se, math.sqrt(0.5))
    assert math.isclose(m.skewness, 0.0, abs_tol=1e-12)
    assert math.isclose(m.kurtosis, -1.2, rel_tol=1e-12)
    assert m.n == 5


def test_hand_skewed():
    # x = [0,0,0,0,10]: var 20, se 2, G1 = 2.2360679..., G2 = 5.0 (worked by hand)
    m = moments_from_values([0, 0, 0, 0, 10])
    assert math.isclose(m.variance, 20.0)
    assert math.isclose(m.se, 2.0)
    assert math.isclose(m.skewness, math.sqrt(5.0), rel_tol=1e-12)
    assert math.isclose(m.kurtosis, 5.0, rel_tol=1e-12)


def test_matches_scipy_random():
    if not HAVE_SCIPY:
        return
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(4, 40))
        x = rng.normal(size=n) * rng.uniform(0.1, 10) + rng.uniform(-5, 5)
        m = moments_from_values(x)
        assert math.isclose(m.skewness, float(ss.skew(x, bias=False)), rel_tol=1e-9,
                            abs_tol=1e-9)
        assert math.isclose(m.kurtosis, float(ss.kurtosis(x, bias=False)), rel_tol=1e-9,
                            abs_tol=1e-9)
        # mean/var/se against numpy
        assert math.isclose(m.mean, float(np.mean(x)), rel_tol=1e-12)
        assert math.isclose(m.variance, float(np.var(x, ddof=1)), rel_tol=1e-9)


def test_population_vs_sample():
    if not HAVE_SCIPY:
        return
    x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    mp = moments_from_values(x, bias_corrected=False)
    assert math.isclose(mp.skewness, float(ss.skew(x, bias=True)), rel_tol=1e-9)
    assert math.isclose(mp.kurtosis, float(ss.kurtosis(x, bias=True)), rel_tol=1e-9)


def test_edge_cases():
    # empty
    m0 = moments_from_values([])
    assert math.isnan(m0.mean) and m0.n == 0
    # one value: mean defined, the rest nan
    m1 = moments_from_values([3.5])
    assert m1.mean == 3.5 and math.isnan(m1.variance) and math.isnan(m1.skewness)
    # two values: var/se defined, skew/kurt nan
    m2 = moments_from_values([1.0, 3.0])
    assert math.isclose(m2.variance, 2.0) and math.isnan(m2.skewness)
    # three values: skew defined, kurt nan
    m3 = moments_from_values([1.0, 2.0, 9.0])
    assert not math.isnan(m3.skewness) and math.isnan(m3.kurtosis)
    # constant: skew/kurt defined as 0
    mc = moments_from_values([4.0, 4.0, 4.0, 4.0, 4.0])
    assert math.isclose(mc.variance, 0.0) and mc.skewness == 0.0 and mc.kurtosis == 0.0


def test_missing_filtering():
    # -9 sentinel and NaN both dropped, must equal clean computation
    clean = moments_from_values([1, 2, 3, 4, 5])
    withmiss = moments_from_values([1, 2, -9, 3, np.nan, 4, 5], missing=-9)
    assert math.isclose(clean.skewness, withmiss.skewness, abs_tol=1e-12)
    assert math.isclose(clean.kurtosis, withmiss.kurtosis, rel_tol=1e-12)
    assert withmiss.n == 5


def test_streaming_equals_batch():
    rng = np.random.default_rng(7)
    for _ in range(20):
        x = rng.normal(size=int(rng.integers(5, 60)))
        batch = moments_from_values(x)
        acc = MomentAccumulator().update_many(x)
        s = acc.finalize()
        assert math.isclose(batch.mean, s.mean, rel_tol=1e-12)
        assert math.isclose(batch.variance, s.variance, rel_tol=1e-10)
        assert math.isclose(batch.skewness, s.skewness, rel_tol=1e-8, abs_tol=1e-10)
        assert math.isclose(batch.kurtosis, s.kurtosis, rel_tol=1e-8, abs_tol=1e-10)


def test_matrix_equals_scalar():
    rng = np.random.default_rng(11)
    L, G = 30, 8
    X = rng.normal(size=(L, G)) * 3 + 1
    # punch holes with the -9 sentinel
    X[rng.integers(0, L, 10), rng.integers(0, G, 10)] = -9
    mm = moments_matrix(X, missing=-9)
    for g in range(G):
        col = moments_from_values(X[:, g], missing=-9)
        for f in ("mean", "variance", "se", "skewness", "kurtosis"):
            a, b = mm[f][g], getattr(col, f)
            if math.isnan(a) or math.isnan(b):
                assert math.isnan(a) and math.isnan(b)
            else:
                assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} moment tests passed"
          + ("" if HAVE_SCIPY else "  (scipy parity skipped: scipy unavailable)"))


if __name__ == "__main__":
    _main()
