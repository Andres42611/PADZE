"""Focused parity checks for streaming feature summaries."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from padze import features as features_module  # noqa: E402
from padze.features import _LociView  # noqa: E402


def _random_loci():
    rng = np.random.default_rng(20260704)
    populations = ["p0", "p1", "p2", "p3"]
    cms = []
    sample_sizes = []
    for _ in range(11):
        alleles = int(rng.integers(2, 6))
        cm = np.zeros((len(populations), alleles), dtype=np.int64)
        for j in range(len(populations)):
            n = int(rng.integers(2, 9))
            cm[j] = rng.multinomial(n, np.ones(alleles) / alleles)
        cms.append(cm)
        sample_sizes.append(cm.sum(axis=1))
    return _LociView(populations, cms, np.asarray(sample_sizes, dtype=np.int64), None)


def _assert_summary_close(got, expected):
    for a, b in zip(got, expected):
        if math.isnan(b):
            assert math.isnan(a)
        else:
            assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)


def test_compute_features_streaming_matches_retained_matrix_path(monkeypatch):
    loci = _random_loci()
    depths = np.arange(2, int(loci.sample_sizes.max()) + 1, dtype=np.int64)

    full = features_module.compute_features(
        loci,
        depths=depths,
        pihat_sizes=(2, 3),
        depth_policy="ragged",
        keep_per_locus=True,
    )

    def fail_moments_matrix(*args, **kwargs):
        raise AssertionError("streaming compute_features must not call moments_matrix")

    monkeypatch.setattr(features_module, "moments_matrix", fail_moments_matrix)
    streamed = features_module.compute_features(
        loci,
        depths=depths,
        pihat_sizes=(2, 3),
        depth_policy="ragged",
        keep_per_locus=False,
    )

    assert streamed.per_locus is None
    assert streamed.stat_keys == full.stat_keys
    assert streamed.moments == full.moments
    assert np.array_equal(streamed.depths, full.depths)
    for stat in full.stat_keys:
        for moment in full.moments:
            np.testing.assert_allclose(
                streamed.values[stat][moment],
                full.values[stat][moment],
                rtol=1e-8,
                atol=1e-10,
                equal_nan=True,
            )


def test_classical_summary_streaming_matches_full_for_single_locus_nan_variance():
    cm = np.array([[2, 1, 0], [0, 2, 1]], dtype=np.int64)
    loci = _LociView(["p0", "p1"], [cm], np.asarray([[3, 3]], dtype=np.int64), None)

    full = features_module.classical_features(loci, max_g=3, keep_per_locus=True)
    streamed = features_module.classical_features(loci, max_g=3, keep_per_locus=False)

    assert streamed.per_locus == {}
    assert set(streamed.summary) == set(full.summary)
    assert streamed.stat_keys == full.stat_keys
    assert streamed.n_loci == 1
    assert np.array_equal(streamed.depths, full.depths)
    for key, expected in full.summary.items():
        _assert_summary_close(streamed.summary[key], expected)
        assert math.isnan(streamed.summary[key][1])
        assert math.isnan(streamed.summary[key][2])
