"""Coverage for multi-population pihat combinations beyond the DNNaic 3-pop default."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from padze.rarefaction import absence_prob_matrix, locus_statistics  # noqa: E402


def test_five_population_pihat_key_count_and_all_population_formula():
    counts = np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [2, 2, 0],
        [1, 1, 2],
    ], dtype=np.int64)
    N = counts.sum(axis=1)
    depths = np.array([2], dtype=np.int64)

    stats = locus_statistics(counts, N, depths, pihat_sizes=(2, 3, 4, 5))
    pihat_keys = [k for k in stats if k.startswith("pihat_")]
    assert len(pihat_keys) == 26  # C(5,2)+C(5,3)+C(5,4)+C(5,5)
    assert "pihat_12345" in stats

    presence = []
    for j in range(counts.shape[0]):
        presence.append(1.0 - absence_prob_matrix(counts[j], int(N[j]), depths)[:, 0])
    expected = float(np.prod(np.vstack(presence), axis=0).sum())
    assert math.isclose(stats["pihat_12345"][0], expected, rel_tol=0, abs_tol=1e-12)


def test_population_order_changes_numbering_not_underlying_pihat_value():
    counts = np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [2, 2, 0],
        [1, 1, 2],
    ], dtype=np.int64)
    N = counts.sum(axis=1)
    depths = np.array([2], dtype=np.int64)

    base = locus_statistics(counts, N, depths, pihat_sizes=(2,))
    rev_counts = counts[::-1]
    rev = locus_statistics(rev_counts, rev_counts.sum(axis=1), depths, pihat_sizes=(2,))
    assert math.isclose(rev["pihat_12"][0], base["pihat_45"][0],
                        rel_tol=0, abs_tol=1e-12)
