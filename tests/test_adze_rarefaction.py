"""Tests for padze.rarefaction.

Parity strategy: a deliberately naive reference (``_ref_*``) transcribes the C++ ADZE
formulas (``ADZE_pop.cpp::calcQjig/calcAg``, ``ADZE_main_tools.cpp::calcPg`` and the
``calcAllPgComb`` combination loop) with explicit Python ``for`` loops. The vectorized
implementation must match it on randomized loci. Plus hand-computed values and sanity
relations (alpha >= pi >= 0, alpha non-decreasing in g).

Runnable standalone or via pytest.
"""
import math
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from padze.rarefaction import absence_prob_matrix, locus_statistics  # noqa: E402


# ---- naive reference transcribing the C++ ------------------------------------------------

def _ref_Q(count, N, g):
    """C++ calcQjig: prod_{u=0}^{g-1} (N - count - u)/(N - u)."""
    Q = 1.0
    for u in range(g):
        Q *= (N - count - u) / (N - u)
    return max(Q, 0.0)


def _ref_locus(count_matrix, N, g, pihat_sizes=(2,)):
    P, A = count_matrix.shape
    Q = [[_ref_Q(count_matrix[j, i], N[j], g) for i in range(A)] for j in range(P)]
    out = {}
    for j in range(P):
        out[f"alpha_{j+1}"] = sum(1 - Q[j][i] for i in range(A))
    for j in range(P):
        s = 0.0
        for i in range(A):
            prod = 1.0
            for jp in range(P):
                if jp != j:
                    prod *= Q[jp][i]
            s += (1 - Q[j][i]) * prod
        out[f"pi_{j+1}"] = s
    for ksz in pihat_sizes:
        for combo in combinations(range(P), ksz):
            s = 0.0
            for i in range(A):
                pin = 1.0
                for j in combo:
                    pin *= (1 - Q[j][i])
                qout = 1.0
                for jp in range(P):
                    if jp not in combo:
                        qout *= Q[jp][i]
                s += pin * qout
            out["pihat_" + "".join(str(c + 1) for c in combo)] = s
    return out


# ---- tests -------------------------------------------------------------------------------

def test_hand_two_pop():
    counts = np.array([[2, 2], [4, 0]])
    N = [4, 4]
    st = locus_statistics(counts, N, np.array([2]), pihat_sizes=(2,))
    assert math.isclose(st["alpha_1"][0], 5 / 3, rel_tol=1e-12)
    assert math.isclose(st["alpha_2"][0], 1.0, rel_tol=1e-12)
    assert math.isclose(st["pi_1"][0], 5 / 6, rel_tol=1e-12)
    assert math.isclose(st["pi_2"][0], 1 / 6, rel_tol=1e-12)
    assert math.isclose(st["pihat_12"][0], 5 / 6, rel_tol=1e-12)


def test_one_population_default_pihat_is_singleton():
    counts = np.array([[2, 2]])
    N = [4]
    depths = np.array([2, 3, 4])

    st = locus_statistics(counts, N, depths)

    assert set(st) == {"alpha_1", "pi_1", "pihat_1"}
    np.testing.assert_allclose(st["pi_1"], st["alpha_1"])
    np.testing.assert_allclose(st["pihat_1"], st["alpha_1"])


def test_Q_boundaries():
    # allele absent (count 0) -> Q = 1; fixed allele (count N) -> Q = 0
    Q = absence_prob_matrix(np.array([0, 4]), 4, np.array([2, 3]))
    assert np.allclose(Q[0], 1.0)       # absent everywhere
    assert np.allclose(Q[1], 0.0)       # fixed -> always sampled
    # g cannot exceed N
    try:
        absence_prob_matrix(np.array([1]), 3, np.array([4]))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_depth_one_rejected_for_adze_parity():
    try:
        absence_prob_matrix(np.array([1]), 3, np.array([1]))
        assert False, "expected ValueError"
    except ValueError as e:
        assert ">= 2" in str(e)


def test_parity_random():
    rng = np.random.default_rng(2024)
    max_abs = 0.0
    for _ in range(300):
        P = int(rng.integers(2, 5))
        A = int(rng.integers(1, 6))
        N = [int(rng.integers(4, 12)) for _ in range(P)]
        # random allele counts per population summing to N_j
        count_matrix = np.zeros((P, A), dtype=np.int64)
        for j in range(P):
            parts = rng.multinomial(N[j], np.ones(A) / A)
            count_matrix[j] = parts
        min_N = min(N)
        if min_N < 2:
            continue
        g = int(rng.integers(2, min_N + 1))
        sizes = tuple(s for s in (2, 3) if s <= P - 1) or (P,)
        got = locus_statistics(count_matrix, N, np.array([g]), pihat_sizes=sizes)
        ref = _ref_locus(count_matrix, N, g, pihat_sizes=sizes)
        for k, v in ref.items():
            assert k in got, k
            diff = abs(got[k][0] - v)
            max_abs = max(max_abs, diff)
            assert diff < 1e-9, (k, got[k][0], v)
    assert max_abs < 1e-9


def test_sanity_relations():
    rng = np.random.default_rng(5)
    P, A = 3, 4
    N = [10, 10, 10]
    count_matrix = np.vstack([rng.multinomial(N[j], np.ones(A) / A) for j in range(P)])
    depths = np.arange(2, 11)
    st = locus_statistics(count_matrix, N, depths, pihat_sizes=(2,))
    for j in range(1, P + 1):
        a, p = st[f"alpha_{j}"], st[f"pi_{j}"]
        assert np.all(a >= p - 1e-12), "alpha >= pi must hold"
        assert np.all(p >= -1e-12), "pi >= 0"
        assert np.all(np.diff(a) >= -1e-9), "alpha non-decreasing in g"


def test_missing_depth_sentinel():
    # one population smaller than requested depth -> sentinel at unsupported depths
    counts = np.array([[2, 1], [3, 2]])
    N = [3, 5]
    depths = np.array([2, 3, 4, 5])
    st = locus_statistics(counts, N, depths, pihat_sizes=(2,), missing_value=-9.0)
    # alpha is gated by its own population size, matching C++ calcAg.
    assert st["alpha_1"][2] == -9.0 and st["alpha_1"][3] == -9.0
    assert st["alpha_2"][2] != -9.0 and st["alpha_2"][3] != -9.0
    # pi/pihat require every population, so depths above min(N) are missing.
    for key in ("pi_1", "pi_2", "pihat_12"):
        assert st[key][2] == -9.0 and st[key][3] == -9.0
        assert st[key][0] != -9.0 and st[key][1] != -9.0


def test_invalid_count_matrices_rejected():
    try:
        locus_statistics(np.array([[2, -1], [1, 1]]), [1, 2], np.array([2]))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "nonnegative" in str(e)

    try:
        locus_statistics(np.array([[2, 1], [1, 1]]), [3, 5], np.array([2]))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "row must sum" in str(e)


def test_invalid_pihat_sizes_rejected():
    counts = np.array([[1, 1], [1, 1]])
    for sizes in ((0,), (3,), (2, 2)):
        try:
            locus_statistics(counts, [2, 2], np.array([2]), pihat_sizes=sizes)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "pihat size" in str(e) or "duplicate" in str(e)


def test_min_depth_ok_false_rejects_impossible_depths():
    counts = np.array([[2, 1], [3, 2]])
    try:
        locus_statistics(counts, [3, 5], np.array([4]), min_depth_ok=False)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "min_depth_ok=False" in str(e)


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} rarefaction tests passed")


if __name__ == "__main__":
    _main()
