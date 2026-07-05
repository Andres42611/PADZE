"""Tests for classical exact-match mode and rolling-window mode.

Classical mode is checked against a naive Python transcription of the C++ ADZE emission
rules (the same reference style as ``tests/test_adze_rarefaction.py``): a ``(statistic,
depth)`` is emitted only when every locus supports it, and the across-loci summary is the
mean, the Bessel-corrected variance, and ``se = sqrt(var / L)`` over all loci. Rolling-window
mode is checked for the ``window = all loci`` consistency identity, a hand-computed small
example, and base-pair windows.

Runnable standalone or via pytest.
"""
import math
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from padze import classical_features, compute_features, rolling_window_features  # noqa: E402
from padze.features import _LociView  # noqa: E402


# ---- naive reference transcribing the C++ emission + summary rules -----------------------

def _ref_Q(count, N, g):
    Q = 1.0
    for u in range(g):
        Q *= (N - count - u) / (N - u)
    return max(Q, 0.0)


def _alpha(counts_j, Nj, g):
    return sum(1.0 - _ref_Q(int(counts_j[i]), Nj, g) for i in range(len(counts_j)))


def _pi(cm, N, j, g):
    P, A = cm.shape
    s = 0.0
    for i in range(A):
        prod = 1.0
        for jp in range(P):
            if jp != j:
                prod *= _ref_Q(int(cm[jp][i]), int(N[jp]), g)
        s += (1.0 - _ref_Q(int(cm[j][i]), int(N[j]), g)) * prod
    return s


def _pihat(cm, N, combo, g):
    P, A = cm.shape
    s = 0.0
    for i in range(A):
        pin = 1.0
        for j in combo:
            pin *= (1.0 - _ref_Q(int(cm[j][i]), int(N[j]), g))
        qout = 1.0
        for jp in range(P):
            if jp not in combo:
                qout *= _ref_Q(int(cm[jp][i]), int(N[jp]), g)
        s += pin * qout
    return s


def _summ(vals):
    L = len(vals)
    mean = sum(vals) / L
    if L >= 2:
        var = sum((v - mean) ** 2 for v in vals) / (L - 1)
        se = math.sqrt(var / L)
    else:
        var = float("nan")
        se = float("nan")
    return mean, var, se


def _ref_classical(populations, cms, ss, max_g, pihat_sizes=(2,)):
    P = len(populations)
    L = len(cms)
    summary = {}
    for g in range(2, max_g + 1):
        for j in range(P):                                    # alpha_j: own N_j only
            vals, ok = [], True
            for l in range(L):
                Nj = int(ss[l][j])
                if g > Nj:
                    ok = False
                    break
                vals.append(_alpha(cms[l][j], Nj, g))
            if ok:
                summary[(f"alpha_{j + 1}", g)] = _summ(vals)
        for j in range(P):                                    # pi_j: every population
            vals, ok = [], True
            for l in range(L):
                if any(g > int(ss[l][p]) for p in range(P)):
                    ok = False
                    break
                vals.append(_pi(cms[l], ss[l], j, g))
            if ok:
                summary[(f"pi_{j + 1}", g)] = _summ(vals)
        for ksz in pihat_sizes:                               # pihat_C: every population
            for combo in combinations(range(P), ksz):
                vals, ok = [], True
                for l in range(L):
                    if any(g > int(ss[l][p]) for p in range(P)):
                        ok = False
                        break
                    vals.append(_pihat(cms[l], ss[l], combo, g))
                if ok:
                    label = "pihat_" + "".join(str(c + 1) for c in combo)
                    summary[(label, g)] = _summ(vals)
    return summary


def _random_loci(rng):
    P = int(rng.integers(2, 6))
    L = int(rng.integers(3, 12))
    populations = [f"p{i}" for i in range(P)]
    cms, ss = [], []
    for _ in range(L):
        A = int(rng.integers(2, 5))
        cm = np.zeros((P, A), dtype=np.int64)
        for j in range(P):
            n = int(rng.integers(2, 9))            # vary sample size per (pop, locus), >= 2
            cm[j] = rng.multinomial(n, np.ones(A) / A)
        cms.append(cm)
        ss.append(cm.sum(axis=1))
    return populations, cms, np.array(ss)


def _one_population_loci():
    cms = [
        np.array([[1, 1]], dtype=np.int64),
        np.array([[2, 0]], dtype=np.int64),
    ]
    ss = np.array([[2], [2]], dtype=np.int64)
    return _LociView(["p0"], cms, ss, None)


def _two_population_loci():
    cms = [
        np.array([[1, 1], [2, 0]], dtype=np.int64),
        np.array([[2, 0], [1, 1]], dtype=np.int64),
    ]
    ss = np.array([[2, 2], [2, 2]], dtype=np.int64)
    return _LociView(["p0", "p1"], cms, ss, None)


# ---- tests -------------------------------------------------------------------------------

def test_one_population_features_use_singleton_pihat_default():
    loci = _one_population_loci()

    table = compute_features(loci, depths=[2], moments=("mean",), keep_per_locus=True)

    assert table.stat_keys == ["alpha_1", "pi_1", "pihat_1"]
    np.testing.assert_allclose(table.per_locus["alpha_1"], [[2.0], [1.0]])
    np.testing.assert_allclose(table.per_locus["pi_1"], table.per_locus["alpha_1"])
    np.testing.assert_allclose(table.per_locus["pihat_1"], table.per_locus["alpha_1"])
    assert math.isclose(table.values["alpha_1"]["mean"][0], 1.5)
    np.testing.assert_allclose(table.values["pi_1"]["mean"],
                               table.values["alpha_1"]["mean"])
    np.testing.assert_allclose(table.values["pihat_1"]["mean"],
                               table.values["alpha_1"]["mean"])

    res = classical_features(loci, max_g=2)

    assert res.stat_keys == ["alpha_1", "pi_1", "pihat_1"]
    assert set(res.summary) == {("alpha_1", 2), ("pi_1", 2), ("pihat_1", 2)}
    for key in res.summary:
        mean, var, se = res.summary[key]
        assert math.isclose(mean, 1.5)
        assert math.isclose(var, 0.5)
        assert math.isclose(se, 0.5)


def test_one_population_rolling_uses_singleton_pihat_default():
    loci = _one_population_loci()

    windows = rolling_window_features(loci, window=2, step=2, unit="loci")

    assert len(windows) == 1
    res = windows[0].result
    assert res.stat_keys == ["alpha_1", "pi_1", "pihat_1"]
    assert set(res.summary) == {("alpha_1", 2), ("pi_1", 2), ("pihat_1", 2)}


def test_one_population_explicit_pairwise_pihat_is_rejected():
    loci = _one_population_loci()
    calls = (
        lambda: compute_features(loci, depths=[2], pihat_sizes=(2,)),
        lambda: classical_features(loci, max_g=2, pihat_sizes=(2,)),
    )
    for call in calls:
        try:
            call()
            assert False, "expected ValueError"
        except ValueError as e:
            assert "valid range 1..1" in str(e)


def test_two_population_default_pihat_remains_pairwise():
    loci = _two_population_loci()

    default_table = compute_features(loci, depths=[2], moments=("mean",))
    explicit_table = compute_features(loci, depths=[2], pihat_sizes=(2,),
                                      moments=("mean",))

    assert default_table.stat_keys == ["alpha_1", "alpha_2", "pi_1", "pi_2", "pihat_12"]
    assert default_table.stat_keys == explicit_table.stat_keys
    for stat in default_table.stat_keys:
        np.testing.assert_allclose(default_table.values[stat]["mean"],
                                   explicit_table.values[stat]["mean"])

    default_res = classical_features(loci, max_g=2)
    explicit_res = classical_features(loci, max_g=2, pihat_sizes=(2,))

    assert default_res.stat_keys == ["alpha_1", "alpha_2", "pi_1", "pi_2", "pihat_12"]
    assert default_res.stat_keys == explicit_res.stat_keys
    assert set(default_res.summary) == set(explicit_res.summary)
    for key, expected in explicit_res.summary.items():
        got = default_res.summary[key]
        for a, b in zip(got, expected):
            assert math.isclose(a, b)


def test_classical_matches_reference():
    """Classical mode reproduces the naive C++-transcribed reference, incl. depth ranges."""
    rng = np.random.default_rng(7)
    max_abs = 0.0
    for _ in range(200):
        populations, cms, ss = _random_loci(rng)
        P = len(populations)
        sizes = tuple(range(2, P)) if P >= 3 else (2,)
        maxg = int(ss.max())
        loci = _LociView(populations, cms, ss, None)
        res = classical_features(loci, max_g=maxg, pihat_sizes=sizes)
        ref = _ref_classical(populations, cms, ss, maxg, pihat_sizes=sizes)
        # Structural: exactly the same (statistic, depth) rows are emitted.
        assert set(res.summary) == set(ref), (
            "emitted-key mismatch",
            sorted(set(res.summary) ^ set(ref))[:5])
        for k, ref_vals in ref.items():
            got = res.summary[k]
            for a, b in zip(got, ref_vals):
                if math.isnan(b):
                    assert math.isnan(a)
                else:
                    max_abs = max(max_abs, abs(a - b))
                    assert abs(a - b) < 1e-9, (k, a, b)
    assert max_abs < 1e-9


def test_classical_summary_only_matches_full_path():
    """Summary-only mode preserves ADZE rows while skipping FULL_* per-locus vectors."""
    rng = np.random.default_rng(17)
    for _ in range(100):
        populations, cms, ss = _random_loci(rng)
        P = len(populations)
        sizes = tuple(range(2, P)) if P >= 3 else (2,)
        maxg = int(ss.max())
        loci = _LociView(populations, cms, ss, None)
        full = classical_features(loci, max_g=maxg, pihat_sizes=sizes)
        summary_only = classical_features(
            loci, max_g=maxg, pihat_sizes=sizes, keep_per_locus=False)

        assert summary_only.per_locus == {}
        assert summary_only.stat_keys == full.stat_keys
        assert summary_only.n_loci == full.n_loci
        assert np.array_equal(summary_only.depths, full.depths)
        assert set(summary_only.summary) == set(full.summary)
        for key, expected in full.summary.items():
            got = summary_only.summary[key]
            for a, b in zip(got, expected):
                if math.isnan(b):
                    assert math.isnan(a)
                else:
                    assert abs(a - b) < 1e-10, (key, a, b)


def test_compute_features_depth_policy_common_vs_ragged():
    cms = [
        np.array([[2, 1], [3, 2]], dtype=np.int64),
        np.array([[2, 1], [4, 1]], dtype=np.int64),
    ]
    ss = np.array([[3, 5], [3, 5]], dtype=np.int64)
    loci = _LociView(["p0", "p1"], cms, ss, None)
    try:
        compute_features(loci, depths=[2, 3, 4], moments=("mean",))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "depth_policy='common'" in str(e)

    table = compute_features(loci, depths=[2, 3, 4], moments=("mean",),
                             depth_policy="ragged", keep_per_locus=True)
    assert table.per_locus["alpha_1"][0, 2] == -9.0
    assert table.per_locus["alpha_2"][0, 2] != -9.0
    assert table.per_locus["pi_1"][0, 2] == -9.0


def test_feature_apis_reject_duplicate_pihat_sizes():
    cms = [
        np.array([[2, 1], [3, 0]], dtype=np.int64),
        np.array([[1, 2], [1, 2]], dtype=np.int64),
    ]
    ss = np.array([[3, 3], [3, 3]], dtype=np.int64)
    loci = _LociView(["p0", "p1"], cms, ss, None)
    try:
        compute_features(loci, depths=[2], pihat_sizes=(2, 2))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "duplicate" in str(e)
    try:
        classical_features(loci, max_g=2, pihat_sizes=(2, 2))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "duplicate" in str(e)


def test_rolling_all_equals_classical():
    """A single window over all loci is identical to classical over the whole dataset."""
    rng = np.random.default_rng(11)
    for _ in range(30):
        populations, cms, ss = _random_loci(rng)
        P = len(populations)
        sizes = tuple(range(2, P)) if P >= 3 else (2,)
        L = len(cms)
        loci = _LociView(populations, cms, ss, None)
        res = classical_features(loci, pihat_sizes=sizes)
        w = rolling_window_features(loci, window=L, step=L, unit="loci", pihat_sizes=sizes)
        assert len(w) == 1
        assert set(w[0].result.summary) == set(res.summary)
        for k in res.summary:
            for a, b in zip(w[0].result.summary[k], res.summary[k]):
                if math.isnan(b):
                    assert math.isnan(a)
                else:
                    assert abs(a - b) < 1e-12


def test_rolling_hand_example():
    """Hand-computed: two non-overlapping windows of two loci each.

    Every population has two haploid gene copies at every locus, so only g=2 is defined.
    Locus a: pop0 = {A,B} (one each) -> alpha=2;   pop1 = {A,A} -> alpha=1.
    Locus b: pop0 = {A,A} -> alpha=1;   pop1 = {A,B} -> alpha=2.
    So in each window pop0's allelic richness is {2, 1}: mean 1.5, Bessel var 0.5,
    se = sqrt(0.5/2) = 0.5.
    """
    poly = np.array([1, 1])   # {A,B}: alpha(g=2) = 2
    mono = np.array([2, 0])   # {A,A}: alpha(g=2) = 1
    cms = [
        np.vstack([poly, mono]),   # locus 0
        np.vstack([mono, poly]),   # locus 1
        np.vstack([poly, mono]),   # locus 2
        np.vstack([mono, poly]),   # locus 3
    ]
    ss = np.array([[2, 2]] * 4)
    loci = _LociView(["p0", "p1"], cms, ss, None)
    w = rolling_window_features(loci, window=2, step=2, unit="loci")
    assert len(w) == 2
    assert [x.n_loci for x in w] == [2, 2]
    for win in w:
        mean, var, se = win.result.summary[("alpha_1", 2)]
        assert math.isclose(mean, 1.5, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(var, 0.5, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(se, 0.5, rel_tol=0, abs_tol=1e-12)


def test_rolling_bp_windows():
    """Base-pair windows group loci by coordinate; a locus falls in [start, start+window)."""
    poly = np.array([1, 1])
    mono = np.array([2, 0])
    cms = [np.vstack([poly, mono]), np.vstack([mono, poly]),
           np.vstack([poly, mono]), np.vstack([mono, poly])]
    ss = np.array([[2, 2]] * 4)
    # positions 100, 200, 1000, 1100 on one chromosome
    ids = ["chr1:100", "chr1:200", "chr1:1000", "chr1:1100"]
    loci = _LociView(["p0", "p1"], cms, ss, ids)
    w = rolling_window_features(loci, window=300, step=300, unit="bp")
    # window [100,400) has loci 0,1; [400,700) empty (skipped); [1000,1300) has loci 2,3
    counts = sorted(x.n_loci for x in w)
    assert counts == [2, 2], counts
    # explicit positions override also works
    w2 = rolling_window_features(loci, window=300, step=300, unit="bp",
                                 positions=[100, 200, 1000, 1100])
    assert sorted(x.n_loci for x in w2) == [2, 2]


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} classical/rolling tests passed")


if __name__ == "__main__":
    _main()
