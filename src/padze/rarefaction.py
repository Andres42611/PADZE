"""Vectorized ADZE allelic-rarefaction statistics (alpha, pi, pihat).

Faithful NumPy port of the rarefaction core in the vendored C++ ADZE
(``ADZE_pop.cpp::calcQjig/calcAg`` and ``ADZE_main_tools.cpp::calcPg/calcAllPgComb``).

For population ``j`` at a locus, with ``N_j`` sampled gene copies and allele ``i`` observed
``N_ji`` times, the probability that allele ``i`` is **absent** from a rarefied subsample of
size ``g`` (drawn without replacement) is the hypergeometric tail

        Q_jig = C(N_j - N_ji, g) / C(N_j, g)
              = prod_{u=0}^{g-1} (N_j - N_ji - u) / (N_j - u)            (C++ product form)

and the **presence** probability is ``P_jig = 1 - Q_jig``. From these:

    allelic richness        alpha_j(g)   = sum_i (1 - Q_jig)
    private richness         pi_j(g)     = sum_i (1 - Q_jig) * prod_{j' != j} Q_j'ig
    combination-private      pihat_C(g)  = sum_i [ prod_{j in C} (1 - Q_jig) ]
                                                 * [ prod_{j' not in C} Q_j'ig ]

``alpha`` counts distinct alleles expected in a size-``g`` subsample; ``pi`` counts alleles
expected to be seen in population ``j`` *and in no other*; ``pihat`` for a population set
``C`` counts alleles expected in *every* population of ``C`` and *no* population outside it.
With three populations and ``|C| = 2`` these are pihat_12, pihat_13, pihat_23.

Vectorization preserves the C++ combinatorial content while removing its triple ``for``
loop: ``Q`` is built for all (allele, depth) pairs at once via a cumulative product, and
the allele sums collapse with ``einsum``/broadcasting. The cumulative-product form is exact
and self-correcting at the boundary ``g > N_j - N_ji`` (the numerator hits 0 there, so the
cumulative product is 0 onward -> an allele present in more than ``N_j - g`` copies is
certain to be sampled), so no special-casing is needed.

Only NumPy is required.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "absence_prob_matrix",
    "locus_statistics",
    "STAT_KINDS",
]

STAT_KINDS = ("alpha", "pi", "pihat")


def absence_prob_matrix(counts: np.ndarray, N: int, depths: np.ndarray) -> np.ndarray:
    """Q_jig for every allele and depth at one (population, locus).

    Parameters
    ----------
    counts : (A,) int array
        Allele counts ``N_ji`` for the ``A`` distinct alleles segregating at this locus
        (aligned across populations by the caller). Counts may be 0 for alleles absent
        from this population.
    N : int
        Total sampled gene copies ``N_j`` (non-missing) for this population at this locus.
    depths : (G,) int array
        Rarefaction depths ``g``. Must satisfy ``2 <= g <= N``.

    Returns
    -------
    Q : (A, G) float array
        ``Q[a, t] = Q_{j, a, depths[t]}``.
    """
    counts = np.asarray(counts, dtype=np.int64).ravel()
    depths = np.asarray(depths, dtype=np.int64).ravel()
    A = counts.size
    if np.any(counts < 0):
        raise ValueError("allele counts must be nonnegative")
    if int(counts.sum()) > int(N):
        raise ValueError("allele counts cannot sum to more than N")
    if depths.size == 0:
        return np.zeros((A, 0), dtype=np.float64)
    max_g = int(depths.max())
    if max_g > N:
        raise ValueError(f"depth {max_g} exceeds sample size N={N}")
    if depths.min() < 2:
        raise ValueError("rarefaction depths must be >= 2")

    # ratios[a, u] = (N - counts[a] - u) / (N - u), for u = 0 .. max_g-1.
    u = np.arange(max_g, dtype=np.float64)              # (U,)
    denom = (N - u)                                     # (U,)  >0 since max_g <= N
    numer = (N - counts[:, None].astype(np.float64) - u[None, :])  # (A, U)
    ratios = numer / denom[None, :]
    Qfull = np.cumprod(ratios, axis=1)                  # (A, U); Q after g=1,2,... terms
    # Q at depth g uses g terms -> column index g-1. Clip negatives (post-zero round-off).
    Q = Qfull[:, depths - 1]
    np.clip(Q, 0.0, 1.0, out=Q)
    return Q


def locus_statistics(
    count_matrix: np.ndarray,
    N: Sequence[int],
    depths: np.ndarray,
    *,
    pihat_sizes: Sequence[int] | None = None,
    min_depth_ok: bool = True,
    missing_value: float = -9.0,
) -> Dict[str, np.ndarray]:
    """All ADZE statistics for one locus, for every population / combination and depth.

    Parameters
    ----------
    count_matrix : (P, A) int array
        Allele counts aligned across the ``P`` populations: ``count_matrix[j, a] = N_{j a}``
        at this locus. Column ``a`` is the *same* allele in every population.
    N : (P,) sequence of int
        Per-population sampled gene copies ``N_j`` at this locus.
    depths : (G,) int array
        Rarefaction depths.
    pihat_sizes : sequence of int, optional
        Combination sizes for pihat. The default is pairwise ``(2,)`` for two or more
        populations, and singleton ``(1,)`` for one population.
    min_depth_ok : bool
        If True, apply ADZE's per-depth gating: ``alpha_j`` at depth ``g`` is emitted only
        when ``g <= N_j`` (population ``j``'s own sample size), while ``pi_j`` and ``pihat_C``
        require ``g <= N_{j'}`` for every population ``j'`` (min over populations). Depths
        that fail the relevant test get the missing sentinel.
    missing_value : float
        Sentinel emitted for depths that exceed a population's sample size.

    Returns
    -------
    dict mapping statistic key -> (G,) float array of the per-locus value at each depth.
        Keys: ``alpha_{j}``, ``pi_{j}`` for j in 1..P, and ``pihat_{j..}`` for each
        combination (1-based population labels), e.g. ``pihat_12``.
    """
    count_matrix = np.asarray(count_matrix, dtype=np.int64)
    if count_matrix.ndim != 2:
        raise ValueError("count_matrix must be a 2D array")
    P, A = count_matrix.shape
    N = np.asarray(N, dtype=np.int64).ravel()
    depths = np.asarray(depths, dtype=np.int64).ravel()
    G = depths.size
    if N.size != P:
        raise ValueError(f"N must have one entry per population ({P})")
    if np.any(count_matrix < 0):
        raise ValueError("allele counts must be nonnegative")
    if np.any(N < 0):
        raise ValueError("sample sizes N must be nonnegative")
    row_sums = count_matrix.sum(axis=1)
    if not np.array_equal(row_sums, N):
        raise ValueError("each count_matrix row must sum to its population sample size N")
    if G and depths.min() < 2:
        raise ValueError("rarefaction depths must be >= 2")
    if pihat_sizes is None:
        pihat_sizes = (1,) if P == 1 else (2,)
    pihat_sizes = tuple(int(k) for k in pihat_sizes)
    if len(pihat_sizes) != len(set(pihat_sizes)):
        raise ValueError("pihat_sizes contains duplicate values")
    for k in pihat_sizes:
        if k < 1 or k > P:
            raise ValueError(f"pihat size {k} is outside the valid range 1..{P}")

    out: Dict[str, np.ndarray] = {}
    min_N = int(N.min()) if P else 0

    # Validity of a depth g at this locus. alpha_j(g) is defined whenever g <= N_j for its
    # own population (C++ ``calcAg`` never consults other populations); pi_j and pihat_C need
    # g <= N_{j'} for *every* population, because they multiply an absence probability across
    # all populations (C++ ``calcPg`` returns the missing sentinel unless every population
    # supports depth g). Gating alpha on the per-population size, not the across-population
    # minimum, is what reproduces the C++ output when sample sizes differ across populations
    # (as they do under missing data).
    if min_depth_ok:
        valid_all = depths <= min_N
        valid_pop = [depths <= int(N[j]) for j in range(P)]
    else:
        if G and int(depths.max()) > min_N:
            raise ValueError("min_depth_ok=False requires all depths to be supported "
                             "by every population")
        valid_all = np.ones(G, dtype=bool)
        valid_pop = [np.ones(G, dtype=bool) for _ in range(P)]

    # Build Q (P, A, G). For depths a population cannot support we set Q to NaN then mask.
    Q = np.full((P, A, G), np.nan, dtype=np.float64)
    for j in range(P):
        ok_j = depths <= int(N[j])
        if ok_j.any():
            # (A, n_ok) assigned into the matching columns -> no advanced-index transpose.
            Q[j][:, ok_j] = absence_prob_matrix(count_matrix[j], int(N[j]), depths[ok_j])
    Pmat = 1.0 - Q                                       # presence probs (P, A, G)

    def _emit(key: str, per_depth: np.ndarray, mask: np.ndarray) -> None:
        vals = np.where(mask, per_depth, missing_value)
        out[key] = vals

    # alpha_j(g) = sum_a P_jag  (gated by population j's own sample size only)
    alpha = np.nansum(Pmat, axis=1)                      # (P, G)
    for j in range(P):
        _emit(f"alpha_{j + 1}", alpha[j], valid_pop[j])

    # pi_j(g) = sum_a P_jag * prod_{j'!=j} Q_j'ag
    # Product terms are reused heavily when many pihat combinations are requested. Cache
    # per-locus subset products instead of recomputing the same population products for
    # each combination.
    ones = np.ones((A, G), dtype=np.float64)
    p_cache: Dict[int, np.ndarray] = {0: ones}
    q_cache: Dict[int, np.ndarray] = {0: ones}

    def _subset_product(cache: Dict[int, np.ndarray], mats: np.ndarray,
                        mask: int) -> np.ndarray:
        if mask not in cache:
            bit = mask & -mask
            j = bit.bit_length() - 1
            cache[mask] = _subset_product(cache, mats, mask ^ bit) * mats[j]
        return cache[mask]

    full_mask = (1 << P) - 1
    for j in range(P):
        Qprod_others = _subset_product(q_cache, Q, full_mask ^ (1 << j))
        pi_j = np.nansum(Pmat[j] * Qprod_others, axis=0)  # (G,)
        _emit(f"pi_{j + 1}", pi_j, valid_all)

    # pihat for each combination C of each requested size.
    for ksz in pihat_sizes:
        for combo in combinations(range(P), ksz):
            inside_mask = 0
            for c in combo:
                inside_mask |= 1 << c
            P_in = _subset_product(p_cache, Pmat, inside_mask)
            Q_out = _subset_product(q_cache, Q, full_mask ^ inside_mask)
            ph = np.nansum(P_in * Q_out, axis=0)         # (G,)
            label = "".join(str(c + 1) for c in combo)
            _emit(f"pihat_{label}", ph, valid_all)

    return out
