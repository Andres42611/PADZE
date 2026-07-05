"""High-level PADZE feature pipeline: loci -> across-loci moment table.

This ties the rarefaction core (:mod:`padze.rarefaction`) to the moment estimators
(:mod:`padze.moments`) to reproduce and extend ADZE-style feature construction.

For each statistic ``T in {alpha_j, pi_j, pihat_C}`` and each rarefaction depth ``g``, the
per-locus values ``T_g(locus)`` are summarized across loci into the chosen moments. The
classic ADZE output keeps ``(mean, variance, se)``; PADZE additionally exposes
``(skewness, kurtosis)`` (the closed "3rd/4th moment gap").

``FeatureTable.to_frame`` emits one row per rarefaction depth, with columns ``g`` then
``<stat>_<moment>`` for every statistic/moment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .moments import MOMENT_FIELDS, MomentAccumulator, moments_matrix
from .rarefaction import locus_statistics

__all__ = [
    "FeatureTable",
    "compute_features",
    "ClassicalResult",
    "classical_features",
    "WindowResult",
    "rolling_window_features",
]

MISSING = -9.0


@dataclass
class FeatureTable:
    """Across-loci moment table for all statistics at all depths.

    ``per_locus`` (when retained) holds, per statistic, the full ``(L, G)`` matrix of
    per-locus values with the missing sentinel ``-9`` where a depth is unsupported. This is
    the PADZE equivalent of the C++ ``FULL_R``/``FULL_P``/``FULL_C`` outputs.
    """

    depths: np.ndarray                       # (G,)
    stat_keys: List[str]                     # ordered statistic names
    moments: List[str]                       # subset/order of MOMENT_FIELDS emitted
    # values[stat][moment] -> (G,) array
    values: Dict[str, Dict[str, np.ndarray]]
    n_loci: int
    populations: List[str]
    per_locus: Dict[str, np.ndarray] | None = None   # stat -> (L, G), the FULL_* equivalent
    locus_ids: List[str] | None = None

    def to_frame(self) -> Tuple[np.ndarray, List[str]]:
        """Return ``(matrix, columns)`` in a row-per-depth feature layout.

        ``matrix`` is ``(G, 1 + len(stat_keys)*len(moments))``: column 0 is ``g``, then the
        moments of each statistic in ``stat_keys`` order.
        """
        cols = ["g"]
        data = [self.depths.astype(np.float64)]
        for s in self.stat_keys:
            for m in self.moments:
                cols.append(f"{s}_{m}")
                data.append(self.values[s][m])
        return np.column_stack(data), cols

    def feature_dim(self) -> int:
        return 1 + len(self.stat_keys) * len(self.moments)

    def per_locus_frame(self) -> Tuple[np.ndarray, List[str]]:
        """Long-format per-locus values (the FULL_R/P/C equivalent).

        Returns ``(rows, columns)`` with columns ``[locus, statistic, g, value]``; one row
        per (locus, statistic, depth) whose value is not the missing sentinel. Requires the
        table to have been built with ``keep_per_locus=True``.
        """
        if self.per_locus is None:
            raise ValueError("per-locus values not retained; "
                             "call compute_features(..., keep_per_locus=True)")
        ids = self.locus_ids or [f"locus_{i}" for i in range(self.n_loci)]
        rows: List[list] = []
        for s in self.stat_keys:
            mat = self.per_locus[s]
            for li in range(mat.shape[0]):
                for gi, g in enumerate(self.depths):
                    v = mat[li, gi]
                    if v != MISSING and not np.isnan(v):
                        rows.append([ids[li], s, int(g), float(v)])
        return np.array(rows, dtype=object), ["locus", "statistic", "g", "value"]


def _validate_pihat_sizes(P: int, pihat_sizes: Sequence[int]) -> Tuple[int, ...]:
    sizes = tuple(int(k) for k in pihat_sizes)
    if len(sizes) != len(set(sizes)):
        raise ValueError("pihat_sizes contains duplicate values")
    for k in sizes:
        if k < 1 or k > P:
            raise ValueError(f"pihat size {k} is outside the valid range 1..{P}")
    return sizes


def _default_pihat_sizes(P: int) -> Tuple[int, ...]:
    return (1,) if P == 1 else (2,)


def _resolve_pihat_sizes(P: int, pihat_sizes: Sequence[int] | None) -> Tuple[int, ...]:
    if pihat_sizes is None:
        pihat_sizes = _default_pihat_sizes(P)
    return _validate_pihat_sizes(P, pihat_sizes)


def _ordered_stat_keys(P: int, pihat_sizes: Sequence[int]) -> List[str]:
    from itertools import combinations
    pihat_sizes = _validate_pihat_sizes(P, pihat_sizes)
    keys = [f"alpha_{j+1}" for j in range(P)]
    keys += [f"pi_{j+1}" for j in range(P)]
    for ksz in pihat_sizes:
        for combo in combinations(range(P), ksz):
            keys.append("pihat_" + "".join(str(c + 1) for c in combo))
    return keys


def compute_features(
    loci,
    *,
    depths: Sequence[int] | None = None,
    pihat_sizes: Sequence[int] | None = None,
    moments: Sequence[str] = MOMENT_FIELDS,
    bias_corrected: bool = True,
    keep_per_locus: bool = False,
    depth_policy: str = "common",
) -> FeatureTable:
    """Compute the full PADZE feature table from a :class:`LociData`.

    Parameters
    ----------
    loci : LociData
        From :func:`padze.io.read_vcf` / :func:`read_structure`, or constructed
        directly (``.count_matrices``, ``.sample_sizes``, ``.populations``).
    depths : sequence of int, optional
        Rarefaction depths ``g``. Default under ``depth_policy="common"``:
        ``2 .. loci.max_depth()`` inclusive.
    pihat_sizes : sequence of int, optional
        pihat combination sizes. The default is pairwise ``(2,)`` for two or more
        populations, and singleton ``(1,)`` for one population. For ``P = 1``, ``pi_1`` and
        ``pihat_1`` are mathematically defined by the current formulas and equal
        ``alpha_1`` because the products over other populations are empty.
    moments : sequence of str
        Subset/order of :data:`MOMENT_FIELDS` to emit.
    bias_corrected : bool
        Use sample (G1/G2) vs population (g1/g2) higher moments.
    depth_policy : {"common", "ragged"}
        ``"common"`` keeps a rectangular feature contract: every requested depth must be
        supported by every population at every locus. ``"ragged"`` allows depths up to the
        largest per-population sample size and lets unsupported per-locus values be omitted
        from each moment column. Use :func:`classical_features` for exact ADZE 1.0
        row-suppression parity.

    Returns
    -------
    FeatureTable
    """
    for m in moments:
        if m not in MOMENT_FIELDS:
            raise ValueError(f"unknown moment {m!r}; valid: {MOMENT_FIELDS}")
    populations = list(loci.populations)
    P = len(populations)
    if P < 1:
        raise ValueError("ADZE statistics require at least 1 population")
    if depth_policy not in ("common", "ragged"):
        raise ValueError("depth_policy must be 'common' or 'ragged'")
    pihat_sizes = _resolve_pihat_sizes(P, pihat_sizes)

    S = np.asarray(loci.sample_sizes, dtype=np.int64)
    max_depth = int(S.min()) if S.size else 0
    max_available_depth = int(S.max()) if S.size else 0
    depth_limit = max_depth if depth_policy == "common" else max_available_depth
    if depths is None:
        if depth_limit < 2:
            raise ValueError(
                f"max usable depth is {depth_limit}; need >= 2 "
                "(check sample sizes/missingness)")
        depths = np.arange(2, depth_limit + 1, dtype=np.int64)
    else:
        depths = np.asarray(depths, dtype=np.int64)
        if depths.size and depths.min() < 2:
            raise ValueError("rarefaction depths must be >= 2")
        if depths.size and depths.max() > depth_limit:
            raise ValueError(
                f"requested depth {int(depths.max())} exceeds max usable depth "
                f"{depth_limit} under depth_policy={depth_policy!r}")

    stat_keys = _ordered_stat_keys(P, pihat_sizes)
    G = depths.size
    L = len(loci.count_matrices)

    values: Dict[str, Dict[str, np.ndarray]] = {}
    per_locus = None

    if keep_per_locus:
        # Build, for each statistic, an (L, G) matrix of per-locus values.
        per_stat = {s: np.full((L, G), MISSING, dtype=np.float64) for s in stat_keys}
        for li, counts in enumerate(loci.count_matrices):
            N = loci.sample_sizes[li]
            stats = locus_statistics(counts, N, depths, pihat_sizes=pihat_sizes,
                                     missing_value=MISSING)
            for s in stat_keys:
                per_stat[s][li] = stats[s]

        for s in stat_keys:
            mm = moments_matrix(per_stat[s], missing=MISSING,
                                bias_corrected=bias_corrected)
            values[s] = {m: mm[m] for m in moments}
        per_locus = {s: per_stat[s] for s in stat_keys}
    else:
        accumulators = {
            s: [
                MomentAccumulator(bias_corrected=bias_corrected, missing=MISSING)
                for _ in range(G)
            ]
            for s in stat_keys
        }
        for li, counts in enumerate(loci.count_matrices):
            N = loci.sample_sizes[li]
            stats = locus_statistics(counts, N, depths, pihat_sizes=pihat_sizes,
                                     missing_value=MISSING)
            for s in stat_keys:
                for gi, v in enumerate(stats[s]):
                    accumulators[s][gi].update(v)

        for s in stat_keys:
            finalized = [acc.finalize() for acc in accumulators[s]]
            values[s] = {
                m: np.array([getattr(summary, m) for summary in finalized],
                            dtype=np.float64)
                for m in moments
            }

    return FeatureTable(
        depths=depths,
        stat_keys=stat_keys,
        moments=list(moments),
        values=values,
        n_loci=L,
        populations=populations,
        per_locus=per_locus,
        locus_ids=list(getattr(loci, "locus_ids", []) or []) or None,
    )


# ---------------------------------------------------------------------------------------
# Classical exact-match mode: reproduce the C++ ADZE output line for line.
# ---------------------------------------------------------------------------------------


@dataclass
class ClassicalResult:
    """The C++ ADZE output: ``(mean, variance, se)`` per statistic and rarefaction depth.

    ``summary[(stat, g)] = (mean, variance, se)`` and ``per_locus[(stat, g)]`` is the
    length-``L`` vector of per-locus values, for every ``(statistic, depth)`` the C++ ADZE
    prints. A statistic is present only over the depth range ADZE emits: ``alpha_j`` for
    ``g = 2 .. min_l N_j(l)`` (population ``j``'s own per-locus minimum), and ``pi_j`` /
    ``pihat_C`` for ``g = 2 .. min_{l, j'} N_{j'}(l)`` (the depth every population supports at
    every locus). Within that range every locus contributes; the variance is the
    Bessel-corrected sample variance ``sum (x - mean)^2 / (L - 1)`` and the standard error is
    ``sqrt(variance / L)``, matching ``ADZE_stats.cpp``.
    """

    populations: List[str]
    stat_keys: List[str]
    summary: Dict[Tuple[str, int], Tuple[float, float, float]]
    per_locus: Dict[Tuple[str, int], np.ndarray]
    n_loci: int
    depths: np.ndarray
    locus_ids: List[str] | None = None
    deleted_loci: List[str] | None = None
    missing_tolerance: float | None = None

    def rows(self):
        """Yield ``(statistic, g, n_loci, mean, variance, se)``, sorted by statistic then g."""
        order = {s: i for i, s in enumerate(self.stat_keys)}
        for (s, g) in sorted(self.summary, key=lambda k: (order[k[0]], k[1])):
            mean, var, se = self.summary[(s, g)]
            yield (s, g, self.n_loci, mean, var, se)


def classical_features(loci, *, max_g: int | None = None,
                       pihat_sizes: Sequence[int] | None = None,
                       keep_per_locus: bool = True) -> ClassicalResult:
    """Reproduce the C++ ADZE result exactly (the parity path).

    For each statistic and each rarefaction depth the C++ ADZE would print, compute the
    across-loci mean, Bessel-corrected variance, and standard error over *all* loci. A
    ``(statistic, depth)`` is emitted only when every locus supports that depth for that
    statistic, exactly as the C++ ``Stats`` routine suppresses a depth whose per-locus vector
    contains the missing sentinel. This is guaranteed identical to the vendored C++ ADZE.

    Parameters
    ----------
    loci : LociData-like
        Anything exposing ``populations``, ``count_matrices``, and ``sample_sizes``.
    max_g : int, optional
        Largest depth to consider (the C++ ``MAX_G``). Default: the largest per-locus
        per-population sample size present, so every depth the data can support is emitted.
    pihat_sizes : sequence of int, optional
        pihat combination sizes. The default is pairwise ``(2,)`` for two or more
        populations, and singleton ``(1,)`` for one population. For ``P = 1``, ``pi_1`` and
        ``pihat_1`` are mathematically defined by the current formulas and equal
        ``alpha_1`` because the products over other populations are empty.
    keep_per_locus : bool
        Retain the full per-locus vectors needed for ADZE FULL_R/FULL_P/FULL_C output.
        Set to ``False`` for summary-only workloads to avoid materializing per-statistic
        ``(L, G)`` matrices.
    """
    populations = list(loci.populations)
    P = len(populations)
    if P < 1:
        raise ValueError("ADZE statistics require at least 1 population")
    L = len(loci.count_matrices)
    pihat_sizes = _resolve_pihat_sizes(P, pihat_sizes)
    stat_keys = _ordered_stat_keys(P, pihat_sizes)
    summary: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    per_locus: Dict[Tuple[str, int], np.ndarray] = {}
    if L == 0:
        meta = getattr(loci, "metadata", None)
        return ClassicalResult(populations, stat_keys, summary, per_locus, 0,
                               np.zeros(0, dtype=np.int64),
                               list(getattr(loci, "locus_ids", []) or []) or None,
                               list(getattr(meta, "deleted_loci", []) or []) or None,
                               getattr(meta, "missing_tolerance", None))
    S = np.asarray(loci.sample_sizes, dtype=np.int64)
    hi = int(S.max()) if max_g is None else int(max_g)
    if hi < 2:
        meta = getattr(loci, "metadata", None)
        return ClassicalResult(populations, stat_keys, summary, per_locus, L,
                               np.zeros(0, dtype=np.int64),
                               list(getattr(loci, "locus_ids", []) or []) or None,
                               list(getattr(meta, "deleted_loci", []) or []) or None,
                               getattr(meta, "missing_tolerance", None))
    depths = np.arange(2, hi + 1, dtype=np.int64)

    if keep_per_locus:
        per_stat = {s: np.full((L, depths.size), MISSING, dtype=np.float64)
                    for s in stat_keys}
        for li, counts in enumerate(loci.count_matrices):
            st = locus_statistics(counts, S[li], depths, pihat_sizes=pihat_sizes,
                                  missing_value=MISSING)
            for s in stat_keys:
                per_stat[s][li] = st[s]

        for s in stat_keys:
            M = per_stat[s]
            for gi in range(depths.size):
                col = M[:, gi]
                if np.any(col == MISSING):
                    continue  # C++ suppresses a depth if any locus is unsupported
                mean = float(col.mean())
                if L >= 2:
                    var = float(((col - mean) ** 2).sum() / (L - 1))
                    se = float(np.sqrt(var / L))
                else:
                    var = float("nan")
                    se = float("nan")
                g = int(depths[gi])
                summary[(s, g)] = (mean, var, se)
                per_locus[(s, g)] = col.copy()
    else:
        accumulators = {
            s: [MomentAccumulator(missing=MISSING) for _ in range(depths.size)]
            for s in stat_keys
        }
        for li, locus_counts in enumerate(loci.count_matrices):
            st = locus_statistics(locus_counts, S[li], depths, pihat_sizes=pihat_sizes,
                                  missing_value=MISSING)
            for s in stat_keys:
                for gi, v in enumerate(st[s]):
                    accumulators[s][gi].update(v)

        for s in stat_keys:
            for gi, acc in enumerate(accumulators[s]):
                if acc.n != L:
                    continue  # C++ suppresses a depth if any locus is unsupported
                result = acc.finalize()
                summary[(s, int(depths[gi]))] = (
                    result.mean, result.variance, result.se)

    meta = getattr(loci, "metadata", None)
    return ClassicalResult(
        populations, stat_keys, summary, per_locus, L, depths,
        list(getattr(loci, "locus_ids", []) or []) or None,
        list(getattr(meta, "deleted_loci", []) or []) or None,
        getattr(meta, "missing_tolerance", None),
    )


# ---------------------------------------------------------------------------------------
# Rolling-window mode: the same rarefaction statistics over sliding genomic windows.
# ---------------------------------------------------------------------------------------


@dataclass
class WindowResult:
    """One genomic window and its classical ADZE result.

    ``result`` is a :class:`ClassicalResult` computed over exactly the loci in the window, so
    a single window spanning all loci (``window`` = number of loci, ``step`` = same) is
    identical to :func:`classical_features` over the whole dataset.
    """

    index: int
    unit: str
    start: float
    end: float
    locus_indices: List[int]
    n_loci: int
    result: ClassicalResult


class _LociView:
    """Minimal loci view over a subset of locus indices (no copy of the count arrays)."""

    def __init__(self, populations, count_matrices, sample_sizes, locus_ids):
        self.populations = populations
        self.count_matrices = count_matrices
        self.sample_sizes = sample_sizes
        self.locus_ids = locus_ids


def _subset(loci, idx: List[int]) -> _LociView:
    cms = [loci.count_matrices[i] for i in idx]
    S = np.asarray(loci.sample_sizes, dtype=np.int64)
    ss = S[idx] if idx else S[:0]
    ids = None
    all_ids = getattr(loci, "locus_ids", None)
    if all_ids:
        ids = [all_ids[i] for i in idx]
    return _LociView(list(loci.populations), cms, ss, ids)


def _parse_positions(loci, positions):
    """Return ``(chroms, pos)`` int arrays from an explicit ``positions`` or ``chrom:pos`` ids."""
    L = len(loci.count_matrices)
    if positions is not None:
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size != L:
            raise ValueError(f"positions has {pos.size} entries but there are {L} loci")
        return ["."] * L, pos
    ids = getattr(loci, "locus_ids", None)
    if not ids or len(ids) != L:
        raise ValueError("bp windows need per-locus positions: pass positions=... or use "
                         "loci whose ids are 'chrom:pos' (e.g. from read_vcf)")
    chroms: List[str] = []
    pos = np.empty(L, dtype=np.int64)
    for i, lid in enumerate(ids):
        if ":" not in lid:
            raise ValueError(f"locus id {lid!r} is not 'chrom:pos'; pass positions=... for bp")
        c, p = lid.rsplit(":", 1)
        chroms.append(c)
        pos[i] = int(p)
    return chroms, pos


def rolling_window_features(loci, *, window: float, step: float, unit: str = "loci",
                            positions=None, max_g: int | None = None,
                            pihat_sizes: Sequence[int] | None = None) -> List[WindowResult]:
    """Compute the classical ADZE statistics over sliding windows of the genome.

    Parameters
    ----------
    window, step : number
        Window size and step. In ``unit='loci'`` these are counts of loci (integers); in
        ``unit='bp'`` they are base-pair spans.
    unit : {'loci', 'bp'}
        Window along locus index (``'loci'``) or genomic coordinate (``'bp'``). For ``'bp'``
        each chromosome is windowed independently.
    positions : array-like, optional
        Per-locus integer positions for ``unit='bp'`` (overrides ids). If omitted, positions
        are parsed from ``chrom:pos`` locus ids (as produced by :func:`read_vcf`).
    max_g, pihat_sizes
        Passed through to :func:`classical_features` for each window.

    Returns
    -------
    list of WindowResult
        One per window that contains at least one locus, in genomic order.

    Notes
    -----
    A single window covering every locus reproduces :func:`classical_features` exactly: this
    is the ``window = all-loci, step = all`` consistency check.
    """
    L = len(loci.count_matrices)
    out: List[WindowResult] = []
    if unit == "loci":
        w = int(window)
        s = int(step)
        if w <= 0 or s <= 0:
            raise ValueError("window and step must be positive integers")
        wi = 0
        for start in range(0, L, s):
            idx = list(range(start, min(start + w, L)))
            if not idx:
                continue
            res = classical_features(_subset(loci, idx), max_g=max_g, pihat_sizes=pihat_sizes)
            out.append(WindowResult(wi, "loci", float(start), float(start + len(idx)),
                                    idx, len(idx), res))
            wi += 1
        return out
    if unit == "bp":
        if window <= 0 or step <= 0:
            raise ValueError("window and step must be positive")
        chroms, pos = _parse_positions(loci, positions)
        w = float(window)
        s = float(step)
        wi = 0
        seen_chroms: List[str] = []
        for c in chroms:
            if c not in seen_chroms:
                seen_chroms.append(c)
        for c in seen_chroms:
            on_c = [i for i in range(L) if chroms[i] == c]
            if not on_c:
                continue
            cpos = pos[on_c]
            lo = int(cpos.min())
            hi = int(cpos.max())
            wstart = float(lo)
            while wstart <= hi:
                wend = wstart + w
                idx = [i for i in on_c if wstart <= pos[i] < wend]
                if idx:
                    res = classical_features(_subset(loci, idx), max_g=max_g,
                                             pihat_sizes=pihat_sizes)
                    out.append(WindowResult(wi, "bp", wstart, wend, idx, len(idx), res))
                    wi += 1
                wstart += s
        return out
    raise ValueError("unit must be 'loci' or 'bp'")
