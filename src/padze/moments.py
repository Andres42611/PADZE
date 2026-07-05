"""Across-loci moment estimators for ADZE rarefaction statistics.

The vendored C++ ADZE (``src/external/adze``) summarizes each rarefaction statistic
across loci with only the **mean**, the **Bessel-corrected sample variance**, and the
**standard error** ``sqrt(var / L)`` (see ``ADZE_stats.cpp::calcAvg/calcVar/calcStdErr``).
Those three moments are the ADZE-compatible summary surface.

This module closes the documented "3rd/4th moment gap": it adds numerically stable,
bias-corrected **skewness** and **excess kurtosis** alongside mean/variance/SE, with
explicit definitions, deterministic edge-case behavior, and a streaming accumulator so the
same numbers can be produced out-of-core (one locus at a time) without holding every value
in memory.

Definitions (data ``x_1..x_L`` are the per-locus values of one statistic at one
rarefaction depth ``g``, after dropping missing loci):

    mean           m1 = (1/L) * sum_l x_l
    central moment Mk = sum_l (x_l - m1)**k                       (k = 2, 3, 4)

    variance (sample, Bessel)  var = M2 / (L - 1)        # matches ADZE C++ exactly
    standard error             se  = sqrt(var / L)       # matches ADZE C++ exactly

    skewness (adjusted Fisher-Pearson, bias-corrected, "G1"):
        g1 = (M3 / L) / (M2 / L)**1.5
        G1 = g1 * sqrt(L * (L - 1)) / (L - 2)

    excess kurtosis (bias-corrected sample, "G2"; 0 in expectation for a Normal):
        g2 = (M4 / L) / (M2 / L)**2 - 3
        G2 = ((L - 1) / ((L - 2) * (L - 3))) * ((L + 1) * g2 + 6)

``G1``/``G2`` are the standard sample estimators used by Excel, SAS, and
``scipy.stats.skew/kurtosis(..., bias=False)``; they are validated against scipy in the
tests. Set ``bias_corrected=False`` to obtain the population (biased) forms ``g1``/``g2``.

Deterministic edge cases (documented and tested):

    L < 1                -> mean = nan
    L < 2                -> var = se = nan      (variance undefined for one locus)
    L < 3                -> skewness = nan      (G1 needs L >= 3)
    L < 4                -> kurtosis = nan      (G2 needs L >= 4)
    M2 == 0 (constant)   -> skewness = kurtosis = 0.0    (a degenerate point mass has no
                            asymmetry or excess tail; chosen for determinism over 0/0 = nan)

Only NumPy is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

__all__ = [
    "MomentSummary",
    "moments_from_values",
    "moments_matrix",
    "MomentAccumulator",
    "MOMENT_FIELDS",
]

# Order matters: this is the per-statistic feature layout PADZE emits. The first
# three reproduce the C++ ADZE output; the last two are the new higher moments.
MOMENT_FIELDS = ("mean", "variance", "se", "skewness", "kurtosis")


@dataclass(frozen=True)
class MomentSummary:
    """The five across-loci moments of one statistic at one rarefaction depth."""

    mean: float
    variance: float
    se: float
    skewness: float
    kurtosis: float
    n: int  # number of (non-missing) loci actually used

    def as_tuple(self) -> tuple:
        return (self.mean, self.variance, self.se, self.skewness, self.kurtosis)

    def as_dict(self) -> Dict[str, float]:
        return {f: getattr(self, f) for f in MOMENT_FIELDS}


def _finalize(n: int, m1: float, M2: float, M3: float, M4: float,
              bias_corrected: bool) -> MomentSummary:
    """Turn raw central-moment sums (M2,M3,M4) and count n into a MomentSummary.

    Shared by the vectorized and streaming paths so they are guaranteed identical.
    """
    nan = float("nan")
    if n < 1:
        return MomentSummary(nan, nan, nan, nan, nan, 0)
    if n < 2:
        return MomentSummary(float(m1), nan, nan, nan, nan, n)

    var = M2 / (n - 1)              # Bessel-corrected sample variance (ADZE C++)
    se = float(np.sqrt(var / n))   # SE = sqrt(var / L)              (ADZE C++)

    # Degenerate (constant) input: standardized moments are 0/0; define as 0 for
    # determinism. A point mass has no asymmetry and no excess tail.
    if M2 <= 0.0:
        skew = 0.0 if n >= 3 else nan
        kurt = 0.0 if n >= 4 else nan
        return MomentSummary(float(m1), float(var), se, skew, kurt, n)

    m2_pop = M2 / n                 # population (biased) variance, for standardization
    # Biased (population) standardized moments.
    g1 = (M3 / n) / m2_pop ** 1.5
    g2 = (M4 / n) / m2_pop ** 2 - 3.0

    if not bias_corrected:
        skew = float(g1) if n >= 3 else nan
        kurt = float(g2) if n >= 4 else nan
        return MomentSummary(float(m1), float(var), se, skew, kurt, n)

    # Bias-corrected sample estimators (G1, G2).
    if n >= 3:
        G1 = g1 * np.sqrt(n * (n - 1.0)) / (n - 2.0)
        skew = float(G1)
    else:
        skew = nan
    if n >= 4:
        G2 = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0)
        kurt = float(G2)
    else:
        kurt = nan
    return MomentSummary(float(m1), float(var), se, skew, kurt, n)


def moments_from_values(values, *, missing=None, bias_corrected: bool = True,
                        nan_is_missing: bool = True) -> MomentSummary:
    """Compute the five across-loci moments of one statistic (vectorized, two-pass).

    Parameters
    ----------
    values : array-like
        Per-locus values of a single statistic at a single rarefaction depth.
    missing : float or None
        Sentinel value to drop (e.g. ADZE's ``-9``). ``None`` disables sentinel filtering.
    bias_corrected : bool
        If True (default) return sample G1/G2; else population g1/g2.
    nan_is_missing : bool
        If True (default) NaN entries are dropped as missing.

    Returns
    -------
    MomentSummary
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if nan_is_missing:
        x = x[~np.isnan(x)]
    if missing is not None:
        x = x[x != missing]

    n = x.size
    if n == 0:
        return _finalize(0, 0.0, 0.0, 0.0, 0.0, bias_corrected)

    # Two-pass: subtract the mean before powering -> numerically stable central moments.
    m1 = float(x.mean())
    d = x - m1
    M2 = float(np.dot(d, d))
    if n < 2:
        return _finalize(n, m1, M2, 0.0, 0.0, bias_corrected)
    d2 = d * d
    M3 = float(np.dot(d2, d))
    M4 = float(np.dot(d2, d2))
    return _finalize(n, m1, M2, M3, M4, bias_corrected)


def moments_matrix(matrix, *, missing=None, bias_corrected: bool = True,
                   nan_is_missing: bool = True) -> Dict[str, np.ndarray]:
    """Vectorized moments for every column of an ``(L, G)`` matrix.

    Each column ``g`` holds the per-locus values of one statistic at one rarefaction depth;
    rows are loci. Missing entries (the ``missing`` sentinel and/or NaN) are dropped
    *per column*, so columns may use different numbers of loci. Returns a dict of length-``G``
    arrays for each field in :data:`MOMENT_FIELDS`, plus ``"n"`` (loci used per column).

    This is the efficient batch equivalent of calling :func:`moments_from_values` on each
    column; it is what the feature builder uses to summarize a whole genome at once.
    """
    X = np.asarray(matrix, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    _, G = X.shape

    valid = np.ones_like(X, dtype=bool)
    if nan_is_missing:
        valid &= ~np.isnan(X)
    if missing is not None:
        valid &= (X != missing)

    Xz = np.where(valid, X, 0.0)                 # zeroed where missing for masked sums
    n = valid.sum(axis=0).astype(np.float64)     # (G,)
    nan = float("nan")

    with np.errstate(invalid="ignore", divide="ignore"):
        s1 = Xz.sum(axis=0)
        mean = np.where(n >= 1, s1 / n, nan)

        d = np.where(valid, X - mean[None, :], 0.0)
        M2 = (d ** 2).sum(axis=0)
        M3 = (d ** 3).sum(axis=0)
        M4 = (d ** 4).sum(axis=0)

        var = np.where(n >= 2, M2 / (n - 1), nan)
        se = np.where(n >= 2, np.sqrt(var / n), nan)

        m2_pop = np.where(n >= 1, M2 / n, nan)
        nonconst = m2_pop > 0.0
        g1 = np.where(nonconst, (M3 / n) / np.where(nonconst, m2_pop, 1.0) ** 1.5, 0.0)
        g2 = np.where(nonconst, (M4 / n) / np.where(nonconst, m2_pop, 1.0) ** 2 - 3.0, 0.0)

        if bias_corrected:
            skew = np.where(n >= 3, g1 * np.sqrt(n * (n - 1.0)) / (n - 2.0), nan)
            kurt = np.where(
                n >= 4,
                ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0),
                nan,
            )
        else:
            skew = np.where(n >= 3, g1, nan)
            kurt = np.where(n >= 4, g2, nan)

        # Constant (zero-variance) columns: standardized moments defined as 0.
        const = (~nonconst) & (n >= 1)
        skew = np.where(const & (n >= 3), 0.0, skew)
        kurt = np.where(const & (n >= 4), 0.0, kurt)

    return {"mean": mean, "variance": var, "se": se,
            "skewness": skew, "kurtosis": kurt, "n": n.astype(np.int64)}


@dataclass
class MomentAccumulator:
    """Streaming (one-pass) accumulator for mean/var/se/skew/kurtosis.

    Uses Pebay's numerically stable online update for the central-moment sums M2, M3, M4,
    so a long genome can be summarized locus-by-locus without materializing every value.
    ``finalize()`` returns a MomentSummary numerically identical (to float tolerance) to
    :func:`moments_from_values`. This preserves the C++ pipeline's streaming-friendly
    nature while extending it to higher moments.
    """

    bias_corrected: bool = True
    missing: float | None = None
    nan_is_missing: bool = True
    n: int = 0
    _mean: float = 0.0
    _M2: float = 0.0
    _M3: float = 0.0
    _M4: float = 0.0

    def update(self, value: float) -> "MomentAccumulator":
        v = float(value)
        if self.nan_is_missing and np.isnan(v):
            return self
        if self.missing is not None and v == self.missing:
            return self
        n1 = self.n
        n = n1 + 1
        delta = v - self._mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1
        self._mean += delta_n
        # Order matters: update M4, then M3, then M2 (each uses the pre-update lower ones).
        self._M4 += (term1 * delta_n2 * (n * n - 3 * n + 3)
                     + 6 * delta_n2 * self._M2 - 4 * delta_n * self._M3)
        self._M3 += term1 * delta_n * (n - 2) - 3 * delta_n * self._M2
        self._M2 += term1
        self.n = n
        return self

    def update_many(self, values) -> "MomentAccumulator":
        for v in np.asarray(values, dtype=np.float64).ravel():
            self.update(v)
        return self

    def finalize(self) -> MomentSummary:
        return _finalize(self.n, self._mean, self._M2, self._M3, self._M4,
                         self.bias_corrected)
