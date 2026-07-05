"""PADZE: a robust, vectorized Python port of the ADZE allelic-rarefaction statistics.

Public API
----------
Rarefaction core:
    :func:`padze.rarefaction.locus_statistics`
    :func:`padze.rarefaction.absence_prob_matrix`
Across-loci moments (incl. the new 3rd/4th moments):
    :func:`padze.moments.moments_from_values`
    :func:`padze.moments.moments_matrix`
    :class:`padze.moments.MomentAccumulator`
Input:
    :func:`padze.io.read_vcf`, :func:`padze.io.read_structure`
High-level pipeline:
    :func:`padze.features.compute_features` -> :class:`padze.features.FeatureTable`

Example
-------
>>> from padze import read_vcf, compute_features
>>> loci = read_vcf("trio.vcf", {"S1": "A", "S2": "A", "S3": "B", "S4": "C"})
>>> table = compute_features(loci, depths=range(2, 11))
>>> mat, cols = table.to_frame()
"""
from __future__ import annotations

from .features import (
    ClassicalResult,
    FeatureTable,
    WindowResult,
    classical_features,
    compute_features,
    rolling_window_features,
)
from .io import LociData, Metadata, read_structure, read_vcf
from .moments import (
    MOMENT_FIELDS,
    MomentAccumulator,
    MomentSummary,
    moments_from_values,
    moments_matrix,
)
from .rarefaction import absence_prob_matrix, locus_statistics

__version__ = "0.1.0"

__all__ = [
    "FeatureTable",
    "compute_features",
    "ClassicalResult",
    "classical_features",
    "WindowResult",
    "rolling_window_features",
    "LociData",
    "Metadata",
    "read_vcf",
    "read_structure",
    "MOMENT_FIELDS",
    "MomentAccumulator",
    "MomentSummary",
    "moments_from_values",
    "moments_matrix",
    "absence_prob_matrix",
    "locus_statistics",
    "__version__",
]
