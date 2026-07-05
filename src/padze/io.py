"""Input readers for PADZE: VCF and STRUCTURE -> aligned allele-count loci.

PADZE's theory-compatible input is a set of **loci**, each providing, for every
labeled population ``j``, the count ``N_ji`` of each distinct allele and the number ``N_j``
of non-missing sampled gene copies. This module produces exactly that from two conventional
population-genetics formats:

* **VCF** (``.vcf`` / ``.vcf.gz``) -- the recommended input. Requires a *population map*
  assigning sample IDs to population labels. Genotypes are read from the ``GT`` subfield;
  ploidy is detected per genotype; ``.`` alleles are treated as missing. Multiallelic sites
  are supported (alleles aligned by VCF allele index across populations).
* **STRUCTURE** (``.stru``) -- the format the vendored C++ ADZE consumes, so the same file
  can be run through both implementations for parity testing.

Returned object: :class:`LociData`, carrying the per-locus aligned count matrices, the
per-locus per-population sample sizes, and a :class:`Metadata` record stating exactly what
was read (population labels, sample IDs, detected ploidy, filters applied, and missingness).
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["Metadata", "LociData", "read_vcf", "read_structure"]


@dataclass
class Metadata:
    """A transparent record of what was read and how it was filtered."""

    source: str
    populations: List[str]
    sample_ids: Dict[str, List[str]]          # population -> sample IDs
    ploidy: Dict[str, int]                    # sample -> detected ploidy (mode)
    n_loci_read: int
    n_loci_kept: int
    filters_applied: List[str]
    missing_fraction: float                   # over kept loci, fraction of missing calls
    deleted_loci: List[str] = field(default_factory=list)
    all_missing_loci: List[str] = field(default_factory=list)
    missing_tolerance: float | None = None
    feature_note: str = ""

    def summary(self) -> str:
        lines = [
            f"source: {self.source}",
            f"populations ({len(self.populations)}): {', '.join(self.populations)}",
            "samples/pop: " + ", ".join(
                f"{p}={len(self.sample_ids.get(p, []))}" for p in self.populations),
            f"loci: kept {self.n_loci_kept} / read {self.n_loci_read}",
            f"filters: {', '.join(self.filters_applied) or 'none'}",
            f"missing call fraction (kept): {self.missing_fraction:.4g}",
        ]
        if self.all_missing_loci:
            lines.append(f"all-missing loci dropped: {len(self.all_missing_loci)}")
        if self.feature_note:
            lines.append(self.feature_note)
        return "\n".join(lines)


@dataclass
class LociData:
    """Aligned per-locus allele counts ready for the rarefaction core.

    Attributes
    ----------
    populations : list of str
        Population labels in the column order used by ``count_matrices`` rows.
    count_matrices : list of (P, A_l) int arrays
        One per kept locus; row ``j`` is population ``populations[j]``; column ``a`` is the
        same allele across populations.
    sample_sizes : (n_loci, P) int array
        ``N_j`` (non-missing gene copies) per locus per population.
    locus_ids : list of str
    metadata : Metadata
    """

    populations: List[str]
    count_matrices: List[np.ndarray]
    sample_sizes: np.ndarray
    locus_ids: List[str]
    metadata: Metadata

    def max_depth(self) -> int:
        """Largest rarefaction depth valid at every kept locus (min over all N_j)."""
        if self.sample_sizes.size == 0:
            return 0
        return int(self.sample_sizes.min())


def _open_text(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _popmap_token(value, what: str, where: str = "") -> str:
    if value is None:
        raise ValueError(f"popmap {what} is missing{where}")
    token = str(value).strip()
    if not token:
        raise ValueError(f"popmap {what} is empty{where}")
    return token


def _parse_popmap(popmap) -> Dict[str, str]:
    """Accept a dict {sample: pop} or a path to a 2-column (sample pop) text file."""
    if isinstance(popmap, dict):
        mapping: Dict[str, str] = {}
        for sample, pop in popmap.items():
            sample_id = _popmap_token(sample, "sample ID")
            pop_label = _popmap_token(pop, "population label")
            if sample_id in mapping:
                raise ValueError(f"duplicate sample ID {sample_id!r} in popmap")
            mapping[sample_id] = pop_label
        if not mapping:
            raise ValueError("popmap is empty")
        return mapping
    mapping: Dict[str, str] = {}
    with _open_text(popmap) as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"popmap line needs 'sample pop': {line!r}")
            sample_id = _popmap_token(parts[0], "sample ID", f" at line {line_no}")
            pop_label = _popmap_token(parts[1], "population label", f" at line {line_no}")
            if sample_id in mapping:
                raise ValueError(
                    f"duplicate sample ID {sample_id!r} in popmap at line {line_no}")
            mapping[sample_id] = pop_label
    if not mapping:
        raise ValueError("popmap is empty")
    return mapping


def _resolve_population_order(
    present: Sequence[str],
    population_order: Sequence[str] | None,
) -> List[str]:
    """Return a validated population order for numbered ADZE statistics."""
    present_list = list(dict.fromkeys(str(p) for p in present))
    present_set = set(present_list)
    if population_order is None:
        return present_list
    order = [str(p).strip() for p in population_order]
    if any(not p for p in order):
        raise ValueError("population_order contains empty labels")
    if len(order) != len(set(order)):
        raise ValueError("population_order contains duplicate labels")
    order_set = set(order)
    missing = sorted(present_set - order_set)
    extra = sorted(order_set - present_set)
    if missing or extra:
        msg = []
        if missing:
            msg.append(f"missing labels: {', '.join(missing)}")
        if extra:
            msg.append(f"unknown labels: {', '.join(extra)}")
        raise ValueError("population_order must contain exactly the populations present "
                         f"({'; '.join(msg)})")
    return order


def _mode_ploidy(counts: Dict[int, int], default_ploidy: int) -> int:
    return max(counts, key=counts.get) if counts else default_ploidy


def _gt_alleles(gt: str, ploidy_hint: int) -> List[str]:
    """Split a VCF GT field into allele tokens, expanding a bare missing ``.`` by ploidy."""
    gt = gt.replace("|", "/")
    if gt in ("", "."):
        return ["."] * ploidy_hint
    return gt.split("/")


def _validate_vcf_header(header: Sequence[str]) -> None:
    expected = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
    if len(header) < len(expected) or header[:len(expected)] != expected:
        raise ValueError(
            "VCF #CHROM header must start with: " + "\t".join(expected))
    if len(header) > len(expected) and header[len(expected)] != "FORMAT":
        raise ValueError("VCF #CHROM header sample columns must be preceded by FORMAT")


def _vcf_gt_subfield(cell: str, gt_idx: int, chrom: str, pos: str, sample: str) -> str:
    if cell == ".":
        return "."
    if cell == "":
        raise ValueError(
            f"empty VCF genotype field at {chrom}:{pos}, sample {sample!r}")
    parts = cell.split(":")
    if gt_idx >= len(parts):
        raise ValueError(
            f"VCF genotype field at {chrom}:{pos}, sample {sample!r} has "
            f"{len(parts)} subfield(s), but GT is FORMAT subfield {gt_idx + 1}")
    return parts[gt_idx]


def _validate_locus_names(locus_names: Sequence[str]) -> None:
    if len(locus_names) != len(set(locus_names)):
        seen = set()
        dupes = []
        for name in locus_names:
            if name in seen and name not in dupes:
                dupes.append(name)
            seen.add(name)
        raise ValueError("STRUCTURE header contains duplicate locus names: "
                         + ", ".join(dupes))


def read_vcf(
    path: str,
    popmap,
    *,
    population_order: Sequence[str] | None = None,
    min_populations_genotyped: int | None = None,
    require_pass: bool = False,
    max_missing_fraction: float = 1.0,
    max_missing_per_population: float = 1.0,
    biallelic_only: bool = False,
    default_ploidy: int = 2,
) -> LociData:
    """Read a VCF into aligned per-locus allele counts.

    Parameters
    ----------
    path : str
        ``.vcf`` or ``.vcf.gz``.
    popmap : dict or str
        ``{sample_id: population}`` or path to a two-column ``sample population`` file.
        Samples not in the map are ignored.
    population_order : sequence of str, optional
        Explicit order for population-numbered statistics (``alpha_1``, ``pihat_13``, ...).
        Must contain exactly the population labels represented by mapped VCF samples.
    min_populations_genotyped : int, optional
        Drop a locus unless at least this many populations have >= 1 non-missing gene copy
        (default: all mapped populations).
    require_pass : bool
        Keep only sites with FILTER in {PASS, .}.
    max_missing_fraction : float
        Drop a locus whose fraction of missing gene copies (over *all* mapped samples)
        exceeds this.
    max_missing_per_population : float
        ADZE ``TOLERANCE`` analog: drop a locus if *any single population's* missing-gene-copy
        fraction exceeds this. This is the per-grouping filter the C++ ADZE applies.
    biallelic_only : bool
        Drop sites with more than one ALT allele.
    default_ploidy : int
        Fallback gene-copy count for a bare missing genotype ``.`` before any ploidy has
        been observed for that sample. Explicit missing genotypes such as ``./.`` use their
        explicit token count.
    """
    if default_ploidy < 1:
        raise ValueError("default_ploidy must be >= 1")
    if min_populations_genotyped is not None and min_populations_genotyped < 1:
        raise ValueError("min_populations_genotyped must be >= 1")
    for name, value in (
        ("max_missing_fraction", max_missing_fraction),
        ("max_missing_per_population", max_missing_per_population),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    mapping = _parse_popmap(popmap)
    populations: List[str] = []
    pop_index: Dict[str, int] = {}
    P = 0

    filters_applied = ["drop all-missing loci"]
    if require_pass:
        filters_applied.append("FILTER in {PASS,.}")
    if biallelic_only:
        filters_applied.append("biallelic only")
    if max_missing_fraction < 1.0:
        filters_applied.append(f"site missing-fraction <= {max_missing_fraction}")
    if max_missing_per_population < 1.0:
        filters_applied.append(
            f"per-population missing-fraction <= {max_missing_per_population} (TOLERANCE)")
    if min_populations_genotyped:
        filters_applied.append(f">= {min_populations_genotyped} pops genotyped")

    count_matrices: List[np.ndarray] = []
    sample_sizes_rows: List[np.ndarray] = []
    locus_ids: List[str] = []
    deleted_loci: List[str] = []
    all_missing_loci: List[str] = []
    samples_by_pop: Dict[str, List[str]] = {}
    ploidy_counts: Dict[str, Dict[int, int]] = {}
    col_for_pop: List[List[int]] = []  # VCF column indices per pop

    n_read = 0
    kept_missing = 0
    kept_calls = 0
    saw_header = False

    with _open_text(path) as fh:
        sample_cols: List[str] = []
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                saw_header = True
                header = line.rstrip("\n").split("\t")
                _validate_vcf_header(header)
                sample_cols = header[9:]
                if len(sample_cols) != len(set(sample_cols)):
                    seen = set()
                    dupes = []
                    for sample in sample_cols:
                        if sample in seen and sample not in dupes:
                            dupes.append(sample)
                        seen.add(sample)
                    raise ValueError(
                        "VCF header contains duplicate sample IDs: " + ", ".join(dupes))
                sample_set = set(sample_cols)
                present_pops = [p for s, p in mapping.items() if s in sample_set]
                if not present_pops:
                    raise ValueError("no VCF samples are present in the popmap")
                populations = _resolve_population_order(present_pops, population_order)
                pop_index = {p: i for i, p in enumerate(populations)}
                P = len(populations)
                if (min_populations_genotyped is not None
                        and not 1 <= min_populations_genotyped <= P):
                    raise ValueError(
                        "min_populations_genotyped must be between 1 and the number "
                        f"of mapped populations ({P})")
                samples_by_pop = {p: [] for p in populations}
                col_for_pop = [[] for _ in populations]
                for ci, s in enumerate(sample_cols):
                    if s in mapping and mapping[s] in pop_index:
                        pop = mapping[s]
                        samples_by_pop[pop].append(s)
                        col_for_pop[pop_index[pop]].append(9 + ci)
                        ploidy_counts[s] = {}
                continue
            if not line.strip():
                continue
            if not saw_header:
                raise ValueError("VCF is missing the #CHROM header before records")
            n_read += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                raise ValueError(f"malformed VCF record with fewer than 8 columns: {line!r}")
            chrom, pos = fields[0], fields[1]
            expected_cols = 9 + len(sample_cols)
            if len(fields) > 8 and len(fields) != expected_cols:
                observed = max(len(fields) - 9, 0)
                raise ValueError(
                    f"VCF record at {chrom}:{pos} has {observed} sample genotype "
                    f"columns, but the header has {len(sample_cols)}")
            alt = fields[4]
            flt = fields[6]
            n_alt_alleles = len([a for a in alt.split(",") if a not in (".", "")])
            n_alleles = 1 + n_alt_alleles

            if require_pass and flt not in ("PASS", ".", ""):
                continue
            if biallelic_only and n_alt_alleles > 1:
                continue

            if len(fields) <= 8:
                raise ValueError(f"VCF record at {chrom}:{pos} is missing the FORMAT column")
            fmt = fields[8].split(":")
            if not fmt or any(part == "" for part in fmt):
                raise ValueError(f"VCF record at {chrom}:{pos} has a malformed FORMAT field")
            if "GT" not in fmt:
                raise ValueError(f"VCF record at {chrom}:{pos} is missing a GT FORMAT field")
            if fmt.count("GT") > 1:
                raise ValueError(f"VCF record at {chrom}:{pos} has duplicate GT FORMAT fields")
            gt_idx = fmt.index("GT")

            counts = np.zeros((P, n_alleles), dtype=np.int64)
            pop_missing = np.zeros(P, dtype=np.int64)
            pop_calls = np.zeros(P, dtype=np.int64)
            site_missing = 0
            site_calls = 0
            for j in range(P):
                for col in col_for_pop[j]:
                    if col >= len(fields):
                        continue
                    sample = sample_cols[col - 9]
                    cell = fields[col]
                    gt = _vcf_gt_subfield(cell, gt_idx, chrom, pos, sample)
                    alleles = _gt_alleles(
                        gt,
                        _mode_ploidy(ploidy_counts[sample], default_ploidy),
                    )
                    for a in alleles:
                        site_calls += 1
                        pop_calls[j] += 1
                        if a == "." or a == "":
                            site_missing += 1
                            pop_missing[j] += 1
                            continue
                        try:
                            ai = int(a)
                        except ValueError as exc:
                            raise ValueError(
                                f"non-integer VCF GT allele {a!r} at {chrom}:{pos}, "
                                f"sample {sample!r}") from exc
                        if ai < 0 or ai >= n_alleles:
                            raise ValueError(
                                f"VCF GT allele index {ai} exceeds REF/ALT allele count "
                                f"at {chrom}:{pos}, sample {sample!r}")
                        counts[j, ai] += 1
                    if alleles:
                        ploidy_counts[sample][len(alleles)] = (
                            ploidy_counts[sample].get(len(alleles), 0) + 1)

            site_miss_frac = site_missing / site_calls if site_calls else 1.0
            if site_miss_frac > max_missing_fraction:
                continue
            # C++ ADZE TOLERANCE: drop the locus if ANY single population's missing
            # fraction exceeds the threshold.
            if max_missing_per_population < 1.0:
                with np.errstate(invalid="ignore", divide="ignore"):
                    pop_frac = np.where(pop_calls > 0, pop_missing / np.maximum(pop_calls, 1),
                                        0.0)
                if np.any(pop_frac > max_missing_per_population):
                    deleted_loci.append(f"{chrom}:{pos}")
                    continue

            N = counts.sum(axis=1)
            genotyped_pops = int((N > 0).sum())
            if genotyped_pops == 0:
                all_missing_loci.append(f"{chrom}:{pos}")
                continue
            need = min_populations_genotyped if min_populations_genotyped else P
            if genotyped_pops < need:
                continue

            count_matrices.append(counts)
            sample_sizes_rows.append(N)
            locus_ids.append(f"{chrom}:{pos}")
            kept_calls += site_calls
            kept_missing += site_missing

    if not saw_header:
        raise ValueError("VCF is missing the #CHROM header")

    ploidy = {}
    for s, c in ploidy_counts.items():
        ploidy[s] = max(c, key=c.get) if c else 0

    sample_sizes = (np.vstack(sample_sizes_rows) if sample_sizes_rows
                    else np.zeros((0, P), dtype=np.int64))
    miss_frac = (kept_missing / kept_calls) if kept_calls else 0.0
    meta = Metadata(
        source=f"VCF:{path}",
        populations=populations,
        sample_ids=samples_by_pop,
        ploidy=ploidy,
        n_loci_read=n_read,
        n_loci_kept=len(count_matrices),
        filters_applied=filters_applied,
        missing_fraction=miss_frac,
        deleted_loci=deleted_loci,
        all_missing_loci=all_missing_loci,
        missing_tolerance=(max_missing_per_population
                           if max_missing_per_population < 1.0 else None),
        feature_note=("ADZE feature shape per locus: alpha_j, pi_j (j=1..P) and "
                      "pihat over population combinations; summarized across loci as "
                      "(mean, variance, se, skewness, kurtosis) per rarefaction depth."),
    )
    return LociData(populations, count_matrices, sample_sizes, locus_ids, meta)


def read_structure(
    path: str,
    *,
    pop_column: int = 1,
    label_column: int = 0,
    meta_columns: int | None = None,
    missing: str = "-9",
    max_missing_per_population: float = 1.0,
    one_row_per_individual: bool = False,
    ploidy: int = 2,
    header_rows: int = 0,
    population_order: Sequence[str] | None = None,
) -> LociData:
    """Read a STRUCTURE-format file (the C++ ADZE input) into aligned loci.

    The STRUCTURE layout: each non-header row is one gene copy (haplotype). Leading columns
    are an individual label and a population index; the remaining columns are the allele at
    each locus (allele tokens are compared as strings, as in ADZE 1.0; ``missing`` marks
    missing). With
    ``one_row_per_individual=True`` each row instead packs ``ploidy`` alleles per locus
    contiguously. This mirrors what ``VCFtoSTRU.py`` emits and lets the same file feed both
    PADZE and the vendored C++ ADZE for parity checks.

    ``header_rows`` leading rows (e.g. ADZE's locus-name row, set ``NON_DATA_ROWS``) are
    skipped before parsing gene copies. ``meta_columns`` corresponds to ADZE's
    ``NON_DATA_COLS``; if omitted, genotype columns start after the larger of
    ``pop_column`` and ``label_column``.
    """
    if ploidy < 1:
        raise ValueError("ploidy must be >= 1")
    if header_rows < 0:
        raise ValueError("header_rows must be >= 0")
    missing = str(missing)
    if missing == "":
        raise ValueError("missing marker must not be empty")
    if not 0.0 <= max_missing_per_population <= 1.0:
        raise ValueError("max_missing_per_population must be between 0 and 1")
    if pop_column < 0 or label_column < 0:
        raise ValueError("pop_column and label_column must be >= 0")
    rows: List[List[str]] = []
    header: List[List[str]] = []
    with _open_text(path) as fh:
        for i, line in enumerate(fh):
            if i < header_rows:
                if line.strip():
                    header.append(line.split())
                continue
            if not line.strip():
                continue
            rows.append(line.split())
    if not rows:
        raise ValueError("STRUCTURE file has no data rows")

    min_meta_cols = max(pop_column, label_column) + 1
    meta_cols = min_meta_cols if meta_columns is None else int(meta_columns)
    if meta_cols < min_meta_cols:
        raise ValueError("meta_columns must include the pop and label columns")
    pops_seen: List[str] = []
    # Determine number of loci from the first data row.
    first = rows[0]
    if len(first) <= meta_cols:
        raise ValueError("STRUCTURE rows have no genotype columns")
    n_geno_cols = len(first) - meta_cols
    if one_row_per_individual and n_geno_cols % ploidy != 0:
        raise ValueError("STRUCTURE genotype columns are not divisible by ploidy")
    n_loci = n_geno_cols // (ploidy if one_row_per_individual else 1)
    locus_names: List[str] | None = None
    if header:
        last_header = header[-1]
        if len(last_header) == n_loci:
            locus_names = last_header
        elif len(last_header) >= meta_cols + n_loci:
            locus_names = last_header[meta_cols:meta_cols + n_loci]
        if locus_names is not None:
            _validate_locus_names(locus_names)

    # Collect, per population, the list of allele tokens at each locus.
    per_pop_locus_alleles: Dict[str, List[List[str]]] = {}
    per_pop_locus_missing: Dict[str, List[int]] = {}
    per_pop_locus_calls: Dict[str, List[int]] = {}
    for r in rows:
        if len(r) != len(first):
            raise ValueError("STRUCTURE rows must all have the same number of columns")
        pop = r[pop_column]
        if pop not in per_pop_locus_alleles:
            per_pop_locus_alleles[pop] = [[] for _ in range(n_loci)]
            per_pop_locus_missing[pop] = [0 for _ in range(n_loci)]
            per_pop_locus_calls[pop] = [0 for _ in range(n_loci)]
            pops_seen.append(pop)
        geno = r[meta_cols:]
        if one_row_per_individual:
            for locus in range(n_loci):
                for h in range(ploidy):
                    code = geno[locus * ploidy + h]
                    per_pop_locus_calls[pop][locus] += 1
                    if code != missing:
                        per_pop_locus_alleles[pop][locus].append(code)
                    else:
                        per_pop_locus_missing[pop][locus] += 1
        else:
            for locus in range(n_loci):
                code = geno[locus]
                per_pop_locus_calls[pop][locus] += 1
                if code != missing:
                    per_pop_locus_alleles[pop][locus].append(code)
                else:
                    per_pop_locus_missing[pop][locus] += 1

    populations = _resolve_population_order(pops_seen, population_order)
    P = len(populations)
    count_matrices: List[np.ndarray] = []
    sample_sizes_rows: List[np.ndarray] = []
    locus_ids: List[str] = []
    deleted_loci: List[str] = []
    all_missing_loci: List[str] = []
    kept_missing = 0
    kept_calls = 0
    for locus in range(n_loci):
        drop = False
        for p in populations:
            calls = per_pop_locus_calls[p][locus]
            miss = per_pop_locus_missing[p][locus]
            frac = (miss / calls) if calls else 1.0
            if frac > max_missing_per_population:
                drop = True
                break
        if drop:
            deleted_loci.append(locus_names[locus] if locus_names else f"locus_{locus}")
            continue
        # Align alleles across populations at this locus in ADZE's first-seen order.
        allele_codes: List[str] = []
        seen_alleles = set()
        for p in populations:
            for a in per_pop_locus_alleles[p][locus]:
                if a not in seen_alleles:
                    seen_alleles.add(a)
                    allele_codes.append(a)
        if not allele_codes:
            all_missing_loci.append(locus_names[locus] if locus_names else f"locus_{locus}")
            continue
        code_index = {c: i for i, c in enumerate(allele_codes)}
        counts = np.zeros((P, len(allele_codes)), dtype=np.int64)
        for j, p in enumerate(populations):
            for a in per_pop_locus_alleles[p][locus]:
                counts[j, code_index[a]] += 1
        count_matrices.append(counts)
        sample_sizes_rows.append(counts.sum(axis=1))
        locus_ids.append(locus_names[locus] if locus_names else f"locus_{locus}")
        kept_missing += sum(per_pop_locus_missing[p][locus] for p in populations)
        kept_calls += sum(per_pop_locus_calls[p][locus] for p in populations)

    sample_sizes = (np.vstack(sample_sizes_rows) if sample_sizes_rows
                    else np.zeros((0, P), dtype=np.int64))
    sample_ids = {p: [] for p in populations}
    filters = ["drop all-missing loci"]
    if max_missing_per_population < 1.0:
        filters.append(
            f"per-population missing-fraction <= {max_missing_per_population} (TOLERANCE)")
    meta = Metadata(
        source=f"STRUCTURE:{path}",
        populations=populations,
        sample_ids=sample_ids,
        ploidy={},
        n_loci_read=n_loci,
        n_loci_kept=len(count_matrices),
        filters_applied=filters,
        missing_fraction=(kept_missing / kept_calls) if kept_calls else 0.0,
        deleted_loci=deleted_loci,
        all_missing_loci=all_missing_loci,
        missing_tolerance=(max_missing_per_population
                           if max_missing_per_population < 1.0 else None),
        feature_note="Read in STRUCTURE layout (C++ ADZE-compatible).",
    )
    return LociData(populations, count_matrices, sample_sizes, locus_ids, meta)
