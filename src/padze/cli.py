"""Command-line interface for PADZE.

Examples
--------
Inspect a VCF + popmap (metadata only)::

    python -m padze info --vcf trio.vcf --popmap pops.txt

Compute the rarefaction feature table (row per rarefaction depth) to CSV::

    python -m padze features --vcf trio.vcf --popmap pops.txt \\
        --max-depth 20 --out features.csv

Use the classic 3-moment (mean,variance,se) layout::

    python -m padze features --vcf trio.vcf --popmap pops.txt \\
        --moments mean variance se --out classic.csv

The output is deterministic: identical inputs and flags always produce identical CSV.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from typing import List

import numpy as np

from .features import (
    classical_features,
    compute_features,
    rolling_window_features,
)
from .io import read_structure, read_vcf
from .moments import MOMENT_FIELDS


def _load_loci(args):
    if bool(args.vcf) == bool(args.structure):
        raise ValueError("provide exactly one of --vcf or --structure")
    if args.vcf:
        if not args.popmap:
            raise ValueError("--popmap is required with --vcf")
        min_pops = args.min_populations_genotyped
        if min_pops is None and (
            getattr(args, "classical", False)
            or getattr(args, "adze_prefix", None)
            or getattr(args, "rolling_window", 0)
        ):
            min_pops = 1
        return read_vcf(
            args.vcf,
            args.popmap,
            population_order=args.population_order,
            min_populations_genotyped=min_pops,
            require_pass=args.require_pass,
            max_missing_fraction=args.max_missing_fraction,
            max_missing_per_population=args.max_missing_per_population,
            biallelic_only=args.biallelic_only,
            default_ploidy=args.default_ploidy,
        )
    if args.structure:
        return read_structure(
            args.structure,
            pop_column=args.structure_pop_column - 1,
            label_column=args.structure_label_column - 1,
            meta_columns=args.structure_meta_columns,
            missing=args.structure_missing,
            max_missing_per_population=args.max_missing_per_population,
            ploidy=args.ploidy,
            one_row_per_individual=args.one_row_per_individual,
            header_rows=args.structure_header_rows,
            population_order=args.population_order,
        )
    raise SystemExit("provide --vcf or --structure")


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vcf", help="input .vcf / .vcf.gz")
    p.add_argument("--popmap", help="dict file: 'sample population' per line")
    p.add_argument("--population-order", nargs="+",
                   help="explicit population order for numbered outputs, e.g. P1 P2 P3")
    p.add_argument("--structure", help="input STRUCTURE .stru (C++ ADZE format)")
    p.add_argument("--ploidy", type=int, default=2, help="STRUCTURE ploidy (default 2)")
    p.add_argument("--one-row-per-individual", action="store_true",
                   help="STRUCTURE rows pack ploidy alleles per locus")
    p.add_argument("--structure-header-rows", type=int, default=0,
                   help="STRUCTURE/ADZE NON_DATA_ROWS (default 0)")
    p.add_argument("--structure-meta-columns", type=int, default=None,
                   help="STRUCTURE/ADZE NON_DATA_COLS (default: enough to include label/pop)")
    p.add_argument("--structure-pop-column", type=int, default=2,
                   help="1-based STRUCTURE/ADZE GROUP_BY_COL (default 2)")
    p.add_argument("--structure-label-column", type=int, default=1,
                   help="1-based STRUCTURE individual label column (default 1)")
    p.add_argument("--structure-missing", default="-9",
                   help="STRUCTURE missing-data token (default -9)")
    p.add_argument("--require-pass", action="store_true",
                   help="VCF: keep only FILTER in {PASS,.}")
    p.add_argument("--max-missing-fraction", type=float, default=1.0,
                   help="VCF: drop sites above this overall missing-call fraction")
    p.add_argument("--max-missing-per-population", type=float, default=1.0,
                   help="drop a site/locus if ANY population exceeds this missing fraction "
                        "(ADZE TOLERANCE)")
    p.add_argument("--min-populations-genotyped", type=int, default=None,
                   help="VCF: drop sites with fewer genotyped populations; default is all "
                        "mapped populations for rectangular feature mode and 1 for "
                        "classical/ADZE-compatible modes")
    p.add_argument("--default-ploidy", type=int, default=2,
                   help="VCF: fallback ploidy for bare missing GT '.' before sample ploidy "
                        "is observed (default 2)")
    p.add_argument("--biallelic-only", action="store_true",
                   help="VCF: drop multiallelic sites")


def cmd_info(args) -> int:
    loci = _load_loci(args)
    print(loci.metadata.summary())
    print(f"max usable rarefaction depth: {loci.max_depth()}")
    return 0


def cmd_features(args) -> int:
    loci = _load_loci(args)
    pihat_sizes = _resolve_cli_pihat_sizes(len(loci.populations), args.pihat_sizes)
    if len(pihat_sizes) != len(set(pihat_sizes)):
        raise ValueError("--pihat-sizes contains duplicate values")

    # Rolling-window mode: classical statistics over sliding genomic windows.
    if args.rolling_window:
        if args.adze_prefix:
            raise ValueError("--adze-prefix cannot be combined with --rolling-window")
        step = args.step if args.step else args.rolling_window
        windows = rolling_window_features(
            loci, window=args.rolling_window, step=step, unit=args.window_unit,
            max_g=(args.max_depth or None), pihat_sizes=pihat_sizes)
        lines = ["window,unit,start,end,n_loci,statistic,g,n_used,mean,variance,se"]
        for w in windows:
            for (s, g, n, mean, var, se) in w.result.rows():
                lines.append(",".join([
                    str(w.index), w.unit, _fmt(w.start), _fmt(w.end), str(w.n_loci),
                    s, str(g), str(n), _fmt(mean), _fmt(var), _fmt(se)]))
        _write("\n".join(lines) + "\n", args.out, f"{len(windows)} windows")
        return 0

    if args.adze_prefix:
        res = classical_features(loci, max_g=(args.max_depth or None),
                                 pihat_sizes=pihat_sizes,
                                 keep_per_locus=args.adze_full)
        written = _write_adze_outputs(
            res,
            args.adze_prefix,
            pihat_sizes=pihat_sizes,
            full=args.adze_full,
        )
        print("wrote ADZE-compatible files: " + ", ".join(written), file=sys.stderr)
        return 0

    # Classical exact-match mode: reproduce the C++ ADZE (mean, variance, se) output.
    if args.classical:
        res = classical_features(loci, max_g=(args.max_depth or None),
                                 pihat_sizes=pihat_sizes,
                                 keep_per_locus=False)
        lines = ["statistic,g,n_loci,mean,variance,se"]
        for (s, g, n, mean, var, se) in res.rows():
            lines.append(",".join([s, str(g), str(n), _fmt(mean), _fmt(var), _fmt(se)]))
        _write("\n".join(lines) + "\n", args.out, f"{len(res.summary)} statistic/depth rows")
        return 0

    if args.depth_policy == "common" and loci.max_depth() < 2:
        raise SystemExit(
            f"max usable depth {loci.max_depth()} < 2; check sample sizes / missingness")
    depths = None
    if args.max_depth:
        if args.max_depth < 2:
            raise ValueError("--max-depth must be >= 2")
        depths = np.arange(2, args.max_depth + 1, dtype=np.int64)
    table = compute_features(
        loci,
        depths=depths,
        pihat_sizes=pihat_sizes,
        moments=tuple(args.moments),
        bias_corrected=not args.population_moments,
        keep_per_locus=bool(args.per_locus_out),
        depth_policy=args.depth_policy,
    )
    mat, cols = table.to_frame()

    if args.per_locus_out:
        rows, pcols = table.per_locus_frame()
        with open(args.per_locus_out, "w") as fh:
            fh.write(",".join(pcols) + "\n")
            for r in rows:
                fh.write(f"{r[0]},{r[1]},{int(r[2])},{repr(float(r[3]))}\n")
        print(f"wrote {len(rows)} per-locus rows (FULL_R/P/C equivalent) -> "
              f"{args.per_locus_out}", file=sys.stderr)

    header = ",".join(cols)
    fmt_rows = []
    for row in mat:
        fmt_rows.append(",".join(_fmt(x) for x in row))
    body = "\n".join(fmt_rows)
    text = header + "\n" + body + "\n"

    if args.out and args.out != "-":
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {mat.shape[0]} rows x {mat.shape[1]} cols -> {args.out}",
              file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def _resolve_cli_pihat_sizes(P: int, raw_sizes) -> tuple[int, ...]:
    if raw_sizes is None:
        return (1,) if P == 1 else (2,)
    return tuple(raw_sizes)


def _fmt(x: float) -> str:
    if np.isnan(x):
        return "NA"
    if float(x).is_integer():
        return str(int(x))
    return repr(float(x))


def _write(text: str, out: str, what: str) -> None:
    if out and out != "-":
        with open(out, "w") as fh:
            fh.write(text)
        print(f"wrote {what} -> {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


def _rows_for_stat(res, key: str):
    for (s, g), (mean, var, se) in sorted(res.summary.items(), key=lambda item: item[0][1]):
        if s == key:
            yield g, mean, var, se


def _write_group_file(path: str, res, groups, *, full: bool, full_path: str,
                      full_header_prefix: List[str]) -> List[str]:
    written = []
    with open(path, "w") as fh:
        for labels, key in groups:
            for g, mean, var, se in _rows_for_stat(res, key):
                fh.write(" ".join([
                    *labels, str(g), str(res.n_loci),
                    _fmt(mean), _fmt(var), _fmt(se),
                ]) + "\n")
            fh.write("\n")
    written.append(path)

    if full:
        locus_ids = res.locus_ids or [f"locus_{i}" for i in range(res.n_loci)]
        with open(full_path, "w") as fh:
            fh.write(" ".join([*full_header_prefix, "G", "NUM_LOCI",
                               *locus_ids, "AVG", "VAR", "STD_ERR"]) + "\n")
            for labels, key in groups:
                for g, mean, var, se in _rows_for_stat(res, key):
                    vals = res.per_locus[(key, g)]
                    fh.write(" ".join([
                        *labels, str(g), str(res.n_loci),
                        *(_fmt(float(v)) for v in vals),
                        _fmt(mean), _fmt(var), _fmt(se),
                    ]) + "\n")
                fh.write("\n")
        written.append(full_path)
    return written


def _write_adze_outputs(res, prefix: str, *, pihat_sizes, full: bool) -> List[str]:
    """Write ADZE-style R/P/C text outputs from a ClassicalResult."""
    written: List[str] = []
    pop_groups = [([pop], f"alpha_{i + 1}") for i, pop in enumerate(res.populations)]
    written.extend(_write_group_file(
        f"{prefix}_r", res, pop_groups, full=full,
        full_path=f"{prefix}_r_fulldata", full_header_prefix=["POP_GROUPING"]))

    priv_groups = [([pop], f"pi_{i + 1}") for i, pop in enumerate(res.populations)]
    written.extend(_write_group_file(
        f"{prefix}_p", res, priv_groups, full=full,
        full_path=f"{prefix}_p_fulldata", full_header_prefix=["POP_GROUPING"]))
    if res.deleted_loci:
        deleted_path = f"{prefix}_p_deletedloci"
        tol = res.missing_tolerance if res.missing_tolerance is not None else 0.0
        pct = f"{tol * 100:g}"
        with open(deleted_path, "w") as fh:
            fh.write(f"{len(res.deleted_loci)} loci have at least one grouping with "
                     f"more than {pct}% missing data.\n")
            # C++ ADZE deletes loci by reverse index to keep array positions stable; its
            # *_deletedloci file therefore lists them in reverse input order.
            for locus in reversed(res.deleted_loci):
                fh.write(f"{locus}\n")
        written.append(deleted_path)

    P = len(res.populations)
    for ksz in pihat_sizes:
        combo_groups = []
        for combo in combinations(range(P), ksz):
            key = "pihat_" + "".join(str(c + 1) for c in combo)
            if any(s == key for s, _ in res.summary):
                combo_groups.append(([res.populations[c] for c in combo], key))
        if combo_groups:
            header = [f"POP_GROUPING{i + 1}" for i in range(ksz)]
            written.extend(_write_group_file(
                f"{prefix}_c_{ksz}", res, combo_groups, full=full,
                full_path=f"{prefix}_c_{ksz}_fulldata", full_header_prefix=header))
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="padze",
        description="PADZE: allelic-rarefaction statistics (alpha, pi, pihat) with "
                    "mean/variance/se plus skewness/kurtosis across loci.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("info", help="read input and print metadata")
    _add_input_args(pi)
    pi.set_defaults(func=cmd_info)

    pf = sub.add_parser("features", help="compute the across-loci rarefaction feature table")
    _add_input_args(pf)
    pf.add_argument("--max-depth", type=int, default=0,
                    help="largest rarefaction depth (default: max supported by data)")
    pf.add_argument("--pihat-sizes", type=int, nargs="+", default=None,
                    help="pihat combination sizes (default: 2 for P>=2, 1 for P=1)")
    pf.add_argument("--classical", action="store_true",
                    help="emit the classical (mean,variance,se) output that reproduces the "
                         "C++ ADZE exactly, over each statistic's full ADZE depth range")
    pf.add_argument("--adze-prefix", default=None,
                    help="write ADZE-compatible R/P/C files using this prefix "
                         "(e.g. PREFIX_r, PREFIX_p, PREFIX_c_2)")
    pf.add_argument("--adze-full", action="store_true",
                    help="with --adze-prefix, also write FULL_R/FULL_P/FULL_C-style "
                         "per-locus files")
    pf.add_argument("--rolling-window", type=int, default=0, metavar="W",
                    help="compute the classical statistics over sliding windows of size W")
    pf.add_argument("--step", type=int, default=0, metavar="S",
                    help="rolling-window step (default: W, i.e. non-overlapping)")
    pf.add_argument("--window-unit", choices=["loci", "bp"], default="loci",
                    help="rolling-window unit: number of loci (default) or base pairs")
    pf.add_argument("--moments", nargs="+", default=list(MOMENT_FIELDS),
                    choices=list(MOMENT_FIELDS),
                    help="which across-loci moments to emit (default: all five)")
    pf.add_argument("--depth-policy", choices=["common", "ragged"], default="common",
                    help="feature-table depth policy: common requires every locus/population "
                         "to support each depth; ragged summarizes available per-locus "
                         "values up to the largest sample size")
    pf.add_argument("--population-moments", action="store_true",
                    help="use biased population skew/kurtosis (g1,g2) not sample (G1,G2)")
    pf.add_argument("--out", default="-", help="output CSV path ('-' = stdout)")
    pf.add_argument("--per-locus-out", default=None,
                    help="also write per-locus values (long format; FULL_R/P/C equivalent)")
    pf.set_defaults(func=cmd_features)
    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError) as e:
        parser.exit(2, f"{parser.prog}: error: {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())
