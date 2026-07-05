"""End-to-end test: VCF -> LociData -> FeatureTable, plus metadata + determinism.

Exercises the user-facing input contract on a tiny 3-population trio fixture (the DNNaic
setting: 9 statistics alpha_1..3, pi_1..3, pihat_12/13/23). Verifies the VCF reader
handles ploidy, missingness, and population labels, and that the feature table has the
expected shape and is deterministic.
"""
import math
import os
import sys
import tempfile
from contextlib import redirect_stderr
import io as stdlib_io

import numpy as np

HERE = os.path.dirname(__file__)
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from padze import compute_features, read_structure, read_vcf  # noqa: E402
from padze.cli import main as cli_main  # noqa: E402

VCF = os.path.join(HERE, "fixtures", "trio.vcf")
POPMAP = os.path.join(HERE, "fixtures", "trio.popmap")


def _expect_value_error(message, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        assert False, "expected ValueError"
    except ValueError as e:
        assert message in str(e)


def _expect_cli_error(argv, message):
    err = stdlib_io.StringIO()
    try:
        with redirect_stderr(err):
            cli_main(argv)
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 2
        assert message in err.getvalue()


def test_vcf_metadata():
    loci = read_vcf(VCF, POPMAP)
    md = loci.metadata
    assert md.populations == ["A", "B", "C"]
    assert all(len(md.sample_ids[p]) == 2 for p in md.populations)
    # diploid samples detected
    assert set(md.ploidy.values()) == {2}
    # 7 sites read; all kept by default (default: require all pops genotyped)
    assert md.n_loci_read == 7
    assert md.n_loci_kept >= 6
    # one missing allele (site 600, sA2 "0/.") -> nonzero missing fraction
    assert md.missing_fraction > 0.0
    # max usable depth = min N_j; one missing call drops a pop's N at site 600,
    # but per-locus N varies; overall max_depth = min over kept loci.
    assert loci.max_depth() >= 2


def test_vcf_sample_sizes():
    loci = read_vcf(VCF, POPMAP)
    # Each population has 2 diploid samples -> N = 4, except where a call is missing.
    ss = loci.sample_sizes
    assert ss.shape[1] == 3
    # Most entries equal 4; the missing genotype reduces one entry to 3.
    assert (ss == 4).sum() >= ss.size - 2
    assert ss.min() >= 3


def test_feature_table_shape():
    loci = read_vcf(VCF, POPMAP)
    table = compute_features(loci, depths=range(2, loci.max_depth() + 1))
    # 9 statistics for 3 populations: 3 alpha + 3 pi + 3 pihat(pairwise)
    assert len(table.stat_keys) == 9
    assert set(table.stat_keys) == {
        "alpha_1", "alpha_2", "alpha_3", "pi_1", "pi_2", "pi_3",
        "pihat_12", "pihat_13", "pihat_23",
    }
    mat, cols = table.to_frame()
    # columns: g + 9 stats * 5 moments = 1 + 45 = 46
    assert mat.shape[1] == 46
    assert cols[0] == "g"
    assert "alpha_1_skewness" in cols and "pihat_23_kurtosis" in cols
    assert mat.shape[0] == table.depths.size


def test_classic_three_moment_layout():
    # Restricting to (mean, variance, se) reproduces the DNNaic 28-D layout: 1 + 9*3 = 28.
    loci = read_vcf(VCF, POPMAP)
    table = compute_features(loci, depths=range(2, loci.max_depth() + 1),
                             moments=("mean", "variance", "se"))
    mat, cols = table.to_frame()
    assert mat.shape[1] == 28
    assert cols == [
        "g",
        "alpha_1_mean", "alpha_1_variance", "alpha_1_se",
        "alpha_2_mean", "alpha_2_variance", "alpha_2_se",
        "alpha_3_mean", "alpha_3_variance", "alpha_3_se",
        "pi_1_mean", "pi_1_variance", "pi_1_se",
        "pi_2_mean", "pi_2_variance", "pi_2_se",
        "pi_3_mean", "pi_3_variance", "pi_3_se",
        "pihat_12_mean", "pihat_12_variance", "pihat_12_se",
        "pihat_13_mean", "pihat_13_variance", "pihat_13_se",
        "pihat_23_mean", "pihat_23_variance", "pihat_23_se",
    ]


def test_determinism():
    loci1 = read_vcf(VCF, POPMAP)
    loci2 = read_vcf(VCF, POPMAP)
    t1 = compute_features(loci1, depths=range(2, 4))
    t2 = compute_features(loci2, depths=range(2, 4))
    m1, _ = t1.to_frame()
    m2, _ = t2.to_frame()
    assert np.array_equal(np.nan_to_num(m1, nan=-999), np.nan_to_num(m2, nan=-999))


def test_alpha_ge_pi_on_real_fixture():
    loci = read_vcf(VCF, POPMAP)
    table = compute_features(loci, depths=range(2, loci.max_depth() + 1))
    for j in (1, 2, 3):
        a = table.values[f"alpha_{j}"]["mean"]
        p = table.values[f"pi_{j}"]["mean"]
        good = ~(np.isnan(a) | np.isnan(p))
        assert np.all(a[good] >= p[good] - 1e-9)


def test_per_population_tolerance():
    # Site chr1:600 has one missing allele in population A (sA2 "0/.") -> A missing frac 0.25.
    # ADZE TOLERANCE analog drops the locus when any population exceeds the threshold.
    base = read_vcf(VCF, POPMAP)
    strict = read_vcf(VCF, POPMAP, max_missing_per_population=0.2)
    assert strict.metadata.n_loci_kept == base.metadata.n_loci_kept - 1
    assert any("TOLERANCE" in f for f in strict.metadata.filters_applied)
    # ADZE uses strict > TOLERANCE, so exactly-at-threshold missingness is retained.
    boundary = read_vcf(VCF, POPMAP, max_missing_per_population=0.25)
    assert boundary.metadata.n_loci_kept == base.metadata.n_loci_kept
    # A lenient threshold keeps everything.
    lenient = read_vcf(VCF, POPMAP, max_missing_per_population=0.5)
    assert lenient.metadata.n_loci_kept == base.metadata.n_loci_kept


def test_missing_fraction_reports_kept_loci_only():
    # The fixture has exactly one missing allele at chr1:600. Filtering missing sites out
    # should make the reported kept-locus missing fraction exactly zero.
    base = read_vcf(VCF, POPMAP)
    strict = read_vcf(VCF, POPMAP, max_missing_fraction=0.0)
    assert strict.metadata.n_loci_kept == base.metadata.n_loci_kept - 1
    assert strict.metadata.missing_fraction == 0.0


def test_vcf_requires_mapped_samples():
    try:
        read_vcf(VCF, {"not_in_vcf": "A"})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "no VCF samples" in str(e)


def test_vcf_rejects_out_of_range_gt_allele():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2",
        "chr1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/2\t0/1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as fh:
        fh.write(text)
        fh.flush()
        try:
            read_vcf(fh.name, {"s1": "A", "s2": "B"})
            assert False, "expected ValueError"
        except ValueError as e:
            assert "exceeds REF/ALT" in str(e)


def test_vcf_rejects_records_without_gt_format():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2",
        "chr1\t1\t.\tA\tG\t.\tPASS\t.\tDP\t12\t14",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as fh:
        fh.write(text)
        fh.flush()
        try:
            read_vcf(fh.name, {"s1": "A", "s2": "B"})
            assert False, "expected ValueError"
        except ValueError as e:
            assert "missing a GT FORMAT field" in str(e)


def test_vcf_rejects_duplicate_sample_ids():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts1",
        "chr1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t0/1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as fh:
        fh.write(text)
        fh.flush()
        try:
            read_vcf(fh.name, {"s1": "A"})
            assert False, "expected ValueError"
        except ValueError as e:
            assert "duplicate sample IDs" in str(e)


def test_popmap_file_rejects_duplicate_sample_ids():
    with tempfile.NamedTemporaryFile("w", suffix=".popmap") as popmap:
        popmap.write("s1 A\ns1 B\n")
        popmap.flush()
        try:
            read_vcf(VCF, popmap.name)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "duplicate sample ID" in str(e)


def test_popmap_rejects_empty_mapping_and_labels():
    _expect_value_error("popmap is empty", read_vcf, VCF, {})
    _expect_value_error("popmap sample ID is empty", read_vcf, VCF, {"": "A"})
    _expect_value_error("popmap population label is empty", read_vcf, VCF, {"sA1": ""})


def test_vcf_rejects_missing_chrom_header():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "##source=no header",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as fh:
        fh.write(text)
        fh.flush()
        _expect_value_error("missing the #CHROM header", read_vcf, fh.name, {"s1": "A"})


def test_vcf_rejects_header_samples_without_format_column():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\ts1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as fh:
        fh.write(text)
        fh.flush()
        _expect_value_error("preceded by FORMAT", read_vcf, fh.name, {"s1": "A"})


def test_vcf_rejects_record_sample_width_mismatch():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2",
        "chr1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as fh:
        fh.write(text)
        fh.flush()
        _expect_value_error("header has 2", read_vcf, fh.name, {"s1": "A", "s2": "B"})


def test_vcf_rejects_sample_cell_missing_gt_subfield():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2",
        "chr1\t1\t.\tA\tG\t.\tPASS\t.\tDP:GT\t12\t10:0/1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as fh:
        fh.write(text)
        fh.flush()
        _expect_value_error("GT is FORMAT subfield 2", read_vcf, fh.name,
                            {"s1": "A", "s2": "B"})


def test_vcf_allows_whole_sample_missing_cell_when_gt_is_later_format_field():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2",
        "chr1\t1\t.\tA\tG\t.\tPASS\t.\tDP:GT\t.\t10:0/1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as fh:
        fh.write(text)
        fh.flush()
        loci = read_vcf(fh.name, {"s1": "A", "s2": "B"},
                        min_populations_genotyped=1)
    assert loci.locus_ids == ["chr1:1"]
    assert np.array_equal(loci.sample_sizes[0], np.array([0, 2]))


def test_vcf_rejects_invalid_reader_parameters():
    _expect_value_error("max_missing_fraction must be between 0 and 1",
                        read_vcf, VCF, POPMAP, max_missing_fraction=-0.1)
    _expect_value_error("max_missing_per_population must be between 0 and 1",
                        read_vcf, VCF, POPMAP, max_missing_per_population=1.1)
    _expect_value_error("default_ploidy must be >= 1",
                        read_vcf, VCF, POPMAP, default_ploidy=0)
    _expect_value_error("min_populations_genotyped must be >= 1",
                        read_vcf, VCF, POPMAP, min_populations_genotyped=0)
    _expect_value_error("number of mapped populations (3)",
                        read_vcf, VCF, POPMAP, min_populations_genotyped=4)


def test_population_order_rejects_invalid_orders():
    _expect_value_error("duplicate labels", read_vcf, VCF, POPMAP,
                        population_order=["A", "A", "B"])
    _expect_value_error("missing labels: C", read_vcf, VCF, POPMAP,
                        population_order=["A", "B"])
    _expect_value_error("unknown labels: Z", read_vcf, VCF, POPMAP,
                        population_order=["A", "B", "Z"])
    _expect_value_error("empty labels", read_vcf, VCF, POPMAP,
                        population_order=["A", "B", ""])


def test_structure_empty_file_errors_clearly():
    with tempfile.NamedTemporaryFile("w", suffix=".stru") as fh:
        try:
            read_structure(fh.name)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "no data rows" in str(e)


def test_per_locus_exposure():
    # The FULL_R/P/C equivalent: per-locus values exposed and convertible to long format.
    loci = read_vcf(VCF, POPMAP)
    table = compute_features(loci, depths=range(2, loci.max_depth() + 1),
                             moments=("mean",), keep_per_locus=True)
    assert table.per_locus is not None
    for s in table.stat_keys:
        assert table.per_locus[s].shape == (table.n_loci, table.depths.size)
    rows, cols = table.per_locus_frame()
    assert cols == ["locus", "statistic", "g", "value"]
    assert len(rows) > 0
    # Without keep_per_locus, asking for the frame errors clearly.
    t2 = compute_features(loci, depths=range(2, 4), moments=("mean",))
    try:
        t2.per_locus_frame()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dict_popmap():
    # popmap may be passed inline as a dict
    mapping = {"sA1": "A", "sA2": "A", "sB1": "B", "sB2": "B", "sC1": "C", "sC2": "C"}
    loci = read_vcf(VCF, mapping)
    assert loci.populations == ["A", "B", "C"]


def test_population_order_controls_numbered_features():
    base = read_vcf(VCF, POPMAP)
    ordered = read_vcf(VCF, POPMAP, population_order=["C", "A", "B"])
    assert ordered.populations == ["C", "A", "B"]

    t_base = compute_features(base, depths=[2], moments=("mean",))
    t_ordered = compute_features(ordered, depths=[2], moments=("mean",))
    assert np.allclose(t_ordered.values["alpha_1"]["mean"],
                       t_base.values["alpha_3"]["mean"])
    assert np.allclose(t_ordered.values["pi_2"]["mean"],
                       t_base.values["pi_1"]["mean"])


def test_structure_reads_adze_layout_and_tolerance():
    text = "\n".join([
        "Lkeep1 Ldel1 Lkeep2 Ldel2 Lkeep3",
        "a1 x x x AMERICA 1 -9 1 1 1",
        "a2 x x x AMERICA 1 1 1 1 2",
        "a3 x x x AMERICA 2 1 2 1 2",
        "a4 x x x AMERICA 2 1 2 1 2",
        "e1 x x x EUROPE 1 1 1 -9 1",
        "e2 x x x EUROPE 1 1 1 1 2",
        "e3 x x x EUROPE 2 1 2 1 2",
        "e4 x x x EUROPE 2 1 2 1 2",
        "ea1 x x x EAST_ASIA 1 1 1 1 1",
        "ea2 x x x EAST_ASIA 1 1 1 1 2",
        "ea3 x x x EAST_ASIA 2 1 2 1 2",
        "ea4 x x x EAST_ASIA 2 1 2 1 2",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".stru") as fh:
        fh.write(text)
        fh.flush()
        loci = read_structure(
            fh.name,
            header_rows=1,
            meta_columns=5,
            pop_column=4,       # zero-based equivalent of ADZE GROUP_BY_COL 5
            label_column=0,
            max_missing_per_population=0.2,
        )
    assert loci.populations == ["AMERICA", "EUROPE", "EAST_ASIA"]
    assert loci.metadata.n_loci_read == 5
    assert loci.metadata.n_loci_kept == 3
    assert loci.locus_ids == ["Lkeep1", "Lkeep2", "Lkeep3"]
    assert loci.metadata.deleted_loci == ["Ldel1", "Ldel2"]
    assert loci.metadata.missing_tolerance == 0.2


def test_structure_rejects_inconsistent_row_widths():
    text = "\n".join([
        "i1 A 1 2",
        "i2 B 1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".stru") as fh:
        fh.write(text)
        fh.flush()
        _expect_value_error("same number of columns", read_structure, fh.name, ploidy=1)


def test_structure_rejects_invalid_meta_and_ploidy_layouts():
    text = "\n".join([
        "i1 A 1 2 3",
        "i2 B 1 2 3",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".stru") as fh:
        fh.write(text)
        fh.flush()
        _expect_value_error("meta_columns must include", read_structure, fh.name,
                            meta_columns=1)
        _expect_value_error("not divisible by ploidy", read_structure, fh.name,
                            one_row_per_individual=True, ploidy=2)
        _expect_value_error("ploidy must be >= 1", read_structure, fh.name, ploidy=0)
        _expect_value_error("header_rows must be >= 0", read_structure, fh.name,
                            header_rows=-1)
        _expect_value_error("missing marker must not be empty", read_structure, fh.name,
                            missing="")


def test_structure_rejects_duplicate_locus_names_in_header():
    text = "\n".join([
        "L1 L1",
        "i1 A 1 2",
        "i2 B 2 1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".stru") as fh:
        fh.write(text)
        fh.flush()
        _expect_value_error("duplicate locus names", read_structure, fh.name,
                            header_rows=1, ploidy=1)


def test_structure_population_order_rejects_invalid_orders():
    text = "\n".join([
        "i1 A 1",
        "i2 B 2",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".stru") as fh:
        fh.write(text)
        fh.flush()
        _expect_value_error("duplicate labels", read_structure, fh.name,
                            ploidy=1, population_order=["A", "A"])
        _expect_value_error("missing labels: B", read_structure, fh.name,
                            ploidy=1, population_order=["A"])
        _expect_value_error("unknown labels: C", read_structure, fh.name,
                            ploidy=1, population_order=["A", "C"])


def test_structure_preserves_string_allele_tokens():
    text = "\n".join([
        "L1",
        "i1 A 01",
        "i2 A 1",
        "i3 B 01",
        "i4 B 1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".stru") as fh:
        fh.write(text)
        fh.flush()
        loci = read_structure(fh.name, header_rows=1, ploidy=1)
    assert loci.count_matrices[0].shape == (2, 2)
    assert np.array_equal(loci.count_matrices[0], np.array([[1, 1], [1, 1]]))


def test_structure_reports_all_missing_loci_separately():
    text = "\n".join([
        "Lmissing Lkept",
        "i1 A -9 1",
        "i2 A -9 1",
        "i3 B -9 2",
        "i4 B -9 2",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".stru") as fh:
        fh.write(text)
        fh.flush()
        loci = read_structure(fh.name, header_rows=1, ploidy=1)
    assert loci.locus_ids == ["Lkept"]
    assert loci.metadata.deleted_loci == []
    assert loci.metadata.all_missing_loci == ["Lmissing"]
    assert "drop all-missing loci" in loci.metadata.filters_applied


def test_vcf_reports_all_missing_loci_separately():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2",
        "chr1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t./.\t.",
        "chr1\t2\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t0/1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as fh:
        fh.write(text)
        fh.flush()
        loci = read_vcf(fh.name, {"s1": "A", "s2": "B"})
    assert loci.locus_ids == ["chr1:2"]
    assert loci.metadata.all_missing_loci == ["chr1:1"]
    assert "drop all-missing loci" in loci.metadata.filters_applied


def test_vcf_bare_missing_gt_uses_inferred_ploidy_for_filters():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2",
        "chr1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t0/1",
        "chr1\t2\t.\tA\tG\t.\tPASS\t.\tGT\t.\t0/1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as fh:
        fh.write(text)
        fh.flush()
        loci = read_vcf(fh.name, {"s1": "A", "s2": "B"}, max_missing_fraction=0.4)
    assert loci.locus_ids == ["chr1:1"]
    assert loci.metadata.ploidy["s1"] == 2


def test_vcf_min_populations_genotyped_can_keep_ragged_classical_loci():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2",
        "chr1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t.",
        "chr1\t2\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t0/1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as fh:
        fh.write(text)
        fh.flush()
        strict = read_vcf(fh.name, {"s1": "A", "s2": "B"})
        ragged = read_vcf(fh.name, {"s1": "A", "s2": "B"}, min_populations_genotyped=1)
    assert strict.locus_ids == ["chr1:2"]
    assert ragged.locus_ids == ["chr1:1", "chr1:2"]
    assert np.array_equal(ragged.sample_sizes[0], np.array([2, 0]))


def test_cli_rejects_ambiguous_input_source():
    try:
        cli_main([
            "features",
            "--vcf", VCF,
            "--popmap", POPMAP,
            "--structure", os.path.join(REPO, "unused.stru"),
        ])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 2


def test_cli_rejects_missing_input_source():
    _expect_cli_error(["info"], "provide exactly one of --vcf or --structure")


def test_cli_rejects_vcf_without_popmap():
    _expect_cli_error(["info", "--vcf", VCF], "--popmap is required with --vcf")


def test_cli_reports_reader_validation_errors():
    _expect_cli_error([
        "info",
        "--vcf", VCF,
        "--popmap", POPMAP,
        "--default-ploidy", "0",
    ], "default_ploidy must be >= 1")
    _expect_cli_error([
        "info",
        "--vcf", VCF,
        "--popmap", POPMAP,
        "--max-missing-fraction", "1.2",
    ], "max_missing_fraction must be between 0 and 1")
    _expect_cli_error([
        "info",
        "--structure", os.path.join(REPO, "unused.stru"),
        "--structure-header-rows", "-1",
    ], "header_rows must be >= 0")


def test_cli_rejects_invalid_pihat_size():
    try:
        cli_main([
            "features",
            "--vcf", VCF,
            "--popmap", POPMAP,
            "--pihat-sizes", "4",
        ])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 2


def test_cli_rejects_duplicate_pihat_sizes():
    try:
        cli_main([
            "features",
            "--vcf", VCF,
            "--popmap", POPMAP,
            "--pihat-sizes", "2", "2",
        ])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code == 2


def test_cli_ragged_depth_policy_allows_zero_common_depth():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2",
        "chr1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t.",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as vcf:
        with tempfile.NamedTemporaryFile("w", suffix=".popmap") as popmap:
            vcf.write(text)
            vcf.flush()
            popmap.write("s1 A\ns2 B\n")
            popmap.flush()
            rc = cli_main([
                "features",
                "--vcf", vcf.name,
                "--popmap", popmap.name,
                "--min-populations-genotyped", "1",
                "--depth-policy", "ragged",
                "--moments", "mean",
            ])
    assert rc == 0


def test_cli_one_population_uses_singleton_pihat_default():
    text = "\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2",
        "chr1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t0/1",
        "chr1\t2\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1",
        "",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".vcf") as vcf:
        with tempfile.NamedTemporaryFile("w", suffix=".popmap") as popmap:
            with tempfile.NamedTemporaryFile("r", suffix=".csv") as out:
                vcf.write(text)
                vcf.flush()
                popmap.write("s1 A\ns2 A\n")
                popmap.flush()
                rc = cli_main([
                    "features",
                    "--vcf", vcf.name,
                    "--popmap", popmap.name,
                    "--max-depth", "2",
                    "--moments", "mean",
                    "--out", out.name,
                ])
                out.seek(0)
                header = out.readline().strip()

    assert rc == 0
    assert header == "g,alpha_1_mean,pi_1_mean,pihat_1_mean"


def test_cli_writes_adze_compatible_outputs_from_vcf():
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "adze")
        rc = cli_main([
            "features",
            "--vcf", VCF,
            "--popmap", POPMAP,
            "--max-depth", "3",
            "--adze-prefix", prefix,
            "--adze-full",
        ])
        assert rc == 0
        for suffix in (
            "_r", "_p", "_c_2",
            "_r_fulldata", "_p_fulldata", "_c_2_fulldata",
        ):
            assert os.path.exists(prefix + suffix), suffix
        rich = open(prefix + "_r").read()
        assert "A 2 " in rich and "B 2 " in rich and "C 2 " in rich
        full_header = open(prefix + "_r_fulldata").readline().strip()
        assert full_header.startswith("POP_GROUPING G NUM_LOCI chr1:100 chr1:200")


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} io/features tests passed")


if __name__ == "__main__":
    _main()
