"""Regression against the example distributed with original ADZE.

The files under ``ADZEOriginal/ADZE-1.0`` are the concrete reproducible artifacts bundled
with ADZE: ``small_data.stru`` plus expected ``small_r``, ``small_p``, ``small_c_2`` and
FULL_* outputs.  This test reads the same data through PADZE and checks that the
classical path reproduces every expected summary and per-locus value to the precision
printed by the original program.
"""
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(__file__)
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "src"))

from padze import classical_features, read_structure  # noqa: E402
from padze.cli import main as cli_main  # noqa: E402

BASE = os.path.join(REPO, "ADZEOriginal", "ADZE-1.0")
DATA = os.path.join(BASE, "small_data.stru")
TOL = 1e-4


def _require_original_example():
    if os.path.exists(DATA):
        return True
    try:
        import pytest  # type: ignore
    except Exception:
        print("SKIP original ADZE example: ADZEOriginal/ADZE-1.0 is absent")
        return False
    pytest.skip("ADZEOriginal/ADZE-1.0 example files are absent")


def _load_result():
    if not _require_original_example():
        return None
    # The bundled expected outputs have three loci and a deleted-loci file that says "20%".
    # Use that artifact-level tolerance, not the stale TOLERANCE line in small_paramfile.txt.
    loci = read_structure(
        DATA,
        header_rows=1,
        meta_columns=5,
        pop_column=4,
        label_column=0,
        max_missing_per_population=0.2,
    )
    return classical_features(loci, max_g=8, pihat_sizes=(2,))


def _parse_summary(path, label_count):
    out = {}
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        labels = tuple(parts[:label_count])
        g = int(parts[label_count])
        n = int(parts[label_count + 1])
        vals = tuple(float(x) for x in parts[label_count + 2:label_count + 5])
        out[(labels, g)] = (n, vals)
    return out


def _parse_full(path, label_count):
    out = {}
    with open(path) as fh:
        header = fh.readline().split()
        locus_ids = header[label_count + 2:-3]
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            labels = tuple(parts[:label_count])
            g = int(parts[label_count])
            n = int(parts[label_count + 1])
            vals = np.array([float(x) for x in parts[label_count + 2:label_count + 2 + n]])
            tail = tuple(float(x) for x in parts[label_count + 2 + n:label_count + 5 + n])
            out[(labels, g)] = (n, vals, tail)
    return locus_ids, out


def _key(labels, kind):
    pops = ["AMERICA", "EUROPE", "EAST_ASIA"]
    idx = {p: i + 1 for i, p in enumerate(pops)}
    if kind == "alpha":
        return f"alpha_{idx[labels[0]]}"
    if kind == "pi":
        return f"pi_{idx[labels[0]]}"
    return "pihat_" + "".join(str(idx[p]) for p in labels)


def _assert_close_tuple(got, expected):
    for a, b in zip(got, expected):
        assert math.isclose(a, b, abs_tol=TOL), (got, expected)


def test_original_adze_example_summary_outputs():
    res = _load_result()
    if res is None:
        return
    specs = [
        ("small_r", 1, "alpha"),
        ("small_p", 1, "pi"),
        ("small_c_2", 2, "pihat"),
    ]
    for filename, label_count, kind in specs:
        expected = _parse_summary(os.path.join(BASE, filename), label_count)
        for (labels, g), (n, vals) in expected.items():
            key = _key(labels, kind)
            assert n == res.n_loci
            assert (key, g) in res.summary
            _assert_close_tuple(res.summary[(key, g)], vals)


def test_original_adze_example_full_outputs():
    res = _load_result()
    if res is None:
        return
    specs = [
        ("small_r_fulldata", 1, "alpha"),
        ("small_p_fulldata", 1, "pi"),
        ("small_c_2_fulldata", 2, "pihat"),
    ]
    for filename, label_count, kind in specs:
        locus_ids, expected = _parse_full(os.path.join(BASE, filename), label_count)
        assert locus_ids == res.locus_ids
        for (labels, g), (n, vals, tail) in expected.items():
            key = _key(labels, kind)
            assert n == res.n_loci
            assert (key, g) in res.per_locus
            assert np.allclose(res.per_locus[(key, g)], vals, atol=TOL)
            _assert_close_tuple(res.summary[(key, g)], tail)


def test_cli_reproduces_original_adze_example_output_files():
    if not _require_original_example():
        return
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "small")
        rc = cli_main([
            "features",
            "--structure", DATA,
            "--structure-header-rows", "1",
            "--structure-meta-columns", "5",
            "--structure-pop-column", "5",
            "--structure-label-column", "1",
            "--structure-missing", "-9",
            "--max-missing-per-population", "0.2",
            "--max-depth", "8",
            "--pihat-sizes", "2",
            "--adze-prefix", prefix,
            "--adze-full",
        ])
        assert rc == 0

        for filename, label_count in [
            ("small_r", 1),
            ("small_p", 1),
            ("small_c_2", 2),
        ]:
            expected = _parse_summary(os.path.join(BASE, filename), label_count)
            got = _parse_summary(prefix + filename.removeprefix("small"), label_count)
            assert set(got) == set(expected)
            for key in expected:
                assert got[key][0] == expected[key][0]
                _assert_close_tuple(got[key][1], expected[key][1])

        for filename, label_count in [
            ("small_r_fulldata", 1),
            ("small_p_fulldata", 1),
            ("small_c_2_fulldata", 2),
        ]:
            expected_loci, expected = _parse_full(os.path.join(BASE, filename), label_count)
            got_loci, got = _parse_full(prefix + filename.removeprefix("small"), label_count)
            assert got_loci == expected_loci
            assert set(got) == set(expected)
            for key in expected:
                assert got[key][0] == expected[key][0]
                assert np.allclose(got[key][1], expected[key][1], atol=TOL)
                _assert_close_tuple(got[key][2], expected[key][2])

        expected_deleted = open(os.path.join(BASE, "small_p_deletedloci")).read()
        got_deleted = open(prefix + "_p_deletedloci").read()
        assert got_deleted == expected_deleted


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} original ADZE example tests passed")


if __name__ == "__main__":
    _main()
