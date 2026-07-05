"""Executable checks for the local HGDP ADZE 1.0 vs PADZE comparison."""
import importlib.util
import json
import os
import sys

try:
    import pytest
except Exception:  # pragma: no cover - standalone fallback
    pytest = None

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, ".."))
COMP = os.path.join(REPO, "data", "competition")
COMPARE = os.path.join(COMP, "compare_tools.py")

EXPECTED_ROWS_BY_PREFIX = {
    "out_H952": {
        "_r": 195,
        "_p": 195,
        "_c_2": 390,
        "_c_3": 390,
        "_c_4": 195,
        "_c_5": 39,
    },
    "out_H1048": {
        "_r": 195,
        "_p": 195,
        "_c_2": 390,
        "_c_3": 390,
        "_c_4": 195,
        "_c_5": 39,
    },
    "out_race_H952": {
        "_r": 235,
        "_p": 235,
        "_c_2": 470,
        "_c_3": 470,
        "_c_4": 235,
        "_c_5": 47,
    },
    "out_race_H1048": {
        "_r": 235,
        "_p": 235,
        "_c_2": 470,
        "_c_3": 470,
        "_c_4": 235,
        "_c_5": 47,
    },
}
EXPECTED_TOTAL_ROWS = sum(
    sum(prefix_rows.values()) for prefix_rows in EXPECTED_ROWS_BY_PREFIX.values()
)


def _skip(reason):
    if pytest is not None:
        pytest.skip(reason)
    print(f"SKIP test_hgdp_competition_outputs: {reason}")
    raise SystemExit(0)


def _compare_module():
    if not os.path.exists(COMPARE):
        _skip("data/competition comparison artifacts are absent")
    spec = importlib.util.spec_from_file_location("adze_compare_tools", COMPARE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_outputs(compare):
    missing = []
    for prefix, suffix_rows in EXPECTED_ROWS_BY_PREFIX.items():
        for suffix in suffix_rows:
            fname = f"{prefix}{suffix}"
            if not (compare.ADZE1 / fname).exists():
                missing.append(str(compare.ADZE1 / fname))
            if not (compare.PY / fname).exists():
                missing.append(str(compare.PY / fname))
    for path in (
        compare.ADZE1 / "out_H952_p_deletedloci",
        compare.PY / "out_H952_p_deletedloci",
    ):
        if not path.exists():
            missing.append(str(path))
    if missing:
        _skip("H952 competition output artifacts are absent")


def _deleted_count(path):
    with open(path) as fh:
        first = fh.readline().split()[0]
    return int(first)


def test_hgdp_competition_output_discovery_covers_regular_and_race_panels():
    compare = _compare_module()
    _require_outputs(compare)
    files = compare.available_files()
    assert {(f.prefix, f.suffix) for f in files} == {
        (prefix, suffix)
        for prefix, suffix_rows in EXPECTED_ROWS_BY_PREFIX.items()
        for suffix in suffix_rows
    }


def test_hgdp_competition_outputs_match_at_print_precision():
    compare = _compare_module()
    _require_outputs(compare)
    total = 0
    results, _worst = compare.compare_all()
    by_name = {result.file.name: result for result in results}
    for prefix, suffix_rows in EXPECTED_ROWS_BY_PREFIX.items():
        for suffix, expected_rows in suffix_rows.items():
            fname = f"{prefix}{suffix}"
            assert fname in by_name
            result = by_name[fname]
            assert result.only_adze1 == 0, fname
            assert result.only_py == 0, fname
            assert result.rows == expected_rows, fname
            assert max(result.max_mean, result.max_var, result.max_se) <= compare.TOL, (
                fname,
                result,
            )
            total += result.rows
    assert total == EXPECTED_TOTAL_ROWS
    assert compare.main() == 0


def test_hgdp_h952_parse_contract_remains_keyed_by_region_set_and_g():
    compare = _compare_module()
    _require_outputs(compare)
    for fname, expected_rows in {
        f"out_H952{suffix}": expected
        for suffix, expected in EXPECTED_ROWS_BY_PREFIX["out_H952"].items()
    }.items():
        k = compare.FILES[fname]
        adze1 = compare.parse(compare.ADZE1 / fname, k)
        py = compare.parse(compare.PY / fname, k)
        assert set(adze1) == set(py), fname
        assert len(adze1) == expected_rows, fname
        max_abs = max(abs(x - y) for key in adze1
                      for x, y in zip(adze1[key], py[key]))
        assert max_abs <= compare.TOL, (fname, max_abs)


def test_hgdp_h952_anchor_counts_present():
    compare = _compare_module()
    _require_outputs(compare)
    manifest = compare.BASE / "canonical" / "manifest.json"
    if not manifest.exists():
        _skip("canonical manifest is absent")
    data = json.loads(manifest.read_text())
    verification = data["verification"]
    assert verification["n_loci"] == 783
    assert verification["n_H952"] == 952
    assert verification["region5_H952_counts"] == {
        "Af": 105, "Am": 64, "Ea": 232, "Eu": 523, "Oc": 28,
    }

    adze_deleted = _deleted_count(compare.ADZE1 / "out_H952_p_deletedloci")
    py_deleted = _deleted_count(compare.PY / "out_H952_p_deletedloci")
    assert adze_deleted == py_deleted == 62
    assert verification["n_loci"] - adze_deleted == 721

    build_log = (compare.PY / "build.log").read_text()
    assert "distinct alleles over retained loci: 8516" in build_log
