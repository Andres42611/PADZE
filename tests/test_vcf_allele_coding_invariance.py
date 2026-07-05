"""ADZE statistics are invariant to VCF allele-code relabeling.

VCF REF/ALT order is a coding convention. If two VCFs encode the same allele partitions
with different allele indexes, the aligned count columns are permuted but alpha/pi/pihat
must be unchanged.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from padze import classical_features, read_vcf  # noqa: E402


SAMPLES = ["A1", "A2", "B1", "B2", "C1", "C2"]
POPMAP = {s: s[0] for s in SAMPLES}


def _recode_gt(gt, mapping):
    out = []
    for token in gt.replace("|", "/").split("/"):
        out.append("." if token == "." else str(mapping[int(token)]))
    return "/".join(out)


def _write_vcf(path, loci):
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        fh.write("\t".join(SAMPLES) + "\n")
        for li, (ref, alt, gts) in enumerate(loci, 1):
            row = ["chr1", str(li), ".", ref, alt, ".", "PASS", ".", "GT"]
            fh.write("\t".join([*row, *gts]) + "\n")


def _summary(res):
    return {k: np.array(v, dtype=float) for k, v in res.summary.items()}


def test_vcf_ref_alt_and_multiallelic_permutation_invariant():
    base_loci = [
        ("A", "C", ["0/1", "0/0", "1/1", "0/1", "0/0", "1/1"]),
        ("A", "C,G", ["0/1", "2/2", "0/2", "1/1", "0/.", "1/2"]),
    ]
    swapped_biallelic = {0: 1, 1: 0}
    permuted_triallelic = {0: 2, 1: 0, 2: 1}
    recoded_loci = [
        ("C", "A", [_recode_gt(gt, swapped_biallelic) for gt in base_loci[0][2]]),
        ("C", "G,A", [_recode_gt(gt, permuted_triallelic) for gt in base_loci[1][2]]),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "base.vcf")
        recoded = os.path.join(tmp, "recoded.vcf")
        _write_vcf(base, base_loci)
        _write_vcf(recoded, recoded_loci)
        a = classical_features(read_vcf(base, POPMAP), max_g=4, pihat_sizes=(2,))
        b = classical_features(read_vcf(recoded, POPMAP), max_g=4, pihat_sizes=(2,))

    sa = _summary(a)
    sb = _summary(b)
    assert set(sa) == set(sb)
    for key in sa:
        assert np.allclose(sa[key], sb[key], equal_nan=True), key
