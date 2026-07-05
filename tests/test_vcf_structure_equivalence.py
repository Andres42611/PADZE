"""VCF and STRUCTURE readers must agree on identical genotypes.

Original ADZE accepts STRUCTURE only; PADZE's main upgrade path is direct VCF input.
These tests lock that upgrade path to the same aligned allele-count loci and classical
statistics when both formats encode the same diploid data, including missing calls and
multiallelic loci.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from padze import classical_features, read_structure, read_vcf  # noqa: E402


POPS = ["A", "B", "C", "D"]
SAMPLES = [f"{p}{i}" for p in POPS for i in (1, 2)]

# Each locus lists one GT per sample in SAMPLES order. Allele indexes are VCF-style and
# STRUCTURE tokens use the same strings, so count matrices should match exactly.
GENOTYPES = [
    ["0/1", "2/2", "0/2", "1/1", "0/0", "1/2", "0/1", "2/2"],
    ["0/.", "1/2", "./.", "0/1", "1/1", "2/.", "0/2", "2/2"],
    ["0/1", "2/3", "0/3", "1/2", "3/3", "0/2", "1/3", "0/0"],
]


def _alleles(gt):
    return gt.split("/")


def _write_vcf(path):
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        fh.write("\t".join(SAMPLES) + "\n")
        alts = ["C,G", "C,G", "C,G,T"]
        for li, gts in enumerate(GENOTYPES, 1):
            row = ["chr1", str(li), ".", "A", alts[li - 1], ".", "PASS", ".", "GT"]
            fh.write("\t".join([*row, *gts]) + "\n")


def _write_popmap(path):
    with open(path, "w") as fh:
        for sample in SAMPLES:
            fh.write(f"{sample} {sample[0]}\n")


def _write_structure_gene_copy(path):
    with open(path, "w") as fh:
        fh.write(" ".join(f"L{i + 1}" for i in range(len(GENOTYPES))) + "\n")
        for si, sample in enumerate(SAMPLES):
            for hap in range(2):
                vals = []
                for gts in GENOTYPES:
                    token = _alleles(gts[si])[hap]
                    vals.append("-9" if token == "." else token)
                fh.write(f"{sample}_{hap} {sample[0]} " + " ".join(vals) + "\n")


def _write_structure_one_row(path):
    with open(path, "w") as fh:
        fh.write(" ".join(f"L{i + 1}" for i in range(len(GENOTYPES))) + "\n")
        for si, sample in enumerate(SAMPLES):
            vals = []
            for gts in GENOTYPES:
                vals.extend("-9" if token == "." else token for token in _alleles(gts[si]))
            fh.write(f"{sample} {sample[0]} " + " ".join(vals) + "\n")


def _summary_tuple(res):
    return {k: tuple(float(x) for x in v) for k, v in res.summary.items()}


def test_vcf_matches_structure_count_matrices_and_classical_stats():
    with tempfile.TemporaryDirectory() as tmp:
        vcf = os.path.join(tmp, "data.vcf")
        popmap = os.path.join(tmp, "data.popmap")
        stru_gene = os.path.join(tmp, "gene_copy.stru")
        stru_one = os.path.join(tmp, "one_row.stru")
        _write_vcf(vcf)
        _write_popmap(popmap)
        _write_structure_gene_copy(stru_gene)
        _write_structure_one_row(stru_one)

        v = read_vcf(vcf, popmap, population_order=POPS, min_populations_genotyped=1)
        s_gene = read_structure(stru_gene, header_rows=1, ploidy=1,
                                population_order=POPS)
        s_one = read_structure(stru_one, header_rows=1, meta_columns=2, ploidy=2,
                               one_row_per_individual=True, population_order=POPS)

    assert v.populations == s_gene.populations == s_one.populations == POPS
    assert np.array_equal(v.sample_sizes, s_gene.sample_sizes)
    assert np.array_equal(v.sample_sizes, s_one.sample_sizes)
    assert len(v.count_matrices) == len(s_gene.count_matrices) == len(s_one.count_matrices)
    for vc, sg, so in zip(v.count_matrices, s_gene.count_matrices, s_one.count_matrices):
        assert np.array_equal(vc, sg)
        assert np.array_equal(vc, so)

    kwargs = dict(max_g=4, pihat_sizes=(2, 3, 4))
    assert _summary_tuple(classical_features(v, **kwargs)) == _summary_tuple(
        classical_features(s_gene, **kwargs))
    assert _summary_tuple(classical_features(v, **kwargs)) == _summary_tuple(
        classical_features(s_one, **kwargs))
