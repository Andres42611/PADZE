# Data Acquisition

Raw empirical datasets are not committed to this repository. The ADZE paper's human
microsatellite analysis uses Rosenberg et al. 2005 HGDP-CEPH files hosted by the
Rosenberg lab.

Fetch the required upstream files into the ignored external-data directory:

```bash
python scripts/data/fetch_hgdp_rosenberg2005.py
```

The script writes to `data/external/hgdp_rosenberg2005/` and records local SHA-256
checksums in `manifest.json`. No official checksums were found on the Rosenberg lab page,
so these hashes are provenance records for the local download, not upstream guarantees.

Expected paper anchors for the follow-up comparison pipeline:

- H1048 source data: 1048 individuals and 783 autosomal microsatellite loci.
- H952 analysis subset: 952 individuals after excluding known close relatives.
- ADZE populations: `Af Eu Ea Oc Am`.
- Missingness filter: `TOLERANCE 0.15`, dropping loci only when a region has missingness
  strictly greater than 15%.
- Retained H952 loci after filtering: 721.
- Table 1 uncorrected distinct-allele total over retained H952 loci: 8516.

In the current sandbox, shell DNS resolution for `rosenberglab.stanford.edu` failed, so the
directory may exist without the raw files. Run the fetch command again in a networked
environment before attempting full ADZE-paper reproduction.
