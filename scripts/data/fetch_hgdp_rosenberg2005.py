#!/usr/bin/env python3
"""Fetch the HGDP microsatellite files needed for ADZE paper reproduction.

The ADZE paper uses the HGDP-CEPH microsatellite data analyzed by Rosenberg et al. (2005).
Rosenberg lab hosts these public text files and recommends using the versions on its site
when comparing against previous lab analyses.

This script intentionally downloads into ``data/external/hgdp_rosenberg2005`` rather than
committing the data files to the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://rosenberglab.stanford.edu/data"

FILES = {
    "combinedmicrosats-1048.stru": (
        f"{BASE_URL}/rosenbergEtAl2005/combinedmicrosats-1048.stru"
    ),
    "rosenbergEtAl2005.codes.txt": (
        f"{BASE_URL}/rosenbergEtAl2005/rosenbergEtAl2005.codes.txt"
    ),
    "rosenbergEtAl2005.coordinates.txt": (
        f"{BASE_URL}/rosenbergEtAl2005/rosenbergEtAl2005.coordinates.txt"
    ),
    "rosenbergEtAl2005.readme.txt": (
        f"{BASE_URL}/rosenbergEtAl2005/rosenbergEtAl2005.readme.txt"
    ),
    "SampleInformation.txt": (
        f"{BASE_URL}/rosenberg2006ahg/SampleInformation.txt"
    ),
}


def _download(url: str, dest: Path) -> dict:
    h = hashlib.sha256()
    n = 0
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                n += len(chunk)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return {"url": url, "path": str(dest), "bytes": n, "sha256": h.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/external/hgdp_rosenberg2005")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "Rosenberg lab HGDP-CEPH diversity panel data page",
        "source_page": "https://rosenberglab.stanford.edu/diversity.html",
        "files": {},
    }
    for name, url in FILES.items():
        dest = out_dir / name
        if dest.exists() and not args.force:
            data = dest.read_bytes()
            manifest["files"][name] = {
                "url": url,
                "path": str(dest),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "status": "already_present",
            }
            continue
        meta = _download(url, dest)
        meta["status"] = "downloaded"
        manifest["files"][name] = meta

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
