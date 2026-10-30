"""Fetch Sleep-EDF Expanded (sleep-cassette) recordings.

The canonical loader is ``mne.datasets.sleep_physionet.age.fetch_data``. The
physionet.org origin is often throttled (~50 kB/s), so we first pull the files
from PhysioNet's official open-data mirror on AWS S3 (same files, same
SHA-1), verify every file against the SHA-1 list shipped with MNE, and then
call ``fetch_data`` which finds them already in place and only re-checks them.

Usage: python scripts/download.py --subjects 0-19
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import urllib.request
from pathlib import Path

import mne
from mne.datasets.sleep_physionet import age

S3 = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette/"


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_range(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="0-19")
    args = ap.parse_args()
    subjects = parse_range(args.subjects)

    records = Path(age.__file__).with_name("age_records.csv")
    dest = Path(mne.get_config("MNE_DATA", str(Path.home() / "mne_data"))) / "physionet-sleep-data"
    dest.mkdir(parents=True, exist_ok=True)

    with open(records) as f:
        rows = [r for r in csv.DictReader(f) if int(r["subject"]) in subjects]
    for r in rows:
        out = dest / r["fname"]
        if out.exists() and sha1(out) == r["sha"]:
            continue
        print("downloading", r["fname"], flush=True)
        tmp = out.with_suffix(".part")
        urllib.request.urlretrieve(S3 + r["fname"], tmp)
        if sha1(tmp) != r["sha"]:
            tmp.unlink()
            raise RuntimeError(f"SHA-1 mismatch for {r['fname']}")
        os.replace(tmp, out)

    # Canonical MNE call: verifies hashes and returns the [PSG, Hypnogram] pairs.
    files = age.fetch_data(subjects=subjects, recording=[1, 2], on_missing="warn")
    print(f"{len(files)} recordings ready in {dest}")


if __name__ == "__main__":
    main()
