#!/usr/bin/env python3
"""Optional download of the four pinned CARE-PD canonical files, on your explicit command only.

    python -I scripts/care_pd/fetch_canonical.py --first --data-dir /path/to/care_pd --accept-care-pd-terms
    python -I scripts/care_pd/schema_check.py --first --data-dir /path/to/care_pd
    python -I scripts/care_pd/fetch_canonical.py --rest --data-dir /path/to/care_pd --accept-care-pd-terms

Nothing is fetched without --accept-care-pd-terms. That flag records your own decision, after you have
read the terms named below, that you may use this data. It grants no rights, and this code cannot decide
that for you. A file is kept only if its size, SHA-256 and MD5 all match the pins in common.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

GATE_NAME = "first_file_gate.local.json"
TERMS_URL = "https://neurips2025.care-pd.ca/terms-of-use"
DATASET_URL = "https://huggingface.co/datasets/vida-adl/CARE-PD"
CHUNK = 1 << 20


def fetch(cohort, data_dir):
    name, size, sha, md5 = C.FILES[cohort]
    dest = data_dir / name
    if dest.exists():
        C.verify_file(dest, cohort)
        print(f"ALREADY_VERIFIED {name} {size} sha256={sha} md5={md5}")
        return
    part = dest.with_suffix(".part")
    req = urllib.request.Request(C.HF_BASE + name)
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
            for chunk in iter(lambda: resp.read(CHUNK), b""):
                n += len(chunk)
                if n > size:
                    out.close()
                    part.unlink()
                    raise C.SourceError(f"SIZE_EXCEEDED:{cohort}")
                out.write(chunk)
    except (urllib.error.URLError, TimeoutError) as e:
        part.unlink(missing_ok=True)
        raise C.SourceError(f"DOWNLOAD_FAILED:{cohort}:{type(e).__name__}") from None
    try:
        C.verify_file(part, cohort)
    except C.SourceError:
        part.unlink()
        raise
    os.replace(part, dest)
    print(f"VERIFIED {name} {size} sha256={sha} md5={md5}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download the pinned CARE-PD files after you accept their terms yourself.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--first", action="store_true", help="3DGait only (about 8 MB)")
    which.add_argument("--rest", action="store_true", help="the other three; needs a passed schema_check.py --first")
    parser.add_argument("--data-dir", required=True, type=Path, help="target folder; keep it out of version control")
    parser.add_argument("--accept-care-pd-terms", action="store_true", help=f"you have read and accept {TERMS_URL}")
    args = parser.parse_args(argv)
    if not args.accept_care_pd_terms:
        print("TERMS_NOT_ACCEPTED: nothing was downloaded.")
        print(f"Read the terms at {TERMS_URL} and the dataset page at {DATASET_URL}.")
        print("If you accept them and your use is permitted, repeat with --accept-care-pd-terms.")
        return 2
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.first:
        fetch(C.FIRST_COHORT, args.data_dir)
        return 0
    gate_path = args.data_dir / GATE_NAME
    if not gate_path.is_file():
        raise C.SourceError("FIRST_FILE_GATE_MISSING")
    if json.loads(gate_path.read_text(encoding="utf-8")).get("status") != "FIRST_FILE_SCHEMA_PASS":
        raise C.SourceError("FIRST_FILE_GATE_NOT_PASSED")
    for cohort in C.COHORTS:
        if cohort != C.FIRST_COHORT:
            fetch(cohort, args.data_dir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except C.SourceError as e:
        print(f"P0_SCHEMA_STOP {e}")
        sys.exit(20)
