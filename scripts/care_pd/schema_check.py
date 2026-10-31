#!/usr/bin/env python3
"""Pre-metric schema check for the CARE-PD files. No label is linked to any feature here.

    python -I scripts/care_pd/schema_check.py --first --data-dir /path/to/care_pd
    python -I scripts/care_pd/schema_check.py --all --data-dir /path/to/care_pd

--first checks 3DGait only and writes the local record that fetch_canonical.py --rest requires.
--all checks all four cohorts with the documented stop checks.

Prints aggregate counts only. Never prints subject IDs, walk IDs or per-walk values.
Labels are only counted. Trans axis summaries are medians across a cohort's walks.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

GATE_NAME = "first_file_gate.local.json"


def cohort_summary(cohort, raw_dir):
    path = Path(raw_dir) / C.FILES[cohort][0]
    if not path.is_file():
        raise C.SourceError(f"FILE_MISSING:{cohort}")
    C.verify_file(path, cohort)
    data = path.read_bytes()
    proto, seen = C.scan_opcodes(data)
    obj = C.safe_loads(data)
    labels = Counter()
    frames, fps_values, extras = [], Counter(), Counter()
    array_dtypes, extent, net = Counter(), [], []
    nonfinite_pose = nonfinite_trans = zero_pose = 0
    subjects = set()
    field_types = Counter()
    for subject, rec in C.walks_of(obj):
        subjects.add(subject)
        lab = C.read_label(rec["UPDRS_GAIT"])
        labels["none" if lab is None else str(lab)] += 1
        fps_values[C.read_fps(rec["fps"])] += 1
        pose, trans = C.read_arrays(rec)
        frames.append(pose.shape[0])
        array_dtypes[f"pose:{rec['pose'].dtype.name},trans:{rec['trans'].dtype.name}"] += 1
        nonfinite_pose += int(not np.isfinite(pose).all())
        nonfinite_trans += int(not np.isfinite(trans).all())
        zero_pose += int(np.ptp(pose[:, 3:], axis=0).max() == 0)
        extent.append(np.ptp(trans, axis=0))
        net.append(np.abs(trans[-1] - trans[0]))
        for k in sorted(C.WALK_KEYS - {"pose", "trans"}):
            field_types[f"{k}:{type(rec[k]).__name__}"] += 1
        for k in set(rec) - C.WALK_KEYS:
            extras["extra_field_present"] += 1
    frames = np.asarray(frames)
    return {
        "cohort": cohort,
        "pickle_protocol": proto,
        "globals_used": [".".join(g) for g in seen],
        "subjects": len(subjects),
        "walks": int(frames.size),
        "labels": dict(sorted(labels.items())),
        "fps_values": {str(k): v for k, v in sorted(fps_values.items())},
        "frames_min_median_max": [int(frames.min()), float(np.median(frames)), int(frames.max())],
        "array_dtypes": dict(array_dtypes),
        "field_types": dict(sorted(field_types.items())),
        "extra_fields": dict(extras),
        "walks_nonfinite_pose": nonfinite_pose,
        "walks_nonfinite_trans": nonfinite_trans,
        "walks_constant_pose": zero_pose,
        "trans_median_extent_xyz": np.round(np.median(np.asarray(extent), axis=0), 4).tolist(),
        "trans_median_net_displacement_xyz": np.round(np.median(np.asarray(net), axis=0), 4).tolist(),
    }, subjects


def check_counts(s):
    """Cohort stop checks, plus eligibility of the documented schema."""
    cohort = s["cohort"]
    if s["subjects"] != C.EXPECTED_SUBJECTS[cohort]:
        raise C.SourceError(f"SUBJECT_COUNT:{cohort}")
    expected = C.EXPECTED_CLASS_WALKS[cohort]
    for k, want in zip("0123", expected):
        if abs(s["labels"].get(k, 0) - want) > C.CLASS_CELL_TOLERANCE:
            raise C.SourceError(f"CLASS_COUNT:{cohort}")
    if s["walks_nonfinite_pose"] or s["walks_nonfinite_trans"]:
        raise C.SourceError(f"NON_FINITE:{cohort}")
    if s["walks_constant_pose"]:
        raise C.SourceError(f"DEGENERATE_POSE:{cohort}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check the pinned CARE-PD files without linking any label to any feature.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--first", action="store_true", help="3DGait only; records the pass for fetch_canonical.py --rest")
    which.add_argument("--all", action="store_true", help="all four cohorts")
    parser.add_argument("--data-dir", required=True, type=Path, help="folder holding the *_canonical.pkl files")
    args = parser.parse_args(argv)
    cohorts = (C.FIRST_COHORT,) if args.first else C.COHORTS
    summaries, all_subjects = [], {}
    for cohort in cohorts:
        s, subjects = cohort_summary(cohort, args.data_dir)
        check_counts(s)
        summaries.append(s)
        all_subjects[cohort] = subjects
    out = {"status": "FIRST_FILE_SCHEMA_PASS" if len(cohorts) == 1 else "ALL_FILES_SCHEMA_PASS",
           "cohorts": summaries}
    if len(cohorts) > 1:
        total = sum(v for s in summaries for k, v in s["labels"].items() if k != "none")
        lo, hi = C.TOTAL_WALKS_RANGE
        if not lo <= total <= hi:
            raise C.SourceError("TOTAL_WALKS")
        names = list(all_subjects)
        out["total_labelled_walks_all_labels"] = total
        out["total_unlabelled_walks"] = sum(s["labels"].get("none", 0) for s in summaries)
        out["subject_id_collisions_across_cohorts"] = sum(
            len(all_subjects[a] & all_subjects[b])
            for i, a in enumerate(names) for b in names[i + 1:])
    else:
        (args.data_dir / GATE_NAME).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except C.SourceError as e:
        print(f"P0_SCHEMA_STOP {e}")
        sys.exit(20)
