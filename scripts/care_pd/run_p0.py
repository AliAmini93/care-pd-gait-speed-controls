#!/usr/bin/env python3
"""Run P0: one fixed logistic model, four leave-one-cohort-out folds, speed against pose plus speed.

    python -I scripts/care_pd/run_p0.py --data-dir /path/to/care_pd --out-dir /path/to/results

Writes aggregate metrics to <out-dir>/p0.json and prints nothing else of substance. No score
appears on screen before the result is complete. A completed result is never overwritten, so a
new run needs a new --out-dir. A stop file from a failed pre-score check does not block a retry.

Exit codes: 0 complete, 10 refused because a completed result exists, 20 a stop check failed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

# The recorded runs used one BLAS and OpenMP thread. Set before numpy loads; an explicit setting wins.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import p0_core as P  # noqa: E402

RESULT_NAME = "p0.json"
LIMITS = [
    "Four cohorts: CIs cover subject sampling inside them, not variation between sites.",
    "CIs hold predictions fixed and exclude refit variance.",
    "Walks are repeated measures, and some subjects have mixed walk labels.",
    "F_U is a plug-in expectation for a specified uniform guesser, not a universal floor.",
    "BMCLab is MoCap and the other three RGB; trans scale may differ between cohorts.",
    "138 pose statistics against few subjects, with fixed regularization and no tuning.",
    "The within-subject control is secondary and not causal.",
    "Not clinical. No diagnostic, prognostic or deployment claim. Nothing here shows encoders are at chance.",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the fixed-logistic P0 analysis on separately obtained CARE-PD files.")
    parser.add_argument("--data-dir", required=True, type=Path, help="folder holding the four pinned *_canonical.pkl files")
    parser.add_argument("--out-dir", required=True, type=Path, help="folder for p0.json; keep it out of version control")
    return parser.parse_args(argv)


def write_result(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def completed(path):
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") != "P0_SCHEMA_STOP"


def main(argv=None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / RESULT_NAME
    if completed(result_path):
        print(f"P0_REFUSED: {result_path} already holds a completed result; choose a new --out-dir")
        return 10
    provenance = {
        "git_commit": os.environ.get("RUN_COMMIT_SHA", "UNSET"),
        "files": {c: {"name": f[0], "bytes": f[1], "sha256": f[2], "md5": f[3]} for c, f in C.FILES.items()},
        "hf_commit": C.HF_COMMIT,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "UNSET"),
    }
    try:
        objs = {c: C.load_cohort(c, args.data_dir) for c in C.COHORTS}
        table = P.build_table(objs)
        del objs
        P.check_stops(table)
    except C.SourceError as e:
        write_result(result_path, {"status": "P0_SCHEMA_STOP", "failed_check": str(e), "provenance": provenance})
        print(f"P0_SCHEMA_STOP {e}")
        return 20
    result = P.run_analysis(table)
    result["provenance"] = provenance
    result["limits"] = LIMITS
    # The result is on disk first, so a failure in the descriptive part cannot lose it.
    result["null_calibration_descriptive"] = "NOT_COMPLETED"
    write_result(result_path, result)
    observed = {a: result["arms"][a]["M"]["point"] for a in ("F_U",) + P.FITTED_ARMS}
    result["null_calibration_descriptive"] = P.null_calibration(table, observed)
    write_result(result_path, result)
    print(f"P0_COMPLETE {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
