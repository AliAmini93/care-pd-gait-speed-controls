#!/usr/bin/env python3
"""Run P1: six candidates chosen with source cohorts only, for the speed and pose-plus-speed arms alike.

    python -I scripts/care_pd/run_p1.py --data-dir /path/to/care_pd --out-dir /path/to/results

P0 must have completed in the same --out-dir. Before any new metric exists, P1 refits the P0 model
through its own path and stops unless it reproduces the P0 result. P1 reuses P0's participants after P0's
results were seen, so it cannot replicate P0 independently.

Writes aggregate metrics to <out-dir>/p1_strong_models.json and prints nothing else of substance.
<out-dir>/p1_attempts.local.tsv allows one more attempt after a pre-score stop and refuses any
attempt once scoring has started. A new run needs a new --out-dir, with its own P0 result.

Exit codes: 0 complete, 10 refused, 20 a stop check failed or the P0 result is missing.
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
import p1_core as Q  # noqa: E402

RESULT_NAME = "p1_strong_models.json"
P0_RESULT_NAME = "p0.json"
LEDGER_NAME = "p1_attempts.local.tsv"
LIMITS = [
    "Four cohorts: CIs cover subject sampling inside them, not unseen sites.",
    "CIs hold predictions fixed and exclude refit variance; see refit_seed_sensitivity.",
    "Gait items that may underlie these labels (MDS-UPDRS 3.10, 1987 UPDRS item 29) mention speed, so some agreement between speed and label is expected by construction. The item behind each cohort's labels is not settled.",
    "Only the primary contrast is confirmatory for the screen. All other contrasts are exploratory and unadjusted.",
    "The null fixes the real selected configs and does not repeat selection in each draw.",
    "trans units are unverified. BMCLab is MoCap; the other three are RGB.",
    "Negative results here say nothing about pretrained encoders or learned temporal models, which were not run.",
    "Not clinical. No diagnostic, prognostic or deployment claim.",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the source-selected P1 analysis after a completed P0 run.")
    parser.add_argument("--data-dir", required=True, type=Path, help="folder holding the four pinned *_canonical.pkl files")
    parser.add_argument("--out-dir", required=True, type=Path, help="the folder holding p0.json; p1_strong_models.json goes here")
    return parser.parse_args(argv)


def rounded(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v) for v in obj]
    return obj


def write_result(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rounded(obj), indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_p0(path):
    """The completed P0 result, or None. P1 never writes it."""
    if not path.is_file():
        return None
    p0 = json.loads(path.read_text(encoding="utf-8"))
    return p0 if p0.get("status") == "P0_COMPLETE" else None


def completed(path):
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") != "P1_PRE_SCORE_STOP"


def main(argv=None):
    args = parse_args(argv)
    result_path = args.out_dir / RESULT_NAME
    p0_path = args.out_dir / P0_RESULT_NAME
    ledger = args.out_dir / LEDGER_NAME
    p0 = read_p0(p0_path)
    if p0 is None:
        print(f"P1_REQUIRES_COMPLETED_P0: run run_p0.py first; no completed result at {p0_path}")
        return 20
    commit = os.environ.get("RUN_COMMIT_SHA", "UNSET")
    refusal = Q.ledger_refusal(Q.ledger_states(ledger))
    if completed(result_path) or refusal:
        print(f"P1_REFUSED: {'RESULT_EXISTS' if completed(result_path) else refusal}; choose a new --out-dir")
        return 10
    Q.ledger_append(ledger, commit, "START")
    provenance = {
        "git_commit": commit,
        "p0_result_sha256": C.hash_file(p0_path)[1],
        "files": {c: {"name": f[0], "bytes": f[1], "sha256": f[2], "md5": f[3]} for c, f in C.FILES.items()},
        "hf_commit": C.HF_COMMIT,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "UNSET"),
    }
    try:
        Q.check_versions(np.__version__, scipy.__version__, sklearn.__version__)
        objs = {c: C.load_cohort(c, args.data_dir) for c in C.COHORTS}
        table = Q.build_table(objs)
        del objs
        Q.check_stops(table)
        parity = Q.parity_check(table, {a: p0["arms"][a]["M"]["point"] for a in Q.PARITY_ARMS})
    except C.SourceError as e:
        Q.ledger_append(ledger, commit, "PRE_SCORE_STOP")
        write_result(result_path, {"status": "P1_PRE_SCORE_STOP", "failed_check": str(e), "provenance": provenance})
        print(f"P1_PRE_SCORE_STOP {e}")
        return 20
    Q.ledger_append(ledger, commit, "SCORED")
    result, state = Q.run_analysis(table)
    result["p0_parity_M"] = parity
    result["provenance"] = provenance
    result["limits"] = LIMITS
    # The arms are on disk first, so a failure in the null cannot lose them.
    result["null_subject_block"] = "NOT_COMPLETED"
    result["decision"] = "NOT_COMPLETED"
    write_result(result_path, result)
    write_result(result_path, Q.finish_with_null(table, result, state))
    print(f"P1_COMPLETE {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
