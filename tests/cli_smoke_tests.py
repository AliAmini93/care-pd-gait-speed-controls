#!/usr/bin/env python3
"""Synthetic-only tests of the command-line entry points, from a fresh checkout (never reads real data).

Fabricated pickles are written to a temporary folder and the hash pins are redirected to them for this
process only. Runs P0 and then P1 through their real entry points, so P1's parity check reads a real P0
result file. Network access is replaced by a local stub. Every number here is fabricated and is never
evidence.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "care_pd"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import fetch_canonical as F  # noqa: E402
import p0_core as P  # noqa: E402
import p1_core as Q  # noqa: E402
import p1_smoke_tests as T1  # noqa: E402
import run_p0 as R0  # noqa: E402
import run_p1 as R1  # noqa: E402
import schema_check as S  # noqa: E402

FAILS = []
REAL_FILES = dict(C.FILES)


def check(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def pin_files(data_dir, objs):
    """Write fabricated cohort pickles and point the pins at them."""
    for cohort, (name, *_rest) in REAL_FILES.items():
        path = data_dir / name
        path.write_bytes(pickle.dumps(objs[cohort], protocol=5))
        C.FILES[cohort] = (name, *C.hash_file(path))


def cli(script, *args):
    return subprocess.run([sys.executable, "-I", str(SCRIPTS / script), *args], capture_output=True, text=True, cwd=REPO)


def test_fresh_checkout_commands():
    for script in ("run_p0.py", "run_p1.py", "schema_check.py", "fetch_canonical.py"):
        check(cli(script, "--help").returncode == 0, f"{script} --help runs from a fresh checkout")
        check(cli(script).returncode == 2, f"{script} refuses to guess a data location")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        done = cli("run_p0.py", "--data-dir", str(d / "absent"), "--out-dir", str(d / "out"))
        stop = json.loads((d / "out" / "p0.json").read_text())
        check(done.returncode == 20 and "FILE_MISSING" in done.stdout, "P0 stops on a missing file")
        check(stop["status"] == "P0_SCHEMA_STOP" and "arms" not in stop, "the P0 stop file holds no metric")
        done = cli("run_p1.py", "--data-dir", str(d / "absent"), "--out-dir", str(d / "out"))
        check(done.returncode == 20 and "P1_REQUIRES_COMPLETED_P0" in done.stdout, "P1 refuses before a completed P0")
        check(not (d / "out" / R1.LEDGER_NAME).exists(), "refusing P1 uses no attempt")
        done = cli("schema_check.py", "--all", "--data-dir", str(d / "absent"))
        check(done.returncode == 20 and "FILE_MISSING" in done.stdout, "the schema check stops on a missing file")
        (d / "evil").mkdir()
        (d / "evil" / REAL_FILES["3DGait"][0]).write_bytes(pickle.dumps({"x": 1}))
        done = cli("schema_check.py", "--first", "--data-dir", str(d / "evil"))
        check(done.returncode == 20 and "SIZE_MISMATCH" in done.stdout, "a file that misses the pins is never parsed")


def test_fetch():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        real_open = F.urllib.request.urlopen
        calls = []

        class Stub(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        def stub(request, timeout=None):
            calls.append(request.full_url)
            return Stub(BODY[request.full_url.rsplit("/", 1)[1]])

        F.urllib.request.urlopen = stub
        target = d / "fetched"
        try:
            check(quiet(F.main, ["--first", "--data-dir", str(target)]) == 2 and not calls and not target.exists(),
                  "no flag: nothing is fetched and nothing is created")
            BODY = {REAL_FILES[c][0]: f"fabricated {c}".encode() for c in C.COHORTS}
            for c in C.COHORTS:
                name = REAL_FILES[c][0]
                body = BODY[name]
                C.FILES[c] = (name, len(body), hashlib.sha256(body).hexdigest(), hashlib.md5(body).hexdigest())
            check(quiet(F.main, ["--first", "--data-dir", str(target), "--accept-care-pd-terms"]) == 0 and len(calls) == 1,
                  "with the flag: one pinned file is fetched")
            check(calls[0].startswith(C.HF_BASE) and C.HF_COMMIT in calls[0], "the URL is pinned to the recorded commit")
            try:
                quiet(F.main, ["--rest", "--data-dir", str(target), "--accept-care-pd-terms"])
                check(False, "--rest needs the first-file record")
            except C.SourceError as e:
                check(str(e) == "FIRST_FILE_GATE_MISSING" and len(calls) == 1, "--rest needs the first-file record")
            (target / F.GATE_NAME).write_text(json.dumps({"status": "FIRST_FILE_SCHEMA_PASS"}))
            check(quiet(F.main, ["--rest", "--data-dir", str(target), "--accept-care-pd-terms"]) == 0 and len(calls) == 4,
                  "--rest fetches the other three after the record")
            name = REAL_FILES["3DGait"][0]
            BODY[name] = b"Fabricated 3DGait"
            bad = d / "bad"
            try:
                quiet(F.main, ["--first", "--data-dir", str(bad), "--accept-care-pd-terms"])
                check(False, "a hash mismatch is refused")
            except C.SourceError as e:
                check("SHA256_MISMATCH" in str(e) and not list(bad.iterdir()), "a hash mismatch keeps no file")
        finally:
            F.urllib.request.urlopen = real_open
            C.FILES.update(REAL_FILES)


def test_pipeline():
    objs = T1.synth_objs("speed", seed=41)
    table = Q.build_table(objs)
    saved = (P.run_analysis, P.null_calibration, Q.run_analysis, Q.finish_with_null, Q.check_stops, Q.check_versions)
    real = saved
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        data, out = d / "data", d / "out"
        data.mkdir()
        pin_files(data, objs)
        argv = ["--data-dir", str(data), "--out-dir", str(out)]
        P.run_analysis = lambda t: real[0](t, n_boot=20)
        P.null_calibration = lambda t, o: real[1](t, o, n_perm=2)
        Q.run_analysis = lambda t: real[2](t, configs=T1.FAST, n_boot=20)
        Q.finish_with_null = lambda t, r, s: real[3](t, r, s, n_null=3, null_budget_s=60)
        Q.check_stops = lambda t: real[4](t, T1.totals_of(table))
        try:
            check(quiet(S.main, ["--first", "--data-dir", str(data)]) == 0 and (data / S.GATE_NAME).is_file(), "schema check passes and records the first file")
            check(quiet(S.main, ["--all", "--data-dir", str(data)]) == 0, "schema check passes on all four synthetic files")
            check(quiet(R0.main, argv) == 0, "P0 completes through its entry point")
            p0_file = out / R0.RESULT_NAME
            p0_text = p0_file.read_text()
            p0 = json.loads(p0_text)
            check(p0["status"] == "P0_COMPLETE" and T1.MARK not in p0_text, "P0 result holds no subject or walk key")
            check(isinstance(p0["null_calibration_descriptive"], dict) and p0["provenance"]["hf_commit"] == C.HF_COMMIT, "P0 records its null and provenance")
            check("omp_num_threads" in p0["provenance"], "P0 records its thread setting")
            check(quiet(R0.main, argv) == 10, "a completed P0 result is not overwritten")
            check(quiet(R1.main, argv) == 0, "P1 completes, and its parity check accepts the real P0 result")
            p1_text = (out / R1.RESULT_NAME).read_text()
            p1 = json.loads(p1_text)
            check(p1["status"] == "P1_COMPLETE" and T1.MARK not in p1_text, "P1 result holds no subject or walk key")
            check(p1["provenance"]["p0_result_sha256"] == C.hash_file(p0_file)[1], "P1 records which P0 result it depends on")
            check(all(abs(p1["p0_parity_M"][a] - p0["arms"][a]["M"]["point"]) < 1e-6 for a in Q.PARITY_ARMS), "P1 reproduces P0 through its own path")
            check(p0_file.read_text() == p0_text, "P1 leaves the P0 result unchanged")
            check(quiet(R1.main, argv) == 10, "a scored P1 is not repeated")
            for key in ("historical_context", "published_unpaired_context"):
                check(key not in p1 and key not in p0, f"results carry no {key}")
            again = d / "out2"
            altered = json.loads(p0_text)
            altered["arms"]["S"]["M"]["point"] += 0.001
            again.mkdir()
            (again / R0.RESULT_NAME).write_text(json.dumps(altered))
            argv2 = ["--data-dir", str(data), "--out-dir", str(again)]
            check(quiet(R1.main, argv2) == 20, "a tampered P0 result stops P1 before any new metric")
            stop = (again / R1.RESULT_NAME).read_text()
            check("macro_f1" not in stop and "P0_PARITY:S" in stop, "the parity stop file holds no metric")
            check(Q.ledger_states(again / R1.LEDGER_NAME) == ["START", "PRE_SCORE_STOP"], "one pre-score attempt is recorded")
            write_figure_inputs = os.environ.get("CARE_PD_SYNTHETIC_RESULTS_DIR")
            if write_figure_inputs:
                Path(write_figure_inputs).mkdir(parents=True, exist_ok=True)
                (Path(write_figure_inputs) / R0.RESULT_NAME).write_text(p0_text)
                (Path(write_figure_inputs) / R1.RESULT_NAME).write_text(p1_text)
        finally:
            P.run_analysis, P.null_calibration, Q.run_analysis, Q.finish_with_null, Q.check_stops, Q.check_versions = saved
            C.FILES.update(REAL_FILES)


def main():
    test_fresh_checkout_commands()
    test_fetch()
    test_pipeline()
    if FAILS:
        print(f"CLI_SMOKE_FAILED {len(FAILS)}")
        for f in FAILS:
            print("  " + f)
        return 1
    print("CLI_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
