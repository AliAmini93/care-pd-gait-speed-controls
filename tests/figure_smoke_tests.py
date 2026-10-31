#!/usr/bin/env python3
"""Synthetic-only test of the optional figure scripts. Reports SKIP when their optional needs are absent.

Runs tests/cli_smoke_tests.py to produce fabricated aggregate JSON, draws each figure into a temporary
folder, and checks that repeating a run gives the same PDF bytes. The figures are fabricated and are
never evidence.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "scripts" / "figures"
OUTPUTS = {"generate_workflow_figure.py": ["workflow_protocol.pdf"],
           "generate_results_figure.py": ["results_primary.pdf", "results_sensitivity.pdf"]}


def skip_reason():
    if importlib.util.find_spec("matplotlib") is None:
        return "matplotlib is not installed"
    from matplotlib import font_manager
    for family in ("Liberation Sans", "Liberation Serif"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            return f"{family} is not installed"
    return None


def main():
    reason = skip_reason()
    if reason:
        print(f"FIGURE_SMOKE_SKIP {reason}")
        return 0
    fails = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        env = dict(os.environ, CARE_PD_SYNTHETIC_RESULTS_DIR=str(d / "json"), MPLCONFIGDIR=str(d / "mpl"))
        made = subprocess.run([sys.executable, "-I", str(REPO / "tests" / "cli_smoke_tests.py")], capture_output=True, text=True, env=env)
        if made.returncode != 0 or not (d / "json" / "p0.json").is_file():
            print(made.stdout[-2000:] + made.stderr[-2000:])
            print("FIGURE_SMOKE_FAILED could not make synthetic JSON")
            return 1
        for script, names in OUTPUTS.items():
            digests = []
            for run in ("a", "b"):
                out = d / run
                done = subprocess.run([sys.executable, "-I", str(FIGURES / script), "--p0-json", str(d / "json" / "p0.json"),
                                       "--p1-json", str(d / "json" / "p1_strong_models.json"), "--out-dir", str(out)],
                                      capture_output=True, text=True, env=env)
                if done.returncode != 0:
                    print(done.stderr[-2000:])
                    fails.append(f"{script} runs")
                    break
                digests.append([(out / n).read_bytes() for n in names])
            else:
                ok = digests[0] == digests[1] and all(b.startswith(b"%PDF") for b in digests[0])
                print(("PASS " if ok else "FAIL ") + f"{script} writes reproducible PDF output")
                if not ok:
                    fails.append(script)
    if fails:
        print(f"FIGURE_SMOKE_FAILED {len(fails)}")
        return 1
    print("FIGURE_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
