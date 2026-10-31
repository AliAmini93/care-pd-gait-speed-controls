#!/usr/bin/env python3
"""Synthetic-only smoke tests for P0 (never reads real data).

Covers hostile pickles, features, metrics against scikit-learn, the published-count
floor arithmetic, LOCO isolation, label-blind scaling, bootstrap pairing, the verdict
table, the within-subject control, stop checks and output sanitization. All numbers
here are fabricated and are never evidence.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "care_pd"))
import common as C  # noqa: E402
import p0_core as P  # noqa: E402

FAILS = []
MARK = "SYNTHID"
FPS = {"3DGait": 30, "BMCLab": 25, "PD-GaM": 25, "T-SDU-PD": 30}


def check(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def refused(blob):
    try:
        C.safe_loads(blob)
    except C.SourceError:
        return True
    return False


class EvilSystem:
    def __reduce__(self):
        return (os.system, ("echo SHOULD_NOT_RUN",))


class EvilEval:
    def __reduce__(self):
        return (eval, ("1+1",))


class Custom:
    pass


def synth_objs(mode="signal", seed=0, T=30):
    """Cohorts with the exact expected subject and class counts. Labels are random or speed-driven."""
    rng = np.random.default_rng(seed)
    objs = {}
    for name in C.COHORTS:
        n_subj = C.EXPECTED_SUBJECTS[name]
        labels = np.repeat(np.arange(4), C.EXPECTED_CLASS_WALKS[name])
        rng.shuffle(labels)
        assign = np.concatenate([np.arange(n_subj), rng.integers(0, n_subj, labels.size - n_subj)])
        rng.shuffle(assign)
        cohort = {f"{MARK}_{name}_{i}": {} for i in range(n_subj)}
        for w, (lab, s) in enumerate(zip(labels, assign)):
            base = 1.4 - 0.4 * lab if mode == "signal" else 1.0
            v = max(0.05, base + rng.normal(0, 0.12))
            t = np.arange(T) / FPS[name]
            trans = np.column_stack([rng.normal(0, 1e-3, T), rng.normal(0, 1e-3, T), v * t]).astype(np.float32)
            pose = rng.normal(0, 0.1, (T, 72)).astype(np.float32)
            cohort[f"{MARK}_{name}_{s}"][f"{MARK}_w{w}"] = {
                "pose": pose, "trans": trans, "beta": np.zeros(10, dtype=np.float32),
                "fps": FPS[name], "UPDRS_GAIT": int(lab), "medication": None, "other": None,
            }
        objs[name] = cohort
    return objs


def test_pickle():
    for label, obj in (("os.system", EvilSystem()), ("eval", EvilEval()), ("custom class", Custom()),
                       ("object array", np.array([None, 1], dtype=object)),
                       ("datetime array", np.array(["2020"], dtype="datetime64[Y]"))):
        for proto in (2, 3, 4, 5):
            check(refused(pickle.dumps(obj, protocol=proto)), f"hostile refused: {label} protocol {proto}")
    check(refused(b"not a pickle"), "garbage refused")
    check(refused(b"cos\nsystem\n(S'echo x'\ntR."), "hand-written GLOBAL + REDUCE refused")
    good = {"s": {"w": {"pose": np.ones((3, 72), "f4"), "trans": np.ones((3, 3), "f8"),
                        "fps": np.int64(30), "UPDRS_GAIT": np.int32(2)}}}
    for proto in (3, 4, 5):
        got = C.safe_loads(pickle.dumps(good, protocol=proto))["s"]["w"]
        check(got["pose"].shape == (3, 72) and float(got["pose"][0, 0]) == 1.0,
              f"array round trip protocol {proto}")
        check(C.read_label(got["UPDRS_GAIT"]) == 2 and C.read_fps(got["fps"]) == 30.0,
              f"numpy scalars read protocol {proto}")
    for bad in (4, -1, 1.5, "1", True, float("nan")):
        try:
            C.read_label(bad)
            check(False, f"label {bad!r} refused")
        except C.SourceError:
            check(True, f"label {bad!r} refused")
    check(C.read_label(None) is None, "absent label stays absent")


def test_features():
    T, fps = 300, 30
    t = np.arange(T) / fps
    trans = np.column_stack([np.zeros(T), 50 * np.sin(t), 1.5 * t])
    s, d = P.speed_and_duration(trans, fps)
    check(abs(d - 10.0) < 1e-12, "duration is T/fps")
    check(abs(s - 1.5) < 0.1, "vertical y is ignored and forward speed is recovered")
    trans2 = np.column_stack([1.5 * t, np.zeros(T), np.zeros(T)])
    check(abs(P.speed_and_duration(trans2, fps)[0] - s) < 1e-9, "x and z enter symmetrically")
    rng = np.random.default_rng(1)
    noisy = np.column_stack([rng.normal(0, 0.02, T), np.zeros(T), 1.5 * t + rng.normal(0, 0.02, T)])
    check(P.speed_and_duration(noisy, fps)[0] < 1.7, "moving average removes frame jitter from the path")
    check(P.speed_and_duration(np.zeros((T, 3)), fps)[0] == 0.0, "still walk has zero speed")
    for f, w in ((30, 15), (150, 75), (25, 12)):
        check(max(1, round(f / 2)) == w, f"window at fps {f} is {w}")
    pose = rng.normal(size=(T, 72))
    stats_full = P.pose_stats(pose)
    check(stats_full.shape == (138,), "138 pose statistics")
    pose2 = pose.copy()
    pose2[:, :3] += 100 * rng.normal(size=(T, 3))
    check(np.array_equal(P.pose_stats(pose2), stats_full), "root orientation excluded")
    check(abs(stats_full[0] - pose[:, 3].std()) < 1e-12, "first statistic is SD of component 3")
    check(abs(stats_full[69] - (np.percentile(pose[:, 3], 95) - np.percentile(pose[:, 3], 5))) < 1e-12,
          "statistic 69 is p95-p5 of component 3")


def test_metrics():
    rng = np.random.default_rng(2)
    for trial, drop in ((0, None), (1, 2)):
        y = rng.integers(0, 3, 500)
        p = rng.integers(0, 3, 500)
        if drop is not None:
            y[y == drop] = 0
            p[p == drop] = 1
        conf = np.zeros((3, 3))
        np.add.at(conf, (y, p), 1)
        check(abs(P.macro_f1_conf(conf) - f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)) < 1e-12,
              f"macro-F1 matches scikit-learn (trial {trial})")
    want = {"3DGait": (0.318, 0.230), "PD-GaM": (0.317, 0.213), "BMCLab": (0.326, 0.203), "T-SDU-PD": (0.329, 0.203)}
    floors = []
    for name, (uniform, constant) in want.items():
        counts = C.EXPECTED_CLASS_WALKS[name][:3]
        floors.append(P.plugin_floor(counts))
        check(abs(floors[-1] - uniform) < 0.0015, f"documented floor reproduced for {name}")
        y = np.repeat(np.arange(3), counts)
        cm = np.zeros((3, 3))
        np.add.at(cm, (y, np.full(y.size, int(np.argmax(counts)))), 1)
        check(abs(P.macro_f1_conf(cm) - constant) < 0.0015, f"documented constant reproduced for {name}")
    check(abs(np.mean(floors) - 0.323) < 0.001, "documented mean floor reproduced")
    check(abs(P.plugin_floor((0, 10, 10)) - P.plugin_floor((10, 10, 0))) < 1e-12, "absent class scores 0 in the floor")
    y = np.repeat(np.arange(3), (5000, 4000, 1000))
    runs = []
    for _ in range(200):
        cm = np.zeros((3, 3))
        np.add.at(cm, (y, rng.integers(0, 3, y.size)), 1)
        runs.append(P.macro_f1_conf(cm))
    check(abs(np.mean(runs) - P.plugin_floor((5000, 4000, 1000))) < 0.003, "large-sample uniform runs agree with plug-in")


def test_loco_and_scaling(table):
    loco, _ = P.loco_predictions(table)
    for ci, name in enumerate(C.COHORTS):
        te = loco[name]["test_mask"]
        check(set(table["cohort"][te]) == {ci}, f"{name}: test holds only that cohort")
        check(len(np.unique(table["subject"][te])) > 0 and not set(table["subject"][te]) & set(
            table["subject"][np.isin(table["label"], (0, 1, 2)) & (table["cohort"] != ci)]),
            f"{name}: no subject in both train and test")
        check(not (table["label"][te] == 3).any(), f"{name}: class 3 excluded from test")
    t2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in table.items()}
    ci0 = table["cohort"] == 0
    elig = np.isin(t2["label"], (0, 1, 2)) & ci0
    t2["label"][elig] = (t2["label"][elig] + 1) % 3
    loco2, _ = P.loco_predictions(t2)
    check(all(np.array_equal(loco[C.COHORTS[0]]["pred"][a], loco2[C.COHORTS[0]]["pred"][a]) for a in P.ARMS),
          "changing held-out labels leaves that cohort's predictions unchanged")
    check(not np.array_equal(loco[C.COHORTS[1]]["proba"]["S"], loco2[C.COHORTS[1]]["proba"]["S"]),
          "changing a training cohort's labels does change another fold's model")
    sz = P.scale_within_cohort(table["speed"], table["cohort"])
    for ci in range(4):
        m = table["cohort"] == ci
        check(abs(np.median(sz[m])) < 1e-9 and abs(P.iqr(sz[m]) - 1) < 1e-9, f"cohort {ci}: Sz has median 0 and IQR 1")


def test_bootstrap():
    subj = np.repeat(np.arange(10), 5)
    y = np.tile([0, 1, 2, 0, 1], 10)
    conf = P.confusion_by_subject(y, y, subj, np.arange(10))
    check(conf.shape == (10, 3, 3) and conf.sum() == 50, "confusion by subject shape")
    counts = P.draw_counts([10, 4], 500, 123)
    check(all((c.sum(axis=1) == n).all() for c, n in zip(counts, (10, 4))), "each draw resamples n subjects")
    again = P.draw_counts([10, 4], 500, 123)
    check(all(np.array_equal(a, b) for a, b in zip(counts, again)), "bootstrap is deterministic for a seed")
    check(np.allclose(P.macro_f1_conf(P.apply_counts(counts[0], conf)), 1.0), "perfect arm stays at 1 in every draw")
    # A subject's walks move together, so a subject-driven outcome gives a wider CI than walks would.
    rng = np.random.default_rng(3)
    n_s, n_w = 12, 20
    subj = np.repeat(np.arange(n_s), n_w)
    y = rng.integers(0, 3, subj.size)
    right = rng.random(n_s) < 0.5
    pred = np.where(right[subj], y, (y + 1) % 3)
    conf = P.confusion_by_subject(y, pred, subj, np.arange(n_s))
    cluster = P.macro_f1_conf(P.apply_counts(P.draw_counts([n_s], 2000, 4)[0], conf))
    walk_conf = P.confusion_by_subject(y, pred, np.arange(y.size), np.arange(y.size))
    walks = P.macro_f1_conf(P.apply_counts(P.draw_counts([y.size], 2000, 4)[0], walk_conf))
    check(cluster.std() > 2 * walks.std(), "cluster bootstrap is wider than walk-level when errors cluster by subject")
    # A class held by one subject can vanish from a draw; it then scores 0 for arm and floor alike.
    y2 = np.array([0, 0, 1, 1, 2, 2])
    conf2 = P.confusion_by_subject(y2, y2, np.array([0, 0, 1, 1, 2, 2]), np.arange(3))
    only_first = np.array([[3, 0, 0]])
    check(abs(P.macro_f1_conf(P.apply_counts(only_first, conf2))[0] - 1 / 3) < 1e-12, "absent class scores 0 for the arm")
    check(abs(P.plugin_floor(P.apply_counts(only_first, conf2).sum(axis=2))[0] - P.plugin_floor((6, 0, 0))) < 1e-12,
          "absent class scores 0 for the floor in the same draw")


def test_verdict():
    cases = [((0.05, 0.001, 0.0, -0.1, 0, -1), "SPEED_FLOOR_HOLDS"),
             ((0.06, 0.01, 0.05, 0.001, 0, -1), "POSE_ADDS_BEYOND_SPEED"),
             ((0.01, -0.02, 0.09, 0.01, 0, -1), "POSE_ONLY_SIGNAL"),
             ((0.049999, 0.01, 0.049999, 0.01, 0, -1), "AT_FLOOR"),
             ((0.09, 0.0, 0.09, 0.0, 0, -1), "AT_FLOOR")]
    for args, want in cases:
        check(P.decide(*args)["verdict"] == want, f"verdict {want} from {args[:4]}")
    check(P.decide(0.01, -0.01, 0, -1, 0.08, 0.02)["wording_qualifier"] == "SPEED_SITE_DEPENDENT",
          "Sz qualifies wording when only Sz would pass")
    check(P.decide(0.01, -0.01, 0, -1, 0.08, 0.02)["verdict"] == "AT_FLOOR", "Sz does not change the verdict")
    check(P.decide(0.08, 0.02, 0, -1, 0.08, 0.02)["wording_qualifier"] is None, "no qualifier when S passes")


def test_control():
    rng = np.random.default_rng(5)
    subj, lab, sp, co = [], [], [], []
    for s in range(20):
        for w in range(6):
            lv = w % 3
            subj.append(s)
            lab.append(lv)
            sp.append(1.5 - 0.3 * lv + rng.normal(0, 0.01))
            co.append(s % 4)
    # Excluded on purpose: too few walks, and a single label.
    subj += [100, 100, 101, 101, 101]
    lab += [0, 1, 1, 1, 1]
    sp += [1.0, 0.5, 1.0, 0.9, 0.8]
    co += [0, 0, 1, 1, 1]
    t = {"subject": np.array(subj), "label": np.array(lab), "speed": np.array(sp), "cohort": np.array(co)}
    r = P.within_subject_control(t, 200)
    check(r["n_subjects"] == 20, "control keeps only subjects with >=3 walks and >=2 labels")
    check(r["n_negative"] == 20 and r["supports_negative"], "control detects a built-in negative relation")
    check(abs(r["sign_test_two_sided_p"] - 2 * 0.5 ** 20) < 1e-12, "exact two-sided sign test")
    t["speed"] = 1.5 + 0.3 * t["label"] + rng.normal(0, 0.01, t["label"].size)
    r2 = P.within_subject_control(t, 200)
    check(not r2["supports_negative"], "control does not support negative when the relation is positive")
    t["label"] = t["label"].copy()
    t["speed"] = np.ones(t["label"].size)
    r3 = P.within_subject_control(t, 200)
    check(r3["n_subjects"] == 0 and r3["n_undefined"] == 20 and not r3["supports_negative"],
          "constant speed is undefined, not evidence")


def stop_raised(table, **change):
    t = dict(table)
    t.update(change)
    try:
        P.check_stops(t)
    except C.SourceError:
        return True
    return False


def test_stops(table):
    check(not stop_raised(table), "matching synthetic counts pass the stop checks")
    lab = table["label"].copy()
    lab[table["label"] == 0] = 1
    check(stop_raised(table, label=lab), "wrong class counts stop")
    sub = table["subject"].copy()
    sub[table["cohort"] == 0] = sub[table["cohort"] == 0][0]
    check(stop_raised(table, subject=sub), "wrong subject count stops")
    spd = table["speed"].copy()
    spd[table["cohort"] == 2] = 0.0
    check(stop_raised(table, speed=spd), "low speed share stops")
    check(stop_raised(table, nonfinite={**table["nonfinite"], "BMCLab": 1}), "non-finite values stop")
    objs = synth_objs("null", seed=9)
    first = next(iter(objs["3DGait"].values()))
    next(iter(first.values())).pop("trans")
    try:
        P.build_table(objs)
        check(False, "missing field stops")
    except C.SourceError:
        check(True, "missing field stops")


def test_pipeline():
    objs = synth_objs("signal", seed=7)
    blob = {name: C.safe_loads(pickle.dumps(o, protocol=5 if i % 2 else 4))
            for i, (name, o) in enumerate(objs.items())}
    table = P.build_table(blob)
    del blob, objs
    test_stops(table)
    test_loco_and_scaling(table)
    res = P.run_analysis(table, n_boot=100)
    text = json.dumps(res)
    check(MARK not in text, "no subject or walk key reaches the output")
    check(len(text) < 200_000, "output stays under the summary size cap")
    check(res["totals"]["walks_all_labels"] == 2952 and res["totals"]["subjects_all"] == 110, "totals counted")
    check(res["totals"]["walks_eligible_012"] == 2908, "eligible walks counted")
    d = res["decision"]
    check(d["S1"] and res["paired_deltas"]["S_minus_F_U"]["M"]["point"] > 0.05, "a built-in speed signal passes S1")
    check(res["within_subject_control"]["supports_negative"], "a built-in speed signal supports the control")
    m = res["arms"]["S"]["M"]
    check(m["ci95"][0] <= m["point"] <= m["ci95"][1], "CI brackets the point estimate")
    check(abs(res["arms"]["F_U"]["M"]["point"] - 0.323) < 0.001, "F_U on synthetic counts equals the documented value")
    check(abs(res["reference_guessers_descriptive"]["observed_uniform_runs"]["M"]["mean"]
              - res["arms"]["F_U"]["M"]["point"]) < 0.015, "observed uniform runs sit near the plug-in")
    check(res["arms"]["C"]["M"]["point"] < res["arms"]["F_U"]["M"]["point"], "constant predictor scores below F_U")
    nul = P.build_table(synth_objs("null", seed=8))
    res0 = P.run_analysis(nul, n_boot=100)
    check(not res0["decision"]["S1"], "no built-in signal fails S1")
    check(not res0["within_subject_control"]["supports_negative"], "no built-in signal gives no control support")
    # A fitted no-signal arm may fall well below F_U, so only the upper side is a test.
    check(res0["paired_deltas"]["S_minus_F_U"]["M"]["point"] < 0.05, "no built-in signal does not clear the S1 margin")
    test_null_calibration(table, res, nul, res0)


def observed_of(res):
    return {a: res["arms"][a]["M"]["point"] for a in ("F_U",) + P.FITTED_ARMS}


def test_null_calibration(table, res, nul, res0):
    before = table["label"].copy()
    cal = P.null_calibration(table, observed_of(res), n_perm=12)
    check(np.array_equal(before, table["label"]), "calibration does not change the labels it is given")
    check(cal == P.null_calibration(table, observed_of(res), n_perm=12), "calibration is deterministic for its seed")
    check(cal["n_permutations_done"] == 12, "calibration runs the planned permutations")
    check(cal["S_minus_F_U"]["rank_p_one_sided"] == 1 / 13, "a built-in speed signal beats every permutation")
    check(cal["arms_M"]["S"]["null_mean"] < res["arms"]["S"]["M"]["point"] - 0.1, "permuting training labels removes the signal")
    cal0 = P.null_calibration(nul, observed_of(res0), n_perm=12)
    check(cal0["arms_M"]["S"]["rank_p_one_sided"] > 1 / 13, "no built-in signal does not beat every permutation")
    short = P.null_calibration(nul, observed_of(res0), n_perm=5, budget_s=-1)
    check(short["n_permutations_done"] == 0, "time budget stops the calibration")
    check(MARK not in json.dumps(cal), "calibration output holds no subject or walk key")


def main():
    test_pickle()
    test_features()
    test_metrics()
    test_bootstrap()
    test_verdict()
    test_control()
    test_pipeline()
    if FAILS:
        print(f"SMOKE_FAILED {len(FAILS)}")
        return 1
    print("SMOKE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
