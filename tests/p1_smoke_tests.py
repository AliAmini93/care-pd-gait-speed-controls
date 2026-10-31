#!/usr/bin/env python3
"""Synthetic-only tests for P1 (never reads real data).

Covers features, frozen configs, weights, fold isolation, source-only selection, class
order and missing classes, scaling, bootstrap pairing, verdict branches, null labels,
parity and stop checks, the ledger, and output sanitization. Every number here is
fabricated and is never evidence.
"""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "care_pd"))
import common as C  # noqa: E402
import p0_core as P  # noqa: E402
import p1_core as Q  # noqa: E402

FAILS = []
MARK = "SYNTHID"
FPS = {"3DGait": 30, "BMCLab": 150, "PD-GaM": 25, "T-SDU-PD": 30}
# Small clones of the frozen configs, so pipeline tests run in minutes. The real ones are tested separately.
FAST = tuple(
    (n, f, dict(p, n_estimators=10) if f == "RF" else (dict(p, max_iter=8) if f == "HGB" else p))
    for n, f, p in Q.CONFIGS)


def totals_of(t):
    """Synthetic subjects with only class-3 walks vary by seed, so the subject total is read from the table."""
    eligible = np.isin(t["label"], (0, 1, 2))
    return {"walks_all": 2952, "walks_eligible": 2908, "subjects_eligible": len(np.unique(t["subject"][eligible]))}


def check(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def raises_source(fn):
    try:
        fn()
    except C.SourceError:
        return True
    return False


def synth_objs(mode, seed=0):
    """Fabricated cohorts with the exact expected subject and class counts.

    `speed`: labels drive forward speed only. `pose`: labels drive knee amplitude only, and
    speed is pure noise. `null`: labels drive nothing.
    """
    rng = np.random.default_rng(seed)
    objs = {}
    for name in C.COHORTS:
        n_subj, fps = C.EXPECTED_SUBJECTS[name], FPS[name]
        labels = np.repeat(np.arange(4), C.EXPECTED_CLASS_WALKS[name])
        rng.shuffle(labels)
        assign = np.concatenate([np.arange(n_subj), rng.integers(0, n_subj, labels.size - n_subj)])
        rng.shuffle(assign)
        cohort = {f"{MARK}_{name}_{i}": {} for i in range(n_subj)}
        T = 2 * fps
        t = np.arange(T) / fps
        for w, (lab, s) in enumerate(zip(labels, assign)):
            v = 1.4 - 0.4 * lab + rng.normal(0, 0.12) if mode == "speed" else 1.0 + rng.normal(0, 0.12)
            trans = np.column_stack([rng.normal(0, 1e-3, T), rng.normal(0, 1e-3, T), max(0.05, v) * t]).astype(np.float32)
            pose = rng.normal(0, 0.01, (T, 72))
            knee = (0.9 - 0.3 * lab + rng.normal(0, 0.06)) if mode == "pose" else 0.6 + rng.normal(0, 0.06)
            pose[:, 3 * 4] += knee * np.sin(2 * np.pi * t)
            pose[:, 3 * 5] += knee * np.sin(2 * np.pi * t + np.pi)
            cohort[f"{MARK}_{name}_{s}"][f"{MARK}_w{w}"] = {
                "pose": pose.astype(np.float32), "trans": trans, "beta": np.zeros(10, dtype=np.float32),
                "fps": fps, "UPDRS_GAIT": int(lab), "medication": None, "other": None}
        objs[name] = cohort
    return objs


def test_speed_alt():
    for fps, w in ((30, 15), (150, 75), (25, 12)):
        vals, shrunk = [], []
        for seconds in (2, 6, 20):
            T = seconds * fps
            trans = np.column_stack([np.zeros(T), np.zeros(T), 1.2 * np.arange(T) / fps])
            vals.append(Q.speed_alt(trans, fps))
            shrunk.append(P.speed_and_duration(trans, fps)[0])
        check(max(abs(v - 1.2) for v in vals) < 1e-9, f"S_alt recovers constant speed at any length, fps {fps}")
        check(shrunk[0] < shrunk[-1] - 0.05, f"P0 S shrinks on short walks at fps {fps}, unlike S_alt")
    T, fps = 300, 30
    t = np.arange(T) / fps
    z = 1.0 * t
    z[:40], z[-40:] = z[40], z[-40]
    trans = np.column_stack([np.zeros(T), 50 * np.sin(t), z])
    check(abs(Q.speed_alt(trans, fps) - 1.0) < 1e-6, "S_alt ignores start and stop segments and vertical motion")
    check(raises_source(lambda: Q.speed_alt(np.zeros((15, 3)), 30)), "S_alt stops when the walk is shorter than the window")
    check(abs(Q.speed_alt(np.zeros((16, 3)), 30)) == 0.0, "S_alt is defined at window length + 1")


def walking_pose(T, fps, knee=0.6, stride_hz=1.0):
    t = np.arange(T) / fps
    pose = np.zeros((T, 72))
    for joint, phase in ((1, 0.0), (2, np.pi), (4, 0.0), (5, np.pi)):
        pose[:, 3 * joint] = knee * (1 + np.sin(2 * np.pi * stride_hz * t + phase))
    return pose, t


def test_kinematics():
    check(len(Q.KIN_NAMES) == 21 and len(set(Q.KIN_NAMES)) == 21, "K has 21 distinct named features")
    fps = 30
    pose, t = walking_pose(10 * fps, fps)
    trans = np.column_stack([np.zeros(t.size), np.zeros(t.size), 1.2 * t])
    k = Q.kinematic_summary(pose, trans, fps)
    named = dict(zip(Q.KIN_NAMES, k))
    check(k.shape == (21,) and np.isfinite(k).all(), "K shape and finiteness")
    check(abs(named["stride_freq_hz"] - 1.0) <= 0.05, "stride frequency recovered from hip alternation")
    check(named["stride_regularity"] > 0.5, "a periodic walk scores high regularity")
    check(named["asym_knee"] < 1e-6 and named["asym_hip"] < 1e-6, "symmetric joints give zero asymmetry")
    check(abs(named["amp_knee"] - (np.percentile(0.6 * (1 + np.sin(2 * np.pi * t)), 95)
                                   - np.percentile(0.6 * (1 + np.sin(2 * np.pi * t)), 5))) < 1e-9, "amplitude is p95-p5 of the angle")
    k2 = Q.kinematic_summary(pose, np.column_stack([np.zeros(t.size), np.zeros(t.size), 2.4 * t]), fps)
    check(np.allclose(k, k2), "K does not change with forward speed")
    pose_root = pose.copy()
    pose_root[:, :3] = np.random.default_rng(1).normal(size=(t.size, 3))
    check(np.allclose(k, Q.kinematic_summary(pose_root, trans, fps)), "K ignores root orientation")
    p150, t150 = walking_pose(10 * 150, 150)
    k150 = Q.kinematic_summary(p150, np.column_stack([np.zeros(t150.size), np.zeros(t150.size), 1.2 * t150]), 150)
    n150 = dict(zip(Q.KIN_NAMES, k150))
    check(abs(n150["angspeed_knee"] - named["angspeed_knee"]) / named["angspeed_knee"] < 0.05,
          "angular speed agrees across 30 and 150 fps")
    check(abs(n150["stride_freq_hz"] - named["stride_freq_hz"]) <= 0.05, "stride frequency agrees across frame rates")
    r = np.zeros((31, 3))
    r[:, 0] = 0.5 * np.arange(31) / 30
    check(abs(Q._geodesic_rate(r, 1 / 30) - 0.5) < 1e-9, "geodesic rate of a uniform rotation")
    r2 = np.zeros((3, 3))
    r2[:, 0] = [3.1, -3.1, 3.1]
    check(Q._geodesic_rate(r2, 1.0) < 0.1, "geodesic rate does not jump at the axis-angle wrap")
    short_pose, ts = walking_pose(20, 25)
    short = Q.kinematic_summary(short_pose, np.column_stack([ts * 0, ts * 0, ts]), 25)
    check(np.isfinite(short).all(), "K is finite on a 20-frame walk")
    still = Q.kinematic_summary(np.zeros((60, 72)), np.zeros((60, 3)), 30)
    check(np.isfinite(still).all() and still[15] == 0.0, "a still walk gives finite K and zero frequency")


def test_configs():
    check(len(Q.CONFIGS) == 6, "exactly six candidates")
    check([c[1] for c in Q.CONFIGS] == ["LR", "LR", "RF", "RF", "HGB", "HGB"], "three families, two presets each, frozen order")
    check(Q.CONFIGS[0][0] == "LR_C1" and Q.CONFIGS[0][2]["C"] == 1.0 and Q.CONFIGS[0][2]["max_iter"] == 2000, "first candidate is the P0 model")
    rf = Q.make_model(Q.CONFIGS[2])
    check(rf.n_jobs == 1 and rf.random_state == Q.SEED and rf.n_estimators == 300, "RF uses one job, the fixed seed and 300 trees")
    hgb = Q.make_model(Q.CONFIGS[4])
    check(hgb.early_stopping is False and hgb.random_state == Q.SEED, "HGB has early stopping off and the fixed seed")
    check(set(Q.PERSON_SCHEME_ARMS) <= set(Q.ARMS) and set(Q.PARITY_ARMS) <= set(Q.ARMS), "named arm sets exist")
    check(not set(Q.TRANSDUCTIVE_ARMS) & set(Q.INDUCTIVE_ARMS), "transductive arms are separate from inductive arms")


def test_real_configs_fit():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] > 0).astype(int) + (X[:, 0] > 1).astype(int)
    w = Q.train_weights(y, np.arange(300) % 30, "walk")
    for config in Q.CONFIGS:
        a = Q.fit_predict_proba(config, X, y, w, X[:50], [0])
        b = Q.fit_predict_proba(config, X, y, w, X[:50], [0])
        check(a.shape == (50, 3) and np.allclose(a.sum(axis=1), 1) and np.array_equal(a, b), f"{config[0]} fits and is deterministic")
        check((a.argmax(axis=1) == y[:50]).mean() > 0.7, f"{config[0]} learns a simple signal")


def test_weights_and_parity():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 400)
    subject = rng.integers(0, 40, 400)
    check(np.allclose(Q.train_weights(y, subject, "walk"), compute_sample_weight("balanced", y)), "walk weights equal scikit-learn balanced")
    w = Q.train_weights(y, subject, "person")
    check(np.allclose([w[y == k].sum() for k in range(3)], w.sum() / 3), "person weights are class balanced")
    raw = 1 / np.bincount(subject, minlength=40)[subject]
    per_subject = np.array([raw[subject == s].sum() for s in np.unique(subject)])
    check(np.allclose(per_subject, 1), "person weights start from equal subject totals")
    check("X" not in inspect.signature(Q.train_weights).parameters, "weights cannot depend on the input arm")
    y2 = np.where(y == 2, 0, y)
    w2 = Q.train_weights(y2, subject, "walk")
    check(np.allclose(w2, compute_sample_weight("balanced", y2)), "walk weights handle an absent class like scikit-learn")
    X = np.column_stack([y + rng.normal(0, 1, 400), rng.normal(size=400)])
    probe = rng.normal(size=(100, 2))
    a = Q.fit_predict_proba(Q.CONFIGS[0], X, y, Q.train_weights(y, subject, "walk"), probe, [0])
    b, proba_b = P.fit_predict(X, y, probe, [0])
    check(np.array_equal(a.argmax(axis=1), b) and np.allclose(a, proba_b, atol=1e-5), "P1 LR path equals the P0 fit path")


def test_missing_classes():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(120, 3))
    y = np.repeat([0, 2], 60)
    for config in Q.CONFIGS:
        proba = Q.fit_predict_proba(config, X, y, Q.train_weights(y, np.arange(120) % 12, "walk"), X[:10], [0])
        check(proba.shape == (10, 3) and np.allclose(proba[:, 1], 0), f"{config[0]}: a class absent in training gets probability 0 in column 1")
    p = rng.integers(0, 3, 200)
    t = np.where(rng.integers(0, 3, 200) == 1, 0, 2)
    check(abs(Q.walk_macro_f1(t, p) - f1_score(t, p, labels=[0, 1, 2], average="macro", zero_division=0)) < 1e-12, "macro-F1 scores an absent class as 0")
    y3 = rng.integers(0, 3, 500)
    p3 = rng.integers(0, 3, 500)
    cm = np.zeros((3, 3))
    np.add.at(cm, (y3, p3), 1)
    check(abs(Q.balanced_accuracy_conf(cm) - balanced_accuracy_score(y3, p3)) < 1e-12, "balanced accuracy matches scikit-learn")
    check(abs(Q.ordinal_mae_conf(cm) - np.abs(y3 - p3).mean()) < 1e-12, "ordinal MAE matches a direct calculation")


def test_selection_rules():
    sig = inspect.signature(Q.inner_scores).parameters
    check(list(sig) == ["X", "y", "site", "subject", "scheme", "configs", "warned"], "inner selection takes source arrays only")
    check(Q.select_config(np.array([0.5, 0.5, 0.4]), [0, 1, 2]) == 0, "a tie keeps the earlier candidate")
    check(Q.select_config(np.array([0.5, 0.5 + 1e-13, 0.4]), [0, 1, 2]) == 0, "a difference within 1e-12 is a tie")
    check(Q.select_config(np.array([0.5, 0.6, 0.7]), [0, 1]) == 1, "selection respects the allowed set")
    pm = Q.policy_members(Q.CONFIGS)
    check(pm["all"] == [0, 1, 2, 3, 4, 5] and pm["nonlinear"] == [2, 3, 4, 5] and pm["RF"] == [2, 3], "policy members")
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 2))
    y = rng.integers(0, 3, 300)
    site = np.repeat([0, 1, 2], 100)
    s = Q.inner_scores(X, y, site, np.arange(300), "walk", FAST[:2], [0])
    check(s.shape == (2, 3), "inner scores are per candidate and per source site")
    seen = []

    class Spy:
        def __call__(self, config, Xtr, ytr, wtr, Xte, warned, seed=Q.SEED):
            seen.append((Xtr.shape[0], Xte.shape[0]))
            return np.full((Xte.shape[0], 3), 1 / 3)
    real, Q.fit_predict_proba = Q.fit_predict_proba, Spy()
    try:
        Q.inner_scores(X, y, site, np.arange(300), "walk", FAST[:1], [0])
    finally:
        Q.fit_predict_proba = real
    check(seen == [(200, 100)] * 3, "each inner fold trains on two sites and validates on the third")


def test_outer_isolation(t):
    feats = Q.arm_features(t)
    base = Q.outer_run(feats["KS"], t["label"], t["cohort"], t["subject"], "walk", FAST, [0])
    eligible = np.isin(t["label"], (0, 1, 2))
    check(len(base) == 4, "four outer folds")
    for ci, name in enumerate(C.COHORTS):
        te = base[ci]["test_mask"]
        tr = eligible & (t["cohort"] != ci)
        check(set(t["cohort"][te]) == {ci}, f"{name}: test holds only that cohort")
        check(not set(t["subject"][te]) & set(t["subject"][tr]), f"{name}: no subject in both train and test")
        check(not (t["label"][te] == 3).any() and not (t["label"][tr] == 3).any(), f"{name}: class 3 is out of train and test")
    label2 = t["label"].copy()
    m0 = eligible & (t["cohort"] == 0)
    label2[m0] = (label2[m0] + 1) % 3
    changed = Q.outer_run(feats["KS"], label2, t["cohort"], t["subject"], "walk", FAST, [0])
    check(base[0]["selected"] == changed[0]["selected"], "changing held-out labels leaves selection unchanged")
    check(np.array_equal(base[0]["inner"], changed[0]["inner"]), "changing held-out labels leaves inner scores unchanged")
    check(all(np.array_equal(a, b) for a, b in zip(base[0]["proba"], changed[0]["proba"])), "changing held-out labels leaves predictions unchanged")
    check(not np.array_equal(base[1]["inner"], changed[1]["inner"]), "the same label change does reach other folds as training data")
    X2 = feats["KS"].copy()
    cohort0 = t["cohort"] == 0
    X2[cohort0, 0] += 100.0
    moved = Q.outer_run(X2, t["label"], t["cohort"], t["subject"], "walk", FAST, [0])
    check(np.array_equal(base[0]["inner"], moved[0]["inner"]) and base[0]["selected"] == moved[0]["selected"],
          "changing held-out features leaves inductive selection unchanged")
    again = Q.outer_run(feats["KS"], t["label"], t["cohort"], t["subject"], "walk", FAST, [0])
    check(all(np.array_equal(a["proba"][i], b["proba"][i]) for a, b in zip(base, again) for i in range(6)), "the outer run is deterministic")


def test_scaling(t):
    feats = Q.arm_features(t)
    check("label" not in inspect.signature(Q.scale_columns_within_cohort).parameters, "column scaling has no label argument")
    t2 = dict(t, label=np.where(t["label"] == 3, 0, (t["label"] + 1) % 3))
    feats2 = Q.arm_features(t2)
    check(np.array_equal(feats["Sz"], feats2["Sz"]) and np.array_equal(feats["PzSz"], feats2["PzSz"]), "transductive scaling ignores every label")
    for ci in range(4):
        m = t["cohort"] == ci
        check(abs(np.median(feats["Sz"][m])) < 1e-9, f"cohort {ci}: Sz is centred using all its walks, class 3 included")
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    z = Q.scale_columns_within_cohort(X, np.zeros(10, int))
    check(np.allclose(z[:, 0], 0) and np.isfinite(z).all(), "a constant column is centred, not divided by zero")
    other = t["cohort"] != 0
    moved = t["pose"].copy()
    moved[t["cohort"] == 0] += 5.0
    z_a = Q.scale_columns_within_cohort(t["pose"], t["cohort"])
    z_b = Q.scale_columns_within_cohort(moved, t["cohort"])
    check(np.array_equal(z_a[other], z_b[other]), "one cohort's values never affect another cohort's scaling")
    check(feats["PS"].shape[1] == 139 and feats["K"].shape[1] == 21 and feats["KS"].shape[1] == 22 and feats["S"].shape[1] == 1, "arm dimensions")
    check(np.array_equal(feats["PS"][:, 138], t["speed"]) and np.array_equal(feats["S"][:, 0], t["speed"]), "PS and S share the same P0 speed column")
    check(np.array_equal(feats["PS"][:, :138], t["pose"]), "P1 pose columns are the P0 statistics")


def test_partial():
    rng = np.random.default_rng(9)
    speed = rng.normal(size=600)
    y = np.digitize(speed + rng.normal(0, 0.3, 600), [-0.5, 0.5])
    extra = np.digitize(rng.normal(size=600), [-0.5, 0.5])
    d = rng.normal(size=600)
    within = Q._partial_rank_corr(y, speed, (speed, speed, d))
    check(abs(within) < 0.2, f"a score equal to the covariate adds almost nothing ({within:.3f})")
    check(Q._partial_rank_corr(y, np.ones(600), (speed, d, d)) == 0.0, "a score fully explained by the covariates counts as 0")
    check(abs(Q._partial_rank_corr(y, np.tanh(speed), (speed, speed, d))) < 0.1, "a monotone nonlinear speed score adds almost nothing")
    check(Q._partial_rank_corr(y, speed + 2 * y, (speed, speed, d)) > 0.5, "a score holding the label beyond speed scores high")
    check(abs(Q._partial_rank_corr(extra, speed, (d, d, d))) < 0.12, "unrelated label and score stay near 0")
    check(np.isnan(Q._partial_rank_corr(np.ones(600), speed, (d, d, d))), "a constant label is undefined")


def test_null_labels(t):
    rng = np.random.default_rng([1, 2])
    a = Q.subject_block_labels(t["label"], t["cohort"], t["subject"], rng)
    b = Q.subject_block_labels(t["label"], t["cohort"], t["subject"], np.random.default_rng([1, 2]))
    check(np.array_equal(a, b), "null labels are deterministic for a seed")
    check(np.array_equal(a[t["label"] == 3], t["label"][t["label"] == 3]) and not (a[t["label"] != 3] == 3).any(), "class 3 stays put")
    eligible = np.isin(t["label"], (0, 1, 2))
    for ci in range(4):
        m = eligible & (t["cohort"] == ci)
        shares = np.bincount(a[m], minlength=3) / m.sum()
        real = np.bincount(t["label"][m], minlength=3) / m.sum()
        check(np.abs(shares - real).max() < 0.2, f"cohort {ci}: null keeps the class mix approximately")
    s = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    out = Q.subject_block_labels(y, np.zeros(12, int), s, np.random.default_rng(1))
    donors_ok = all(set(out[s == k]) <= {0} or set(out[s == k]) <= {1} or set(out[s == k]) <= {2} or set(out[s == k]) <= {0, 1, 2} for k in range(4))
    pure = sum(len(set(out[s == k])) == 1 for k in range(4))
    check(donors_ok and pure >= 3, "null labels keep each subject tied to one donor subject")


def test_decide():
    args = dict(null_rank_p=0.01, null_done=100, guard_delta=0.2, guard_lo=0.1)
    d = Q.decide(0.1, 0.02, 0.2, [0.1, 0.1, -0.1, -0.1], **args)
    check(d["verdict"] == "P1_GO_BEYOND_SPEED" and d["qualifier"] is None, "go when screen, null and guard pass")
    check(Q.decide(0.05, 0.001, 0.2, [0, 0, -1, -1], **args)["screen_pass"], "margin and zero-valued sites sit inside the screen")
    check(Q.decide(0.049999, 0.01, 0.2, [1, 1, 1, 1], **args)["verdict"] == "P1_NO_GO_NOT_BEYOND_SPEED", "below the margin is no go")
    check(Q.decide(0.1, 0.0, 0.2, [1, 1, 1, 1], **args)["verdict"] == "P1_NO_GO_NOT_BEYOND_SPEED", "a lower bound of exactly 0 is no go")
    check(Q.decide(0.1, 0.01, 0.2, [1, -1, -1, -1], **args)["verdict"] == "P1_NO_GO_NOT_BEYOND_SPEED", "one nonnegative site is no go")
    check(Q.decide(0.1, 0.01, 0.2, [1, 1, 1, 1], 0.2, 100, 0.2, 0.1)["verdict"] == "P1_SCREEN_PASS_NOT_ABOVE_NULL", "a pass inside the null is not a go")
    check(Q.decide(0.1, 0.01, 0.2, [1, 1, 1, 1], 0.01, 38, 0.2, 0.1)["verdict"] == "P1_SCREEN_PASS_NULL_UNAVAILABLE", "too few null draws is not a go")
    check(Q.decide(0.1, 0.01, 0.2, [1, 1, 1, 1], 0.01, 39, 0.2, 0.1)["verdict"] == "P1_GO_BEYOND_SPEED", "39 null draws suffice")
    check(Q.decide(0.1, 0.01, 0.2, [1, 1, 1, 1], 0.01, 100, 0.04, 0.1)["qualifier"] == "GAIN_DEPENDS_ON_WEAKER_CONTROL", "a gain against only a weakened control is qualified")
    check(Q.decide(0.1, 0.01, 0.2, [1, 1, 1, 1], 0.01, 100, 0.2, 0.0)["qualifier"] == "GAIN_DEPENDS_ON_WEAKER_CONTROL", "a guard lower bound of 0 is qualified")
    check(Q.decide(-0.1, -0.2, -0.01, [-1, -1, -1, -1], **args)["qualifier"] == "POSE_ARM_WORSE", "a clearly negative delta is labelled worse")
    check(Q.decide(0.0, -0.1, 0.1, [1, 1, -1, -1], **args)["qualifier"] == "INCONCLUSIVE", "an interval across 0 is inconclusive")
    check(Q.decide(0.02, 0.01, 0.1, [1, 1, 1, 1], **args)["qualifier"] == "SMALL_POSITIVE", "a small clear gain is labelled small")


def test_stops():
    t = Q.build_table(synth_objs("speed", seed=11))
    TOTALS = totals_of(t)
    check(not raises_source(lambda: Q.check_stops(t, TOTALS)), "valid synthetic table passes")
    check(raises_source(lambda: Q.check_stops(t)), "wrong totals stop")
    kin = t["kin"].copy()
    kin[0, 0] = np.nan
    check(raises_source(lambda: Q.check_stops(dict(t, kin=kin), TOTALS)), "non-finite K stops")
    check(raises_source(lambda: Q.check_stops(dict(t, kin=t["kin"][:, :20]), TOTALS)), "wrong K shape stops")
    check(raises_source(lambda: Q.check_stops(dict(t, nonfinite={**t["nonfinite"], "BMCLab": 1}), TOTALS)), "P0 stop checks still apply")
    check(raises_source(lambda: Q.check_versions("1.26.3", "1.16.0", "1.6.1")), "a wrong numpy version stops")
    check(raises_source(lambda: Q.check_versions("1.26.4", "1.16.0", "1.7.0")), "a wrong scikit-learn version stops")
    check(not raises_source(lambda: Q.check_versions("1.26.4", "1.16.0", "1.6.1")), "the recorded versions pass")
    feats = Q.arm_features(t)
    eligible = np.isin(t["label"], (0, 1, 2))
    expected = {}
    for arm, X in ((a, feats[a]) for a in Q.PARITY_ARMS):
        scores = []
        for ci in range(4):
            tr, te = eligible & (t["cohort"] != ci), eligible & (t["cohort"] == ci)
            pred, _ = P.fit_predict(X[tr], t["label"][tr], X[te], [0])
            scores.append(Q.walk_macro_f1(t["label"][te], pred))
        expected[arm] = float(np.mean(scores))
    found = Q.parity_check(t, expected)
    check(all(abs(found[a] - expected[a]) < 1e-9 for a in expected), "P1 parity reproduces the P0 path exactly")
    wrong = dict(expected, S=expected["S"] + 0.001)
    check(raises_source(lambda: Q.parity_check(t, wrong)), "a parity mismatch stops before any new metric")


def test_ledger():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.tsv"
        check(Q.ledger_states(path) == [] and Q.ledger_refusal([]) is None, "no ledger allows a first attempt")
        Q.ledger_append(path, "abc", "START")
        Q.ledger_append(path, "abc", "PRE_SCORE_STOP")
        check(Q.ledger_states(path) == ["START", "PRE_SCORE_STOP"], "ledger records states")
        check(Q.ledger_refusal(Q.ledger_states(path)) is None, "one pre-score stop allows one documented fix")
        Q.ledger_append(path, "abd", "START")
        check(Q.ledger_refusal(Q.ledger_states(path)) == "PRE_SCORE_FIX_ALREADY_USED", "a second pre-score attempt is the last")
    check(Q.ledger_refusal(["START", "SCORED"]) == "SCORES_EXIST", "any scored attempt refuses a re-run")


def run_small(mode, seed, n_null=45):
    t = Q.build_table(synth_objs(mode, seed=seed))
    Q.check_stops(t, totals_of(t))
    res, state = Q.run_analysis(t, configs=FAST, n_boot=60)
    res = Q.finish_with_null(t, res, state, n_null=n_null, null_budget_s=900)
    return t, res


def test_pipeline():
    t, res = run_small("speed", 21)
    check(res["decision"]["verdict"] == "P1_NO_GO_NOT_BEYOND_SPEED", "speed-only signal: pose adds nothing, so no go")
    check(res["arms_inductive_strong"]["S"]["macro_f1"]["M"]["point"] > 0.6, "speed-only signal is recovered by the speed arm")
    check(res["contrasts_exploratory_inductive"]["S_minus_F_U"]["M"]["point"] > 0.2, "speed arm beats the floor")
    check(abs(res["baselines"]["F_U"]["M"]["point"] - 0.323) < 0.001, "F_U on synthetic counts equals the documented value")
    partial = res["speed_conditional"]["partial_rank_correlation"]
    left = partial["S"]["given_speed_duration"]["M"]["point"]
    raw = partial["S"]["unconditional"]["M"]["point"]
    check(0 <= left < raw / 2, f"adjustment removes most of a sharp speed signal from the speed arm ({raw:.3f} to {left:.3f})")
    leak = partial["PS_minus_S"]["given_speed_duration"]["M"]["point"]
    check(abs(leak) < 0.1, f"a sharp speed-only signal leaks little partial gain to the pose arm ({leak:.3f})")
    text = json.dumps(res)
    check(MARK not in text, "no subject or walk key reaches the output")
    check(len(text) < 262144, f"output fits the summary size cap ({len(text)} bytes)")
    conf = res["confusions_by_cohort"]["S_strong"]
    check(all(sum(map(sum, conf[n])) == C.EXPECTED_CLASS_WALKS[n][0] + C.EXPECTED_CLASS_WALKS[n][1] + C.EXPECTED_CLASS_WALKS[n][2] for n in C.COHORTS),
          "confusions count each cohort's eligible walks")
    check(set(res["selected_configs"]["S"]["BMCLab"]) == {"all", "nonlinear", "LR", "RF", "HGB"}, "selection is reported for every policy")
    check(res["null_subject_block"]["n_draws_done"] == 45, "the null runs the requested draws")
    check(res["null_subject_block"]["M_S"]["rank_p_one_sided"] == 1 / 46, "a built-in speed signal beats every null draw")
    check(res["null_subject_block"]["M_S"]["null_mean"] < res["null_subject_block"]["M_S"]["observed"] - 0.1, "null labels remove the signal")
    t2, res2 = run_small("speed", 21)
    check(json.dumps(res, sort_keys=True) == json.dumps(res2, sort_keys=True), "the whole suite is deterministic for its seeds")

    tp, resp = run_small("pose", 22)
    d = resp["decision"]
    check(d["verdict"] == "P1_GO_BEYOND_SPEED", f"pose-only signal: the go verdict fires ({d['verdict']})")
    check(resp["contrasts_exploratory_inductive"]["KS_minus_S"]["M"]["point"] > 0.1, "pose-only signal: KS beats S")
    check(resp["speed_conditional"]["partial_rank_correlation"]["KS"]["given_speed_duration"]["M"]["point"] > 0.3, "pose-only signal survives conditioning on speed")
    check(resp["speed_conditional"]["partial_rank_correlation"]["KS_minus_S"]["given_speed_duration"]["M"]["ci95"][0] > 0, "pose-only signal gives a clear partial gain over the speed arm")

    tn, resn = run_small("null", 23)
    check(resn["decision"]["verdict"] != "P1_GO_BEYOND_SPEED", f"no signal: no go ({resn['decision']['verdict']})")
    check(resn["arms_inductive_strong"]["S"]["macro_f1"]["M"]["point"] < 0.4, "no signal: the speed arm stays near chance")

    shuffled = dict(t, label=t["label"].copy())
    rng = np.random.default_rng(24)
    el = np.isin(t["label"], (0, 1, 2))
    for ci in range(4):
        m = np.flatnonzero(el & (t["cohort"] == ci))
        shuffled["label"][m] = rng.permutation(t["label"][m])
    res_s, state_s = Q.run_analysis(shuffled, configs=FAST, n_boot=40)
    res_s = Q.finish_with_null(shuffled, res_s, state_s, n_null=10, null_budget_s=300)
    check(res_s["decision"]["verdict"] != "P1_GO_BEYOND_SPEED", "shuffled labels cannot yield a go")
    check(res_s["arms_inductive_strong"]["S"]["macro_f1"]["M"]["point"] < 0.4, "shuffled labels remove the speed signal")
    return t


def test_main_entry():
    """The real entry point on fabricated data, redirected to a temporary folder."""
    import run_p1 as G
    objs = synth_objs("speed", seed=31)
    t = Q.build_table(objs)
    saved = (C.load_cohort, Q.run_analysis, Q.finish_with_null, Q.check_stops, Q.parity_check)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        argv = ["--data-dir", str(d), "--out-dir", str(d)]
        RESULT, P0_RESULT, LEDGER = d / G.RESULT_NAME, d / G.P0_RESULT_NAME, d / G.LEDGER_NAME
        check(G.main(argv) == 20 and not LEDGER.exists(), "P1 refuses without a completed P0 result and uses no attempt")
        P0_RESULT.write_text(json.dumps({"status": "P0_COMPLETE", "arms": {a: {"M": {"point": 0.5}} for a in Q.PARITY_ARMS}}), encoding="utf-8")
        C.load_cohort = lambda c, raw_dir: objs[c]
        real_stops, real_run, real_null = Q.check_stops, Q.run_analysis, Q.finish_with_null
        Q.check_stops = lambda table: real_stops(table, totals_of(t))
        try:
            check(G.main(argv) == 20 and json.loads(RESULT.read_text())["status"] == "P1_PRE_SCORE_STOP", "a parity mismatch writes a stop file with no metric")
            check("macro_f1" not in RESULT.read_text(), "the stop file holds no metric")
            check(Q.ledger_states(LEDGER) == ["START", "PRE_SCORE_STOP"], "the ledger records the pre-score stop")
            Q.parity_check = lambda table, expected: dict(expected)
            Q.run_analysis = lambda table: real_run(table, configs=FAST, n_boot=20)
            Q.finish_with_null = lambda table, result, state: real_null(table, result, state, n_null=3, null_budget_s=60)
            check(G.main(argv) == 0, "the real entry point completes")
            res = json.loads(RESULT.read_text())
            check(res["status"] == "P1_COMPLETE" and isinstance(res["decision"], dict) and isinstance(res["null_subject_block"], dict), "the final file holds the decision and the null")
            check(MARK not in RESULT.read_text(), "the final file holds no subject or walk key")
            check(len(RESULT.read_text()) < 262144, "the final file fits the summary size cap")
            check(Q.ledger_states(LEDGER)[-1] == "SCORED", "the ledger records the scored attempt")
            check(G.main(argv) == 10, "a second run is refused while the result exists")
            RESULT.unlink()
            check(G.main(argv) == 10, "a second run is refused by the ledger even without the result")
        finally:
            C.load_cohort, Q.run_analysis, Q.finish_with_null, Q.check_stops, Q.parity_check = saved


def test_scorer():
    rng = np.random.default_rng(8)
    t = {"label": np.tile([0, 1, 2], 40), "cohort": np.repeat([0, 1, 2, 3], 30),
         "subject": np.repeat(np.arange(24), 5)}
    s = Q.Scorer(t, 200)
    preds = [s.y[n] for n in s.names]
    perfect = s.conf(preds)
    noisy = s.conf([np.where(rng.random(30) < 0.5, s.y[n], (s.y[n] + 1) % 3) for n in s.names])
    pt, dr = s.stats(perfect)
    check(np.allclose(pt, 1) and np.allclose(dr, 1), "a perfect arm stays at 1 in every draw")
    same = s.contrast(noisy, noisy)
    check(same["M"]["point"] == 0 and same["M"]["ci95"] == [0.0, 0.0], "identical arms give a zero paired difference in every draw")
    diff = s.contrast(perfect, noisy)
    a, b = s.stats(perfect)[1], s.stats(noisy)[1]
    check(abs(diff["M"]["ci95"][0] - np.percentile((a - b).mean(axis=1), 2.5)) < 1e-12, "the contrast uses paired draws from one set of subject resamples")
    check(np.array_equal(P.draw_counts([6, 6, 6, 6], 200, Q.SEED)[0], s.counts[0]), "bootstrap draws use the P0 generator and seed")
    pw = s.stats(perfect, person=True)[0]
    check(np.allclose(pw, 1), "person-weighted evaluation of a perfect arm is 1")
    m = s.masks["3DGait"]
    check(m.sum() == 30, "scorer holds the held-out cohort's eligible walks")


def main():
    test_speed_alt()
    test_kinematics()
    test_configs()
    test_real_configs_fit()
    test_weights_and_parity()
    test_missing_classes()
    test_selection_rules()
    test_decide()
    test_ledger()
    test_scorer()
    test_partial()
    test_stops()
    t = test_pipeline()
    test_main_entry()
    test_outer_isolation(t)
    test_scaling(t)
    test_null_labels(t)
    if FAILS:
        print(f"P1_SMOKE_FAILED {len(FAILS)}")
        for f in FAILS:
            print("  " + f)
        return 1
    print("P1_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
