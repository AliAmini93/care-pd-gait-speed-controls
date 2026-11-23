"""Fabricated-only tests of the record-level root parity kernels and the root-parity command. No clinical file, no network.

Trajectory: known canonical transforms (rigid re-framing, per-frame orientation-coupled re-framing, time shift, unit
change, frame offset and stride, different content) and HumanML3D inputs built with the official forward rule (one
leg-length scale, heading rotation about the vertical axis, stride-5 `_down` keys, mirrored keys, vertical discarded,
wrong stride, per-walk scale corruption), the five classes, PD-GaM cross-era handling and mechanism attribution.
Labels: equality, changed grade, grade 3 on either side, 0 as a grade, None/NaN/missing-key encodings, one-sided
records, invalid values, medication occasions (no collapsing to participants), identity collisions, reordered keys,
the encoder-probe subset; small-cell disclosure; inputs (download receipt, pins, unsafe bytes) and the command line.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "care_pd"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_core as IC  # noqa: E402
import provenance_fixtures as FX  # noqa: E402
import root_core as G  # noqa: E402
import root_parity as Q  # noqa: E402
import run_provenance as RP  # noqa: E402

PASS = 0
SUBJ0 = 987650


def ok(name, cond):
    global PASS
    if not cond:
        print(f"FAIL {name}")
        sys.exit(1)
    PASS += 1
    print(f"PASS {name}")


class Sentinel:
    def __getattribute__(self, name):
        raise AssertionError("clinical value touched")


class FakeNpz(dict):
    @property
    def files(self):
        return list(self.keys())


# ---- fabricated motion ----------------------------------------------------------------------------------------
def walk_path(rng, frames, fps, bob=0.004):
    t = np.arange(frames - 1)
    step = 1.1 / fps * (1.0 + 0.25 * np.sin(2 * np.pi * t / (fps * 0.55))) + rng.normal(0, 0.002, frames - 1)
    heading = np.deg2rad(rng.uniform(-40, 40)) * t / max(1, frames - 2) + rng.uniform(0, 6.28)
    d = np.column_stack([step * np.sin(heading), rng.normal(0, 0.0015, frames - 1) + bob * np.cos(2 * np.pi * t / (fps * 0.55)),
                         step * np.cos(heading)])
    start = np.array([rng.uniform(-3, 3), 0.9, rng.uniform(-3, 3)])
    return np.concatenate([[start], start + np.cumsum(d, axis=0)])


def make_root(rng, n_subj=10, walks=2, fps=30, frames=(70, 110), bob=0.004, grade=None):
    out, i = {}, 0
    for s in range(n_subj):
        out[SUBJ0 + s] = {}
        for w in range(walks):
            T = int(rng.integers(*frames))
            pose = (rng.normal(0, 0.3, (T, 72))).astype(np.float32)
            g = (i % 4) if grade is None else grade(i)
            out[SUBJ0 + s][f"trial{w}"] = {"pose": pose, "trans": walk_path(rng, T, fps, bob).astype(np.float32),
                                           "beta": np.zeros(10, np.float32), "fps": fps, "UPDRS_GAIT": g,
                                           "medication": "on" if w else "off", "other": None}
            i += 1
    return out


def copy_cohort(c):
    return {s: {w: dict(r) for w, r in wd.items()} for s, wd in c.items()}


G_RIGID = Rotation.from_euler("xyz", [-90, 35, 5], degrees=True).as_matrix()
J0 = np.array([0.003, -0.24, 0.028])


def reframe(rec, mode, rng):
    """Canonical-like record from a root record under a known transform."""
    pose, tr = rec["pose"].astype(float), rec["trans"].astype(float)
    T = len(tr)
    if mode == "per_frame":
        phi = np.cumsum(rng.normal(0, 0.8, T))
        gs = np.stack([G_RIGID @ Rotation.from_euler("y", p, degrees=True).as_matrix() for p in phi])
    else:
        gs = np.repeat(G_RIGID[None], T, 0)
    rr = Rotation.from_rotvec(pose[:, :3]).as_matrix()
    newpose = pose.copy()
    newpose[:, :3] = Rotation.from_matrix(gs @ rr).as_rotvec()
    newtr = np.einsum("tij,tj->ti", gs, tr + J0) - J0
    newtr -= np.array([newtr[0, 0], 0.0, newtr[0, 2]])
    if mode == "shift":
        newtr = newtr[np.minimum(np.arange(T) + 2, T - 1)]
    elif mode == "units":
        newtr = newtr * 1000.0
    elif mode == "content":
        newtr = walk_path(rng, T, 30).astype(float)
    out = dict(rec)
    out["pose"], out["trans"] = newpose.astype(np.float32), newtr.astype(np.float32)
    if mode == "offset":
        out["pose"], out["trans"] = out["pose"][3:], out["trans"][3:]
    if mode == "stride":
        out["pose"], out["trans"] = out["pose"][0::5], out["trans"][0::5]
    return out


def canonical(root, mode, rng, which=None):
    out = copy_cohort(root)
    for s, wd in out.items():
        for w in wd:
            m = mode if which is None else which(s, w)
            wd[w] = reframe(root[s][w], m, rng)
    return out


def rot2(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.column_stack([v[:, 0] * c - v[:, 1] * s, v[:, 1] * c + v[:, 0] * s])


def to_raw(sub, scale, rng, vertical=True):
    """Official forward rule for the root channels (create_redundant_representation.py:140-142,192-195)."""
    p = np.asarray(sub, float) * scale
    T = len(p)
    half = rng.normal(0, 0.01, T - 1)
    raw = np.zeros((T - 1, 263), np.float32)
    raw[:, 0] = half
    raw[:, 1:3] = rot2(np.diff(p[:, [0, 2]], axis=0), -2 * np.cumsum(half))
    raw[:, 3] = p[:-1, 1] if vertical else 0.9
    raw[:, 4:] = rng.normal(0, 0.01, (T - 1, 259))
    return raw


def humanml(root, rng, rate=1, scale=1.03, mode="official", vertical=True, mirror=True, scale_fn=None):
    npz = FakeNpz()
    for s, wd in root.items():
        for w, rec in wd.items():
            tr = rec["trans"].astype(float)
            if mode == "content":
                tr = walk_path(rng, len(tr), 30)
            sc = scale if scale_fn is None else scale_fn()
            for d in range(rate):
                key = f"{s}__{w}" + (f"_down{d}" if rate > 1 else "")
                npz[key] = to_raw(tr[d::rate], sc, rng, vertical)
                if mirror:
                    npz[key + "_M"] = npz[key].copy()
    return npz


def report(name, root, canon, npz):
    return Q.cohort_report(name, root, canon, npz.files, lambda k: npz[k], lambda: {"stub": True})


def main():
    IC.block_network()
    rng = np.random.default_rng(20261113)

    # --- 1. numeric helpers -----------------------------------------------------------------------------------
    aa = rng.normal(0, 1, (50, 3))
    ok("rotmats equals scipy", np.allclose(Q.rotmats(aa), Rotation.from_rotvec(aa).as_matrix(), atol=1e-10))
    p = rng.normal(size=(60, 3))
    q = 2.5 * p @ G_RIGID.T + np.array([1, 2, 3])
    rot, s, res = Q.procrustes(p, q, with_scale=True)
    ok("procrustes recovers rotation and scale", np.allclose(rot, G_RIGID, atol=1e-8) and abs(s - 2.5) < 1e-8 and res < 1e-10)
    ok("procrustes without scale leaves a residual for a scaled copy", Q.procrustes(p, q)[2] > 0.1)
    p2 = rng.normal(size=(40, 2))
    ok("2-D reflection: proper fit fails, reflection-allowed fit succeeds",
       Q.procrustes(p2, p2 * [1, -1])[2] > 0.1 and Q.procrustes(p2, p2 * [1, -1], allow_reflection=True)[2] < 1e-10)
    a = np.abs(rng.normal(1, 0.3, 80))
    ok("best lag finds known shifts of either sign", Q.best_lag(a[2:], a[:-2])[0] == 2 and Q.best_lag(a[:-2], a[2:])[0] == -2 and Q.best_lag(a, a)[0] == 0)
    med, dev = Q.ratio_stats(3 * a, a)
    ok("ratio stats: constant ratio has zero deviation", abs(med - 3) < 1e-12 and dev < 1e-12)
    ok("ratio stats: too few steps is undefined", Q.ratio_stats(a[:5], a[:5]) == (None, None))
    gs = np.stack([Rotation.from_euler("y", v, degrees=True).as_matrix() for v in np.linspace(0, 40, 50)])
    tr = rng.normal(size=(50, 3))
    tc = np.einsum("tij,tj->ti", gs, tr + J0) - J0 + 7.0
    ok("orientation model: exact per-frame coupling about the vertical axis has zero residual and reproduces the path",
       Q.orientation_model(tr, tc, gs)[0] < 1e-9 and Q.orientation_model(tr, tc, gs)[1] < 1e-9
       and np.allclose(Q.orientation_model(tr, tc, gs)[3], tc, atol=1e-9))
    gm = Rotation.from_rotvec(rng.normal(0, 0.4, (50, 3))).as_matrix()
    tm = np.einsum("tij,tj->ti", gm, tr + J0) - J0 + 7.0
    ok("orientation model: with rotations about mixed axes |J| is recovered",
       Q.orientation_model(tr, tm, gm)[0] < 1e-9 and abs(Q.orientation_model(tr, tm, gm)[2] - np.linalg.norm(J0)) < 1e-6)
    ok("orientation model: a translation unrelated to the rotations leaves a residual",
       Q.orientation_model(tr, rng.normal(size=(50, 3)), gs)[0] > 0.1)
    ok("down index parsed after the walk name", Q.down_index("12__walk_a_down3", "walk_a") == 3
       and Q.down_index("12__walk_a", "walk_a") == 0 and Q.down_index("12__walk_a_down0_M", "walk_a") == 0)
    ok("official stride table", Q.OFFICIAL_RATE == {"3DGait": 1, "BMCLab": 5, "T-SDU-PD": 1, "PD-GaM": 1})
    ok("pair verdict thresholds", Q.pair_verdict(20, 19) == "MATCH" and Q.pair_verdict(20, 10) == "NO_MATCH"
       and Q.pair_verdict(20, 15) == "AMBIGUOUS" and Q.pair_verdict(4, 4) == "INSUFFICIENT")
    ok("class table", Q.trajectory_class("MATCH", "NO_MATCH", False) == "ROOT_MATCHES_HUMANML_ONLY"
       and Q.trajectory_class("NO_MATCH", "MATCH", False) == "ROOT_MATCHES_CANONICAL_ONLY"
       and Q.trajectory_class("MATCH", "MATCH", False) == "BOTH_MATCH_WITH_TRANSFORM"
       and Q.trajectory_class("NO_MATCH", "NO_MATCH", False) == "NEITHER_MATCH"
       and Q.trajectory_class("MATCH", "AMBIGUOUS", False) == "UNRESOLVED"
       and Q.trajectory_class("MATCH", "MATCH", True) == "UNRESOLVED")
    ok("mechanism verdict needs 95% agreement", Q.mechanism_verdict(["TIME_SHIFT"] * 19 + ["UNIDENTIFIED"]) == "TIME_SHIFT"
       and Q.mechanism_verdict(["TIME_SHIFT"] * 18 + ["UNIDENTIFIED"] * 2) == "MIXED"
       and Q.mechanism_verdict(["STEP_MATCH"] * 3) == "NONE_ALL_MATCH")

    # --- 2. trajectory classes -------------------------------------------------------------------------------
    root = make_root(rng)
    h_ok, h_bad = humanml(root, rng), humanml(root, rng, mode="content")
    c_rigid, c_content = canonical(root, "rigid", rng), canonical(root, "content", rng)
    r = report("3DGait", root, c_content, h_ok)
    t = r["trajectory"]
    ok("3DGait-like: input matches root, canonical does not -> ROOT_MATCHES_HUMANML_ONLY",
       t["class"] == "ROOT_MATCHES_HUMANML_ONLY" and t["root_vs_humanml"]["verdict"] == "MATCH"
       and t["root_vs_canonical"]["verdict"] == "NO_MATCH")
    ok("... and no derivation step is claimed for unidentified content", t["derivation_step"] == "UNRESOLVED"
       and t["root_vs_canonical"]["mechanism_verdict"] == "UNIDENTIFIED")
    ok("input scale recovered as one constant (1.03) with tiny spread", abs(t["root_vs_humanml"]["scale"]["median"] - 1.03) < 1e-4
       and t["root_vs_humanml"]["scale_rel_iqr_across_walks"] < 1e-4)
    ok("frames: body pose identical, all aligned at equal length",
       t["partitions"]["rc_alignment"]["cells"]["equal_pose_same"] == 20)
    ok("mirrored and stride keys never chosen (all root walks joined once)",
       t["partitions"]["rh_join"]["cells"] == {"joined": 20, "root_walk_without_input_key": 0})
    r = report("3DGait", root, c_rigid, h_bad)
    ok("input differs, rigid canonical matches -> ROOT_MATCHES_CANONICAL_ONLY",
       r["trajectory"]["class"] == "ROOT_MATCHES_CANONICAL_ONLY"
       and r["trajectory"]["root_vs_canonical"]["rigid_resid"]["q75"] < 1e-4
       and r["trajectory"]["partitions"]["rc_scale_unity"]["cells"]["unity"] == 20
       and r["trajectory"]["partitions"]["rc_orientation_constant"]["cells"]["constant"] == 20)
    r = report("3DGait", root, c_rigid, h_ok)
    ok("both match -> BOTH_MATCH_WITH_TRANSFORM; no step to attribute",
       r["trajectory"]["class"] == "BOTH_MATCH_WITH_TRANSFORM" and r["trajectory"]["derivation_step"] == "NOT_REQUIRED")
    ok("both match: same-frame planar input check also matches",
       r["trajectory"]["partitions"]["rh_planar_same_frame"]["cells"]["match"] == 20)
    ok("neither matches -> NEITHER_MATCH", report("3DGait", root, c_content, h_bad)["trajectory"]["class"] == "NEITHER_MATCH")
    half = canonical(root, None, rng, which=lambda s, w: "rigid" if (s - SUBJ0) < 7 else "content")
    r = report("3DGait", root, half, h_ok)
    ok("boundary: exactly half matching is NO_MATCH by the frozen rule", Q.pair_verdict(20, 10) == "NO_MATCH")
    ok("14 of 20 canonical walks match -> AMBIGUOUS -> UNRESOLVED",
       r["trajectory"]["root_vs_canonical"]["verdict"] == "AMBIGUOUS" and r["trajectory"]["class"] == "UNRESOLVED")
    r = report("3DGait", root, canonical(root, "per_frame", rng), h_ok)
    ok("per-frame orientation-coupled re-framing identified as the derivation step",
       r["trajectory"]["class"] == "ROOT_MATCHES_HUMANML_ONLY"
       and r["trajectory"]["derivation_step"] == "ORIENTATION_COUPLED_TRANSLATION"
       and r["trajectory"]["partitions"]["rc_orientation_constant"]["cells"]["varies"] == 20)
    straight = copy_cohort(c_rigid)
    for wd in straight.values():
        for rec in wd.values():
            T_ = len(rec["trans"])
            d_ = np.column_stack([np.zeros(T_), np.zeros(T_), np.cumsum(np.full(T_, 1.1 / 30) * (1 + 0.1 * rng.normal(size=T_)))])
            rec["trans"] = (np.array([0.0, 0.9, 0.0]) + d_).astype(np.float32)
    r = report("3DGait", root, straight, h_ok)
    ok("unrelated straight walks are not attributed to any mechanism (similarity is descriptive only)",
       r["trajectory"]["root_vs_canonical"]["verdict"] == "NO_MATCH" and r["trajectory"]["derivation_step"] == "UNRESOLVED"
       and r["trajectory"]["partitions"]["rc_mechanism"]["cells"]["UNIDENTIFIED"] in (20, "GE5")
       and all(v == 0 for k, v in r["trajectory"]["partitions"]["rc_mechanism"]["cells"].items() if k != "UNIDENTIFIED"))
    r = report("3DGait", root, canonical(root, "shift", rng), h_ok)
    ok("a two-frame shift is identified as TIME_SHIFT", r["trajectory"]["derivation_step"] == "TIME_SHIFT")
    r = report("3DGait", root, canonical(root, "units", rng), h_ok)
    ok("corrupt unit conversion: steps match with scale 1000, flagged as not unity",
       r["trajectory"]["root_vs_canonical"]["verdict"] == "MATCH"
       and abs(r["trajectory"]["root_vs_canonical"]["scale"]["median"] - 1000) < 0.1
       and r["trajectory"]["partitions"]["rc_scale_unity"]["cells"]["not_unity"] == 20)
    r = report("3DGait", root, canonical(root, "offset", rng), h_ok)
    ok("canonical missing three leading frames: aligned by pose at an offset, still matches",
       r["trajectory"]["partitions"]["rc_alignment"]["cells"]["offset_pose_same"] == 20
       and r["trajectory"]["root_vs_canonical"]["verdict"] == "MATCH")
    r = report("3DGait", root, c_rigid, humanml(root, rng, scale_fn=lambda: rng.uniform(0.5, 1.5)))
    ok("per-walk scale corruption: each walk matches but the across-walk scale spread exposes it",
       r["trajectory"]["root_vs_humanml"]["verdict"] == "MATCH" and r["trajectory"]["root_vs_humanml"]["scale_rel_iqr_across_walks"] > 0.1)

    # stride-5 cohort (BMCLab-like) and wrong stride
    root5 = make_root(rng, fps=150, frames=(300, 420))
    c5 = canonical(root5, "rigid", rng)
    r = report("BMCLab", root5, c5, humanml(root5, rng, rate=5))
    ok("stride 5 with _down keys: official frames [0::5] match", r["trajectory"]["root_vs_humanml"]["verdict"] == "MATCH"
       and r["trajectory"]["class"] == "BOTH_MATCH_WITH_TRANSFORM")
    r = report("BMCLab", root5, c5, humanml(root5, rng, rate=1))
    ok("fps/stride error: inputs at the wrong stride are flagged, not scored",
       r["trajectory"]["partitions"]["rh_frames"]["cells"]["unexpected_relation"] == 20
       and r["trajectory"]["root_vs_humanml"]["verdict"] == "INSUFFICIENT")
    r = report("BMCLab", root5, canonical(root5, "stride", rng), humanml(root5, rng, rate=5))
    ok("canonical at stride 5 aligned by pose", r["trajectory"]["partitions"]["rc_alignment"]["cells"]["stride_pose_same"] == 20)
    # vertical discarded (T-SDU-PD) and cross-era (PD-GaM)
    rootv = make_root(rng, bob=0.03)
    hv = humanml(rootv, rng, vertical=False)
    r = report("T-SDU-PD", rootv, canonical(rootv, "rigid", rng), hv)
    ok("vertical discarded: planar comparison matches", r["trajectory"]["root_vs_humanml"]["verdict"] == "MATCH"
       and r["trajectory"]["root_vs_humanml"]["comparison"].startswith("planar"))
    k0 = next(iter(hv.files))
    s0, w0 = k0.split("__")
    ok("... while a 3-D comparison would fail there",
       not Q.root_vs_humanml(hv[k0], rootv[int(s0)][w0]["trans"], 0, 1, True)["match"])
    r = report("PD-GaM", root, c_rigid, h_ok)
    ok("PD-GaM: input side is cross-era, class UNRESOLVED, canonical pair still reported",
       r["trajectory"]["class"] == "UNRESOLVED" and r["trajectory"]["root_vs_humanml"]["verdict"] == "CROSS_ERA_DESCRIPTIVE_ONLY"
       and r["trajectory"]["root_vs_humanml"]["verdict_if_same_era"] == "MATCH"
       and r["trajectory"]["root_vs_canonical"]["verdict"] == "MATCH")

    # trajectory code never reads clinical values
    sroot = copy_cohort(root)
    for wd in sroot.values():
        for rec in wd.values():
            rec["UPDRS_GAIT"], rec["medication"], rec["other"] = Sentinel(), Sentinel(), Sentinel()
    scan = copy_cohort(c_rigid)
    for wd in scan.values():
        for rec in wd.values():
            rec["UPDRS_GAIT"], rec["medication"], rec["other"] = Sentinel(), Sentinel(), Sentinel()
    try:
        tr_raw = Q.trajectory_raw("3DGait", Q.walk_index(sroot)[0], Q.walk_index(scan)[0], h_ok.files, lambda k: h_ok[k])
        Q.trajectory_partitions(tr_raw)
        touched = False
    except AssertionError:
        touched = True
    ok("trajectory functions never touch a clinical value (sentinel)", not touched)

    # --- 3. labels ---------------------------------------------------------------------------------------------
    def labels(root_c, canon_c, npz=h_ok):
        return report("3DGait", root_c, canon_c, npz)["clinical"]

    rootL = make_root(rng)
    canL = canonical(rootL, "rigid", rng)
    hL = humanml(rootL, rng)
    cl = labels(rootL, canL, hL)
    ok("equal labels -> LABEL_PARITY_PASS, exact mode", cl["label_verdict"] == "LABEL_PARITY_PASS" and cl["mode"] == "exact"
       and cl["label_partition"]["cells"]["EQUAL_GRADE"] == 20)
    ok("grade counts shown when labels pass", cl["grade_counts_equal_walks"]["cells"] == {"grade_0": 5, "grade_1": 5, "grade_2": 5, "grade_3": 5})
    ok("c3x subset (canonical grade 0-2, joined, usable) equal", cl["c3x_subset"]["cells"]["EQUAL_GRADE"] == 15
       and cl["c3x_subset_verdict"] == "C3X_SUBSET_LABELS_EQUAL")
    ok("medication and other equal", cl["medication"]["cells"]["EQUAL"] == 20 and cl["other_field"]["cells"]["EQUAL"] == 20)

    def mutate(fn):
        c = copy_cohort(canL)
        fn(c)
        return c

    s1, s2 = SUBJ0 + 1, SUBJ0 + 2
    cl = labels(rootL, mutate(lambda c: c[s1]["trial0"].update(UPDRS_GAIT=(c[s1]["trial0"]["UPDRS_GAIT"] + 1) % 3)), hL)
    ok("one changed grade -> FAIL, coarse disclosure, nothing exact to subtract",
       cl["label_verdict"] == "LABEL_PARITY_FAIL" and cl["mode"] == "coarse"
       and cl["label_partition"]["cells"]["CHANGED_GRADE"] == "LT5" and cl["label_partition"]["cells"]["EQUAL_GRADE"] == "GE5"
       and cl["records"]["cells"]["both"] == "GE5" and cl["grade_counts_equal_walks"] == "WITHHELD_LABELS_NOT_PASS")
    ok("coarse clinical unit forces a coarse trajectory unit", report("3DGait", rootL, mutate(
        lambda c: c[s1]["trial0"].update(UPDRS_GAIT=0 if c[s1]["trial0"]["UPDRS_GAIT"] else 1)), hL)["trajectory"]["mode"] == "coarse")
    g3 = [(s, w) for s, wd in rootL.items() for w, rec in wd.items() if rec["UPDRS_GAIT"] == 3][0]
    cl = labels(rootL, mutate(lambda c: c[g3[0]][g3[1]].update(UPDRS_GAIT=2)), hL)
    ok("root grade 3 vs canonical 2: FAIL, grade-3 sets differ, and the c3x subset sees the change",
       cl["label_verdict"] == "LABEL_PARITY_FAIL" and cl["grade3_sets_same"] is False
       and cl["c3x_subset"]["cells"]["CHANGED_GRADE"] == "LT5" and cl["c3x_subset_verdict"] == "C3X_SUBSET_LABELS_DIFFER")
    g2 = [(s, w) for s, wd in rootL.items() for w, rec in wd.items() if rec["UPDRS_GAIT"] == 2][0]
    cl = labels(rootL, mutate(lambda c: c[g2[0]][g2[1]].update(UPDRS_GAIT=3)), hL)
    ok("canonical grade 3 vs root 2: FAIL overall, but outside the c3x subset (grade 3 excluded there)",
       cl["label_verdict"] == "LABEL_PARITY_FAIL" and cl["c3x_subset_verdict"] == "C3X_SUBSET_LABELS_EQUAL"
       and cl["grade3_sets_same"] is False)
    g0 = [(s, w) for s, wd in rootL.items() for w, rec in wd.items() if rec["UPDRS_GAIT"] == 0][0]
    cl = labels(rootL, mutate(lambda c: c[g0[0]][g0[1]].update(UPDRS_GAIT=None)), hL)
    ok("grade 0 is a grade, not missing: root 0 vs canonical None -> GRADED_ROOT_ONLY -> FAIL",
       cl["label_partition"]["cells"]["GRADED_ROOT_ONLY"] == "LT5" and cl["label_verdict"] == "LABEL_PARITY_FAIL")
    cl = labels(rootL, mutate(lambda c: c[g0[0]][g0[1]].pop("UPDRS_GAIT")), hL)
    ok("missing grade key on the canonical side -> GRADED_ROOT_ONLY",
       cl["label_partition"]["cells"]["GRADED_ROOT_ONLY"] == "LT5" and cl["label_encodings_canonical_joined"]["cells"]["ABSENT_KEY"] == "LT5")
    rootN = make_root(rng, grade=lambda i: None if i % 4 == 0 else i % 3)
    canN = canonical(rootN, "rigid", rng)
    for wd in canN.values():
        for rec in wd.values():
            if rec["UPDRS_GAIT"] is None:
                rec["UPDRS_GAIT"] = float("nan")
    cl = labels(rootN, canN, humanml(rootN, rng))
    ok("None vs NaN: absent on both sides, encoding difference reported, not a label mismatch",
       cl["label_partition"]["cells"]["ABSENT_ENCODING_DIFFERS"] == 5 and cl["label_verdict"] == "LABEL_PARITY_PASS")
    rootX = copy_cohort(rootL)
    rootX[s2]["extra_trial"] = dict(rootL[s2]["trial0"])
    cl = labels(rootX, canL, hL)
    ok("graded root record without canonical counterpart -> PARTIAL", cl["label_verdict"] == "LABEL_PARITY_PARTIAL"
       and cl["label_partition"]["cells"]["ROOT_RECORD_ONLY_GRADED"] == "LT5")
    cl = labels(rootL, mutate(lambda c: c[s1]["trial1"].update(UPDRS_GAIT=7)), hL)
    ok("invalid canonical value (7) -> PARTIAL, INVALID_VALUE", cl["label_verdict"] == "LABEL_PARITY_PARTIAL"
       and cl["label_partition"]["cells"]["INVALID_VALUE"] == "LT5")
    cl = labels(rootL, mutate(lambda c: c[s1]["trial1"].update(UPDRS_GAIT=1.5)), hL)
    ok("non-integral canonical value -> INVALID_VALUE", cl["label_partition"]["cells"]["INVALID_VALUE"] == "LT5")

    def swap_occasions(c):
        a_, b_ = c[s1]["trial0"], c[s1]["trial1"]
        a_["UPDRS_GAIT"], b_["UPDRS_GAIT"] = b_["UPDRS_GAIT"], a_["UPDRS_GAIT"]
        a_["medication"], b_["medication"] = b_["medication"], a_["medication"]
    cl = labels(rootL, mutate(swap_occasions), hL)
    ok("medication occasions swapped within a participant: two walk-level changes, not hidden by collapsing",
       cl["label_verdict"] == "LABEL_PARITY_FAIL" and cl["label_partition"]["cells"]["CHANGED_GRADE"] == "LT5"
       and cl["medication"]["cells"]["DIFFERS"] == "LT5")
    rootC = copy_cohort(rootL)
    rootC[str(s2)] = {"trial9": dict(rootL[s2]["trial0"])}
    cl = labels(rootC, canL, hL)
    ok("int and str subject keys colliding -> set aside, PARTIAL", cl["label_verdict"] == "LABEL_PARITY_PARTIAL"
       and cl["label_partition"]["cells"]["IDENTITY_COLLISION_RECORDS"] == "LT5")
    rev = {s: dict(reversed(list(wd.items()))) for s, wd in reversed(list(canL.items()))}
    a1 = json.dumps(report("3DGait", rootL, canL, hL), sort_keys=True)
    a2 = json.dumps(report("3DGait", rootL, rev, hL), sort_keys=True)
    ok("reordered canonical keys give an identical report", a1 == a2)

    # --- 4. disclosure and aggregate -----------------------------------------------------------------------------
    big = report("3DGait", rootL, canL, hL)
    small = report("3DGait", rootL, mutate(lambda c: c[s1]["trial0"].update(UPDRS_GAIT=(c[s1]["trial0"]["UPDRS_GAIT"] + 1) % 3)), hL)
    ok("disclosure rules hold in exact and coarse reports", G.disclosure_problems(big) == [] and G.disclosure_problems(small) == [])
    ok("coarse report holds no integer count but zero", all(v in (0, "LT5", "GE5") for part in
       (small["clinical"]["label_partition"], small["clinical"]["records"]) for v in part["cells"].values()))
    ok("coarse trajectory shares are bands", isinstance(small["trajectory"]["root_vs_canonical"]["share_match"], str))
    ok("exact render refuses a small cell", refused_render())
    out = {"card": "root-parity-first", "card_sha256": "0" * 64, "run_commit": "a" * 40, "seed": 1,
           "protocol_class": "METADATA", "status": "G_ROOT_PARITY_COMPLETE", "labels_read": True,
           "cohorts": {"3DGait": big, "BMCLab": small}, "compute": IC.Receipt().block()}
    text = IC.render(out)
    ok("aggregate passes the renderer", IC.problems_of(json.loads(text)) == [])
    ok("no subject key, walk key or label value in the aggregate", all(x not in text for x in (str(SUBJ0), "trial0", "trial1", '"on"', '"off"')))

    # --- 5. inputs ---------------------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        saved = (G.ROOT_DIR, dict(G.ROOT_PINS))
        G.ROOT_DIR = Path(td) / "root"
        out_dir = Path(td) / "results"
        out_dir.mkdir()
        try:
            blob = pickle.dumps(rootL, protocol=4)
            G.ROOT_PINS["3DGait"] = ("3DGait.pkl", 926879, len(blob), hashlib.md5(blob).hexdigest(), "v")
            G.root_path("3DGait").parent.mkdir(parents=True)
            G.root_path("3DGait").write_bytes(blob)
            good_row = {"cohort": "3DGait", "publisher_datafile": 926879, "md5_pinned": hashlib.md5(blob).hexdigest(),
                        "sha256": hashlib.sha256(blob).hexdigest(), "integrity_ok": True}
            ok("load_root: verified file decodes", G.trees_equal(Q.load_root("3DGait", {"3DGait": good_row}), rootL, ()))
            ok("load_root: SHA-256 not equal to the download receipt stops",
               exits(lambda: Q.load_root("3DGait", {"3DGait": dict(good_row, sha256="0" * 64)})))
            ok("load_root: no receipt row stops", exits(lambda: Q.load_root("3DGait", {})))
            evil = pickle.dumps(Reduce(os.system, ("true",)), protocol=2)
            G.ROOT_PINS["3DGait"] = ("3DGait.pkl", 926879, len(evil), hashlib.md5(evil).hexdigest(), "v")
            G.root_path("3DGait").write_bytes(evil)
            ok("load_root: injected unsafe pickle with matching pins and receipt stops", exits(lambda: Q.load_root(
                "3DGait", {"3DGait": dict(good_row, md5_pinned=hashlib.md5(evil).hexdigest(), sha256=hashlib.sha256(evil).hexdigest())})))
            receipt_file = out_dir / RP.result_name("fetch-originals", "first")
            ok("download receipt missing stops", exits(lambda: RP.download_receipt(out_dir, "first")))
            receipt_file.write_text(json.dumps({"status": "ROOT_SAFE_LOADER_BLOCKED"}))
            ok("download receipt not passed stops", exits(lambda: RP.download_receipt(out_dir, "first")))
            receipt_file.write_text(json.dumps({"status": Q.DOWNLOAD_PASS, "files": [good_row, {"cohort": "x", "stop": "y"}]}))
            ok("download receipt passed: rows with verified integrity only", list(RP.download_receipt(out_dir, "first")) == ["3DGait"])
        finally:
            G.ROOT_DIR = saved[0]
            G.ROOT_PINS.clear()
            G.ROOT_PINS.update(saved[1])
    ok("portable module keeps no private route or archived comparison",
       not any(hasattr(Q, k) for k in ("main", "download_receipt", "e2_parity", "e2_archived", "E2_RESULT")))
    test_cli(rng)
    print(f"ROOT_PARITY_TESTS_PASS n={PASS}")


def npz_arrays(npz):
    return {k: np.asarray(v) for k, v in npz.items()}


def test_cli(rng):
    """root-parity --first through the command line on fabricated, pinned files in temporary folders."""
    saved = FX.snapshot()
    try:
        with tempfile.TemporaryDirectory() as d:
            data, out = Path(d) / "data", Path(d) / "out"
            argv = ["root-parity", "--first", "--data-dir", str(data), "--out-dir", str(out)]
            code, text = FX.quiet_main(RP, argv)
            ok("root-parity stops without the download aggregate and writes nothing",
               code == 20 and not (out / RP.result_name("root-parity", "first")).exists())
            FX.layout(data)
            root = make_root(rng)
            canon = canonical(root, "rigid", rng)
            FX.pin_canonical(data, {"3DGait": canon})
            shas = FX.pin_roots({"3DGait": root})
            FX.pin_npz("3DGait", npz_arrays(humanml(root, rng)))
            FX.receipt(out, RP.result_name("fetch-originals", "first"), ("3DGait",), shas)
            code, text = FX.quiet_main(RP, argv)
            res = out / RP.result_name("root-parity", "first")
            agg = json.loads(res.read_text()) if res.is_file() else {}
            t = agg.get("cohorts", {}).get("3DGait", {}).get("trajectory", {})
            ok("root-parity --first: fabricated rigid re-framing gives BOTH_MATCH and LABEL_PARITY_PASS",
               code == 0 and agg.get("status") == "G_ROOT_PARITY_COMPLETE" and t.get("class") == "BOTH_MATCH_WITH_TRANSFORM"
               and agg["summary"]["3DGait"]["label_verdict"] == "LABEL_PARITY_PASS" and agg.get("card") == "root-parity-first")
            ok("root-parity: the archived same-definition comparison is reported as unavailable, not computed",
               t.get("humanml_vs_canonical_e2_parity") == RP.E2_UNAVAILABLE)
            text = res.read_text() if res.is_file() else ""
            ok("root-parity: aggregate holds no identifier, label value or local path",
               all(x not in text for x in (str(SUBJ0), "trial0", '"on"', '"off"', str(d))) and not G.disclosure_problems(agg))
            ok("root-parity: the root file is verified against the user's own receipt",
               agg["cohorts"]["3DGait"]["root_file"]["sha256"] == shas["3DGait"]
               and agg["cohorts"]["3DGait"]["root_file"]["verified_before_use"] is True)
            code, text = FX.quiet_main(RP, argv)
            ok("root-parity: an existing result is never overwritten", code == 10)
            res.unlink()
            FX.pin_npz("3DGait", {"987650__trial0": np.zeros((5, 263), np.float32)})
            path = Path(F_dv_path("3DGait"))
            path.write_bytes(path.read_bytes() + b"x")
            code, text = FX.quiet_main(RP, argv)
            ok("root-parity: an input that no longer matches its pin stops before any comparison",
               code == 20 and not res.exists())
    finally:
        FX.restore(saved)


def F_dv_path(cohort):
    import encoder_inputs as F
    return F.dv_path(cohort)


class Reduce:
    def __init__(self, fn, args):
        self.r = (fn, args)

    def __reduce__(self):
        return self.r


def exits(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def refused_render():
    try:
        G.render_partition({"a": 3, "b": 10}, False)
    except G.SourceError:
        return True
    return False


if __name__ == "__main__":
    main()
