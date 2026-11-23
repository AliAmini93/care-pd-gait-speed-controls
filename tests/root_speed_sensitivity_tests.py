"""Fabricated-only tests of the original-trajectory speed-sensitivity kernels and the root-speed command.

No clinical file is opened (an audit-hook tripwire refuses every path under a sentinel data folder, where the root
loader is pointed) and every network event is refused.

Covered: row order equal to p0_core.build_table; identity join independent of dict order and of int/str subject keys;
shuffled, missing, extra, duplicated (str-colliding) and moved walks; label, grade-3, absent-label, medication, other,
fps, frame-count, schema, non-finite, body-pose and whole-record-swap corruptions; units (x1000, x0.001), z-up axis
convention, inverted speed (rank gate), legitimate mild per-walk scale and global-orientation differences; the smoothing
replica; substitution (A/C equal the P1 arm builder, D differs only in the speed column); fold membership; held-out
label leakage into selection; locked versus re-selected policies; shared paired bootstrap draws; the frozen decision
rule; baseline parity against a P1 aggregate produced by the frozen p1_core.run_analysis (and its failure on perturbed
values, selections, confusions, totals, configs and missing sections); cross-process determinism; the portable
loaders (user P1 pinned by SHA-256, user download receipts, canonical folder); blocked statuses with no root arm
fitted; the aggregate renderer (no identifier string, no row-level list, small cells coarse, injected row-level keys
refused); and the root-speed command end to end on fabricated pinned files with reduced learners.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SCRIPTS = REPO / "scripts" / "care_pd"
for _p in (str(SCRIPTS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import aggregate_core as IC  # noqa: E402
import root_core as G  # noqa: E402
import common as C  # noqa: E402
import p0_core as P  # noqa: E402
import p1_core as Q  # noqa: E402
import root_speed_sensitivity as H  # noqa: E402

# Sentinel data folder: never created; any open below it is refused. The root loader points here by default.
DATA = os.path.realpath(os.path.join(tempfile.gettempdir(), "care_pd_tripwire_sentinel_data"))
G.ROOT_DIR = Path(DATA) / "original"


def _tripwire(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        real = os.path.realpath(os.fsdecode(args[0]))
        if real == DATA or real.startswith(DATA + os.sep):
            raise PermissionError("H1_TEST_TRIPWIRE_DATA_PATH")


sys.addaudithook(_tripwire)
IC.block_network()

PASS = 0
NB = 60
FPS = {"3DGait": 30, "BMCLab": 60, "PD-GaM": 25, "T-SDU-PD": 30}
SECRET_SUBJ = "QQSECRETSUBJ"
SECRET_WALK = "ZZSECRETWALK"
FAST = (Q.CONFIGS[0], Q.CONFIGS[1],
        ("RF_small", "RF", dict(Q.RF_BASE, n_estimators=15, min_samples_leaf=5)),
        ("HGB_small", "HGB", dict(Q.HGB_BASE, max_leaf_nodes=7, max_iter=15, min_samples_leaf=10, l2_regularization=1.0)))
META = {"card": "root-speed", "run_commit": "0" * 40, "card_sha256": "f" * 64, "seed": Q.SEED, "protocol_class": "SCORED"}


def ok(name, cond):
    global PASS
    if not cond:
        print(f"FAIL {name}")
        sys.exit(1)
    PASS += 1
    print(f"PASS {name}")


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


# ---- fabricated cohorts ---------------------------------------------------------------------------------------------
def make_canonical(seed=7, n_subj=10, walks=4):
    """Four cohorts with the CARE-PD record layout. Speed falls with grade; pose spread rises with grade; two grade-3
    walks; 3DGait subject keys are integers, the others strings; every key carries a recognisable secret marker."""
    rng = np.random.default_rng(seed)
    objs = {}
    for ci, name in enumerate(C.COHORTS):
        fps, cohort = FPS[name], {}
        for s in range(n_subj):
            key = 7700 + s if name == "3DGait" else f"{SECRET_SUBJ}{ci}{s:02d}"
            base = s % 3
            cohort[key] = {}
            for w in range(walks):
                grade = int(np.clip(base + (int(rng.integers(-1, 2)) if w % 2 else 0), 0, 2))
                if name in ("3DGait", "PD-GaM") and s == 1 and w == 3:
                    grade = 3
                T = int(rng.integers(70, 110))
                t = np.arange(T) / fps
                v = 1.35 - 0.3 * min(grade, 2) + rng.normal(0, 0.07) + 0.04 * ci
                head = rng.uniform(-0.3, 0.3)
                fwd = v * t + 0.02 * np.sin(2 * np.pi * 1.8 * t) + np.cumsum(rng.normal(0, 0.002, T))
                trans = np.column_stack([fwd * np.sin(head) + rng.normal(0, 0.003, T),
                                         0.9 + 0.02 * np.sin(2 * np.pi * 1.8 * t),
                                         fwd * np.cos(head) + rng.normal(0, 0.003, T)])
                pose = rng.normal(0, 0.25, (T, 72))
                pose[:, 3:30] *= 1 + 0.3 * min(grade, 2)
                cohort[key][f"{SECRET_WALK}_{s}_{w}"] = {
                    "pose": pose.astype(np.float32), "trans": trans.astype(np.float32),
                    "beta": np.zeros((1, 10), np.float32), "fps": fps, "UPDRS_GAIT": grade,
                    "medication": "on" if w % 2 else "off", "other": None}
        objs[name] = cohort
    return objs


def make_root(canon, seed=11, scale=1.01, noise=0.002, shuffle=True, str_3dgait=True, rotate=True, per_walk=None):
    """Original-root stand-in: same records and body pose; trans rotated about the vertical axis, re-originated,
    scaled and jittered per frame (non-rigid), optionally a per-walk scale; dict order shuffled."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, cohort in canon.items():
        subjects = list(cohort)
        if shuffle:
            subjects = [subjects[i] for i in rng.permutation(len(subjects))]
        oc = {}
        for s in subjects:
            walks = list(cohort[s])
            if shuffle:
                walks = [walks[i] for i in rng.permutation(len(walks))]
            key = str(s) if (str_3dgait and name == "3DGait") else s
            oc[key] = {}
            for w in walks:
                rec = cohort[s][w]
                tr = rec["trans"].astype(np.float64)
                a = rng.uniform(0, 2 * np.pi) if rotate else 0.0
                rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
                k = scale * (per_walk(rng) if per_walk else 1.0)
                new = ((tr - tr[0]) @ rot.T) * k + np.array([rng.uniform(-2, 2), 0.1, rng.uniform(-2, 2)])
                if noise:
                    new = new + rng.normal(0, noise, new.shape)
                oc[key][w] = dict(rec, pose=rec["pose"].copy(), trans=new.astype(np.float32))
        out[name] = oc
    return out


def fab_check_stops(t):
    if not (np.isfinite(t["speed"]).all() and np.isfinite(t["pose"]).all()):
        raise C.SourceError("FAB_NON_FINITE")


def walks_with(objs, name, pred):
    return [(s, w) for s, ws in objs[name].items() for w, r in ws.items() if pred(r)]


def first_key(objs, name, pred=lambda r: True):
    return walks_with(objs, name, pred)[0]


def join(root, canon, table):
    return H.join_root(table, canon, root)


def fold_digest(folds):
    h = hashlib.sha256()
    for f in folds:
        h.update(np.ascontiguousarray(f["inner"]).tobytes())
        h.update(json.dumps(f["selected"], sort_keys=True).encode())
        for p in f["proba"]:
            h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()


def max_list_len(obj):
    if isinstance(obj, dict):
        return max([0] + [max_list_len(v) for v in obj.values()])
    if isinstance(obj, (list, tuple)):
        return max([len(obj)] + [max_list_len(v) for v in obj])
    return 0


def jsonable(obj):
    return json.loads(json.dumps(H.rounded(H._plain(obj))))


# ---- tests -------------------------------------------------------------------------------------------------------------
def main():
    # -- environment, tripwire, constants, pins ------------------------------------------------------------------------
    ok("tripwire_refuses_data_paths", raises(lambda: open(Path(DATA) / "care_pd" / "3DGait_canonical.pkl", "rb"),
                                             PermissionError))
    ok("tripwire_refuses_root_dir", raises(lambda: open(G.root_path("3DGait"), "rb"), PermissionError))
    import socket
    ok("network_blocked", raises(lambda: socket.getaddrinfo("example.org", 443), PermissionError))
    ok("frozen_constants", H.MARGIN == 0.05 == Q.SCREEN_MIN and H.PARITY_TOL == 1e-5 and H.POSE_TOL == 1e-4
       and H.UNIT_BAND == (0.8, 1.25) and H.H_SHARE_MIN == 0.90 and H.H_SHARE_MAX_BELOW == 0.05 and H.RANK_MIN == 0.5
       and Q.SEED == 20261030 and Q.N_BOOT == 2000 and Q.REFIT_SEEDS == (1, 2, 3, 4, 5))
    sig = inspect.signature(H.run).parameters
    ok("run_defaults_are_frozen_engine",
       sig["configs"].default is Q.CONFIGS and sig["n_boot"].default == 2000 and sig["check_stops"].default is Q.check_stops
       and sig["fit"].default is Q.outer_run and sig["refit_seeds"].default == Q.REFIT_SEEDS
       and sig["canonical_loader"].default is H.load_canonical and sig["root_loader"].default is H.load_roots
       and sig["frozen_loader"].default is H.frozen_p1 and sig["receipts"].default is H.receipt_rows)
    ok("canonical_pins_unchanged", C.HF_COMMIT == "d53fe929aaacbd6822bcad5ef2b1d215401800b7"
       and C.FILES["3DGait"][2] == "c765e5612b5b1993ef7cd71d0e93211c7a416c7557faef5de0c837dfd511acfc"
       and C.COHORTS == ("3DGait", "BMCLab", "PD-GaM", "T-SDU-PD"))
    ok("root_md5_pins_match_publisher", {c: G.ROOT_PINS[c][3] for c in G.COHORTS} == {
        "3DGait": "a4fb5b909b4a7d913530a8ac82ce9c99", "BMCLab": "8744c40c868de8f32138673cc0b05485",
        "T-SDU-PD": "a1b978b7bd5e1d4edbb1306218c22d26", "PD-GaM": "92884135f342f715ab4f5efd21b8ad7b"})
    # -- smoothing replica, coordinate gate --------------------------------------------------------------------------
    rng = np.random.default_rng(3)
    same = True
    for fps in (25, 30, 60, 150):
        tr = np.cumsum(rng.normal(0, 0.01, (200, 3)), axis=0) + np.linspace(0, 3, 200)[:, None] * [0.2, 0.0, 1.0]
        s0, d0 = P.speed_and_duration(tr, fps)
        xz, xyz = H.smoothed_lengths(tr, fps)
        same &= abs(xz / d0 - s0) <= 1e-12 * max(1.0, s0) and xyz >= xz
    ok("smoothing_replica_equals_p0_speed", same)
    ok("coordinate_verdicts", H.coordinate_verdict([0.99] * 40) == "PASS"
       and H.coordinate_verdict([0.8] * 40) == "FAIL_MEDIAN_BELOW_MIN"
       and H.coordinate_verdict([0.99] * 36 + [0.5] * 4) == "FAIL_TOO_MANY_WALKS_BELOW_MIN"
       and H.coordinate_verdict([0.99] * 39 + [0.5]) == "PASS"
       and H.coordinate_verdict([0.99] * 39 + [None]) == "FAIL_UNDEFINED" and H.coordinate_verdict([0.99] * 4) == "FAIL_UNDEFINED")
    ok("root_speed_stop_rules", H.root_speed_stops(np.array([1.0, 1.1, 0.9, 1.2, 0.8])) == "PASS"
       and H.root_speed_stops(np.array([1.0, np.nan, 1.0, 1.0, 1.0])) != "PASS"
       and H.root_speed_stops(np.array([0.001] * 4 + [1.0, 1.2])) == "FAIL_LOW_SPEED_SHARE"
       and H.root_speed_stops(np.ones(10)) == "FAIL_ZERO_IQR")

    # -- fabricated fixtures -----------------------------------------------------------------------------------------
    canon = make_canonical()
    table = Q.build_table(canon)
    root = make_root(canon)
    n = table["label"].size
    rows_ = list(H.iter_rows(canon))
    ok("iter_rows_matches_build_table_order", len(rows_) == n and all(
        P.speed_and_duration(C.read_arrays(r)[1], C.read_fps(r["fps"]))[0] == table["speed"][i]
        and table["cohort"][i] == ci for i, (ci, _s, _w, r) in enumerate(rows_)))
    ok("fixture_has_grade3_and_all_classes", int((table["label"] == 3).sum()) == 2 and all(
        all(((table["label"] == k) & (table["cohort"] == ci)).any() for k in (0, 1, 2)) for ci in range(4)))

    speed_root, raw, failure = join(root, canon, table)
    ok("valid_join_passes_despite_shuffle_and_str_keys", failure is None and np.isfinite(speed_root).all()
       and type(next(iter(root["3DGait"]))) is str and type(next(iter(canon["3DGait"]))) is int)
    ok("valid_join_checks_all_equal", all(raw[c]["checks"][k]["differs"] == 0 for c in C.COHORTS for k in H.CHECKS)
       and sum(raw[c]["records"]["joined"] for c in C.COHORTS) == n)
    by_key = {}
    for name in C.COHORTS:
        for s, ws in root[name].items():
            for w, r in ws.items():
                by_key[(name, str(s), w)] = P.speed_and_duration(r["trans"].astype(np.float64), r["fps"])[0]
    ok("root_speed_joined_by_identity_not_position",
       all(speed_root[i] == by_key[(C.COHORTS[ci], str(s), w)] for i, (ci, s, w, _r) in enumerate(rows_)))
    ok("root_speed_differs_from_canonical", not np.array_equal(speed_root, table["speed"]))

    exact = make_root(canon, scale=1.1, noise=0.0, rotate=False)
    sr, _, f = join(exact, canon, table)
    ok("root_speed_proportional_under_pure_scale", f is None and np.allclose(sr / table["speed"], 1.1, rtol=0, atol=1e-4))
    rot_only = make_root(canon, scale=1.0, noise=0.0, rotate=True)
    sr, _, f = join(rot_only, canon, table)
    ok("root_speed_invariant_to_vertical_axis_rotation", f is None and np.allclose(sr, table["speed"], rtol=1e-4, atol=0))
    mild = make_root(canon, per_walk=lambda r: r.uniform(0.95, 1.05))
    ok("mild_per_walk_scale_differences_pass", join(mild, canon, table)[2] is None)

    def mutated(fn):
        r = copy.deepcopy(root)
        fn(r)
        return join(r, canon, table)[2] or "NONE"

    def rec_of(r, name, pred=lambda x: True):
        s, w = first_key(r, name, pred)
        return r[name][s][w]

    def drop(r):
        s, w = first_key(r, "BMCLab")
        del r["BMCLab"][s][w]

    def extra(r):
        s, w = first_key(r, "PD-GaM")
        r["PD-GaM"][s]["EXTRA_WALK"] = copy.deepcopy(r["PD-GaM"][s][w])

    def collide(r):
        s = next(iter(r["3DGait"]))
        r["3DGait"][int(s)] = copy.deepcopy(r["3DGait"][s])

    def move(r):
        subs = list(r["T-SDU-PD"])
        w = next(iter(r["T-SDU-PD"][subs[0]]))
        r["T-SDU-PD"][subs[1]][w] = r["T-SDU-PD"][subs[0]].pop(w)

    def set_field(name, field, value, pred=lambda x: True):
        return lambda r: rec_of(r, name, pred).__setitem__(field, value)

    def truncate(r):
        x = rec_of(r, "PD-GaM")
        x["pose"], x["trans"] = x["pose"][:-1], x["trans"][:-1]

    def nan_trans(r):
        rec_of(r, "3DGait")["trans"][5, 0] = np.nan

    def body(r):
        rec_of(r, "T-SDU-PD")["pose"][:, 10] += 0.01

    def global_orient(r):
        for name in C.COHORTS:
            for ws in r[name].values():
                for x in ws.values():
                    x["pose"][:, 0:3] += 0.3

    def swap_records(r):
        (s1, w1), (s2, w2) = walks_with(r, "BMCLab", lambda x: True)[:2]
        r["BMCLab"][s1][w1], r["BMCLab"][s2][w2] = r["BMCLab"][s2][w2], r["BMCLab"][s1][w1]

    def no_beta(r):
        del rec_of(r, "BMCLab")["beta"]

    def flat_trans(r):
        x = rec_of(r, "BMCLab")
        x["trans"] = x["trans"][:, :2]

    def scale_all(k):
        def f(r):
            for name in C.COHORTS:
                for ws in r[name].values():
                    for x in ws.values():
                        x["trans"] = (x["trans"].astype(np.float64) * k).astype(np.float32)
        return f

    def z_up(r):
        for ws in r["3DGait"].values():
            for x in ws.values():
                x["trans"] = x["trans"][:, [0, 2, 1]].copy()

    def invert(r):
        med = np.median(table["speed"][table["cohort"] == 2])
        for ws in r["PD-GaM"].values():
            for x in ws.values():
                s0 = P.speed_and_duration(x["trans"].astype(np.float64), x["fps"])[0]
                x["trans"] = (x["trans"].astype(np.float64) * (med / s0) ** 2).astype(np.float32)

    grade2 = lambda x: x["UPDRS_GAIT"] == 2   # noqa: E731
    grade0 = lambda x: x["UPDRS_GAIT"] == 0   # noqa: E731
    grade3 = lambda x: x["UPDRS_GAIT"] == 3   # noqa: E731
    cases = [("missing_root_walk", drop, "RECORD_SETS:BMCLab"), ("extra_root_walk", extra, "RECORD_SETS:PD-GaM"),
             ("str_collision_duplicate", collide, "RECORD_SETS:3DGait"), ("walk_moved_to_other_participant", move, "RECORD_SETS:T-SDU-PD"),
             ("label_changed", set_field("BMCLab", "UPDRS_GAIT", 1, grade2), "LABEL:BMCLab"),
             ("grade3_relabelled", set_field("3DGait", "UPDRS_GAIT", 2, grade3), "LABEL:3DGait"),
             ("label_absent_vs_zero", set_field("T-SDU-PD", "UPDRS_GAIT", None, grade0), "LABEL:T-SDU-PD"),
             ("medication_mismatch", set_field("PD-GaM", "medication", "unknown"), "MEDICATION:PD-GaM"),
             ("other_mismatch", set_field("BMCLab", "other", "x"), "OTHER:BMCLab"),
             ("wrong_fps", set_field("T-SDU-PD", "fps", 31), "FPS:T-SDU-PD"), ("frame_count", truncate, "FRAMES:PD-GaM"),
             ("non_finite_trans", nan_trans, "FINITE:3DGait"), ("body_pose_changed", body, "BODY_POSE:T-SDU-PD"),
             ("whole_record_swap", swap_records, None), ("missing_field", no_beta, "SCHEMA:BMCLab"),
             ("trans_two_columns", flat_trans, "SCHEMA:BMCLab"), ("unit_millimetres", scale_all(1000.0), "UNIT_BAND:3DGait"),
             ("z_up_axis_convention", z_up, "COORDINATE_CONVENTION:3DGait"), ("inverted_speed", invert, "RANK_AGREEMENT:PD-GaM")]
    for label, fn, want in cases:
        got = mutated(fn)
        ok(f"corruption_{label}_fails_closed", got != "NONE" and (want is None or got == want))
    got = mutated(scale_all(0.001))
    ok("corruption_unit_kilometres_fails_closed", got.startswith(("ROOT_SPEED_STOPS", "UNIT_BAND")))
    ok("root_global_orientation_difference_passes", mutated(global_orient) == "NONE")
    reordered = {name: dict(reversed(list(canon[name].items()))) for name in C.COHORTS}
    ok("canonical_row_reordering_fails_row_order", join(root, reordered, table)[2] == "ROW_ORDER:3DGait")
    ok("missing_root_cohort_fails", join({k: v for k, v in root.items() if k != "BMCLab"}, canon, table)[2] == "RECORD_SETS:BMCLab")

    # -- disclosure of the join report --------------------------------------------------------------------------------------
    r2 = copy.deepcopy(root)
    for s, w in walks_with(r2, "BMCLab", grade2)[:2]:
        r2["BMCLab"][s][w]["UPDRS_GAIT"] = 1
    _, raw2, f2 = join(r2, canon, table)
    rj = H.render_join(raw2)
    ok("small_mismatch_makes_cohort_coarse", f2 == "LABEL:BMCLab" and rj["BMCLab"]["mode"] == "coarse"
       and rj["BMCLab"]["checks"]["label"]["cells"]["differs"] == "LT5" and rj["3DGait"]["mode"] == "exact"
       and not G.disclosure_problems(rj))

    # -- substitution gate ------------------------------------------------------------------------------------------------
    X = H.arm_inputs(table, speed_root)
    ok("substitution_valid", H.substitution_problems(X, table, speed_root) == [] and X["D"].shape[1] == X["C"].shape[1] == 139)

    def sub_after(fn):
        Y = {k: v.copy() for k, v in X.items()}
        fn(Y)
        return H.substitution_problems(Y, table, speed_root)

    ok("substitution_changed_pose_column", "D_POSE_COLUMNS_DIFFER_FROM_C" in sub_after(lambda Y: Y["D"].__setitem__((slice(None), 3), 0.0)))
    ok("substitution_C_not_p1", "C_NOT_P1_PS" in sub_after(lambda Y: Y["C"].__setitem__((0, 0), 9.0)))
    ok("substitution_A_not_p1", "A_NOT_P1_S" in sub_after(lambda Y: Y["A"].__setitem__((0, 0), 9.0)))
    ok("substitution_B_not_root", "ROOT_COLUMN_NOT_S_ROOT" in sub_after(lambda Y: Y["B"].__setitem__((0, 0), 9.0)))
    ok("substitution_D_speed_is_canonical", "ROOT_COLUMN_NOT_S_ROOT" in sub_after(lambda Y: Y["D"].__setitem__((slice(None), -1), table["speed"])))
    ok("substitution_non_finite", "NON_FINITE" in sub_after(lambda Y: Y["D"].__setitem__((1, -1), np.nan)))

    # -- folds, leakage, policies ----------------------------------------------------------------------------------------
    warned = [0]
    folds = {a: Q.outer_run(X[a], table["label"], table["cohort"], table["subject"], "walk", FAST, warned) for a in H.ARMS}
    ok("fold_membership_valid", H.fold_problems(folds, table) == [])
    tampered = {a: list(v) for a, v in folds.items()}
    tm = tampered["B"][1]["test_mask"].copy()
    tm[np.flatnonzero(tm)[0]] = False
    tampered["B"][1] = dict(tampered["B"][1], test_mask=tm)
    ok("fold_membership_change_detected", H.fold_problems(tampered, table) == ["TEST_MASK:B:1"])
    leak = table["label"].copy()
    te = np.isin(leak, (0, 1, 2)) & (table["cohort"] == 2)
    leak[te] = (leak[te] + 1) % 3
    f_leak = Q.outer_run(X["B"], leak, table["cohort"], table["subject"], "walk", FAST, [0])
    ok("held_out_labels_cannot_reach_selection", f_leak[2]["selected"] == folds["B"][2]["selected"]
       and all(np.array_equal(a, b) for a, b in zip(f_leak[2]["proba"], folds["B"][2]["proba"]))
       and fold_digest([f_leak[0]]) != fold_digest([folds["B"][0]]))
    forced_a = [dict(f, selected=dict(f["selected"], all=(f["selected"]["all"] + 1) % len(FAST))) for f in folds["A"]]
    locked = H.predictions(folds["B"], index_from=forced_a)
    ok("locked_policy_uses_canonical_index", all(np.array_equal(locked[c], folds["B"][c]["proba"][forced_a[c]["selected"]["all"]].argmax(axis=1))
                                                 for c in range(4)))
    own = H.predictions(folds["B"])
    ok("primary_policy_uses_own_selection", all(np.array_equal(own[c], folds["B"][c]["proba"][folds["B"][c]["selected"]["all"]].argmax(axis=1))
                                                for c in range(4)))
    ok("locked_policy_refits_not_copies", any(not np.allclose(folds["B"][c]["proba"][k], folds["A"][c]["proba"][k])
                                              for c in range(4) for k in range(len(FAST))))

    # -- paired bootstrap ------------------------------------------------------------------------------------------------
    scorer = Q.Scorer(table, NB)
    conf = {a: scorer.conf(H.predictions(folds[a])) for a in H.ARMS}
    cons = H.contrasts(scorer, conf, "macro_f1")
    st = {a: scorer.stats(conf[a]) for a in H.ARMS}
    did_draws = ((st["D"][1] - st["B"][1]) - (st["C"][1] - st["A"][1])).mean(axis=1)
    did = cons["difference_in_differences"]["M"]
    ok("did_uses_shared_draws", np.allclose(did["ci95"], P.ci95(did_draws), atol=1e-12)
       and abs(did["point"] - ((st["D"][0] - st["B"][0]) - (st["C"][0] - st["A"][0])).mean()) < 1e-12)
    other = P.draw_counts([len(scorer.subjects[nm]) for nm in scorer.names], NB, 99)
    indep = np.empty((NB, 4))
    for c, nm in enumerate(scorer.names):
        f1 = {a: P.macro_f1_conf(P.apply_counts(other[c] if a in ("B", "D") else scorer.counts[c], conf[a][nm])) for a in H.ARMS}
        indep[:, c] = (f1["D"] - f1["B"]) - (f1["C"] - f1["A"])
    ok("independent_draws_would_differ", not np.allclose(P.ci95(indep.mean(axis=1)), did["ci95"]))
    ok("delta_root_and_did_consistent", abs(cons["delta_root_D_minus_B"]["M"]["point"] - cons["delta_canonical_C_minus_A"]["M"]["point"]
                                            - did["point"]) < 1e-12)

    # -- decision rule ------------------------------------------------------------------------------------------------------
    cl = lambda *a: H.classify(*a)[0]   # noqa: E731
    ok("classify_sensitive_did", cl(0.06, 0.01, 0.11, -0.12, -0.2, -0.05) == "CLAIM_SENSITIVE_TO_ROOT_SPEED"
       and cl(-0.07, -0.12, -0.02, -0.25, -0.3, -0.2) == "CLAIM_SENSITIVE_TO_ROOT_SPEED")
    ok("classify_sensitive_sign_reversal", cl(0.2, -0.01, 0.4, 0.05, 0.01, 0.1) == "CLAIM_SENSITIVE_TO_ROOT_SPEED")
    ok("classify_robust", cl(0.0, -0.03, 0.03, -0.15, -0.2, -0.1) == "CLAIM_ROBUST_TO_ROOT_SPEED"
       and cl(0.02, 0.005, 0.045, -0.16, -0.21, -0.1) == "CLAIM_ROBUST_TO_ROOT_SPEED")
    ok("classify_indeterminate", cl(0.0, -0.03, 0.03, -0.02, -0.06, 0.01) == "INDETERMINATE"
       and cl(0.04, 0.01, 0.07, -0.14, -0.2, -0.08) == "INDETERMINATE"
       and cl(0.0, -0.05, 0.03, -0.15, -0.2, -0.1) == "INDETERMINATE"
       and cl(0.03, -0.02, 0.09, -0.15, -0.2, -0.1) == "INDETERMINATE")

    # -- compare_tree --------------------------------------------------------------------------------------------------------
    def cmp(a, b):
        lv, fl = [], []
        H.compare_tree(a, b, "", 1e-5, lv, fl)
        return fl

    ok("compare_tree_rules", not cmp({"a": [1, 0.5, "x", None, True]}, {"a": [1, 0.5000004, "x", None, True]})
       and cmp({"a": 0.5}, {"a": 0.50002}) and cmp({"a": 1}, {"a": 1.0}) and cmp({"a": True}, {"a": 1})
       and cmp({"a": 1}, {"a": 1, "b": 2}) and cmp({"a": [1, 2]}, {"a": [1]}) and cmp({"a": None}, {"a": 0.0})
       and cmp({"a": 0.5}, {"a": float("nan")}) and not cmp({"a": 3}, {"a": np.int64(3)}))

    # -- baseline parity against the frozen engine's own run_analysis -------------------------------------------------------
    p1_result, _state = Q.run_analysis(table, configs=FAST, n_boot=NB)
    frozen_fab = jsonable(p1_result)
    report, passed = H.baseline_parity(table, {"A": folds["A"], "C": folds["C"]}, scorer, FAST, frozen_fab)
    ok("replay_equals_p1_run_analysis", passed and report["failed_values"] == 0 and report["max_abs_deviation"] == 0.0
       and report["compared_values"] > 350)

    def parity_after(fn):
        fz = copy.deepcopy(frozen_fab)
        fn(fz)
        return H.baseline_parity(table, {"A": folds["A"], "C": folds["C"]}, scorer, FAST, fz)

    def bump(path, delta):
        def f(fz):
            node = fz
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = node[path[-1]] + delta
        return f

    rep, good = parity_after(bump(["arms_inductive_strong", "S", "macro_f1", "per_cohort", "BMCLab", "point"], 2e-5))
    ok("parity_fails_beyond_tolerance", not good and any("arms_inductive_strong/S/macro_f1/per_cohort/BMCLab/point" in p
                                                          for p in rep["failing_paths"]))
    ok("parity_passes_within_tolerance", parity_after(bump(["primary", "contrast", "M", "point"], 8e-6))[1])
    ok("parity_fails_ci_shift", not parity_after(lambda fz: fz["primary"]["contrast"]["M"]["ci95"].__setitem__(0, fz["primary"]["contrast"]["M"]["ci95"][0] + 3e-5))[1])

    def other_config(fz):
        cur = fz["selected_configs"]["PS"]["PD-GaM"]["all"]
        fz["selected_configs"]["PS"]["PD-GaM"]["all"] = next(c[0] for c in FAST if c[0] != cur)

    ok("parity_fails_selected_config", not parity_after(other_config)[1])
    ok("parity_fails_confusion_cell", not parity_after(lambda fz: fz["confusions_by_cohort"]["S_strong"]["3DGait"][0].__setitem__(0, fz["confusions_by_cohort"]["S_strong"]["3DGait"][0][0] + 1))[1])
    ok("parity_fails_totals", not parity_after(bump(["totals", "walks_eligible_012"], 1))[1])
    ok("parity_fails_missing_section", not parity_after(lambda fz: fz.pop("per_config_macro_f1_exploratory"))[1])
    ok("parity_fails_config_param", not parity_after(lambda fz: fz["candidate_configs"][0]["params"].__setitem__("max_iter", 2001))[1])
    ok("parity_fails_inner_selection_score", not parity_after(bump(["selection_stability", "PS", "T-SDU-PD", "selected_inner_mean"], 1e-4))[1])
    swapped = copy.deepcopy(frozen_fab)
    swapped["arms_inductive_strong"]["S"], swapped["arms_inductive_strong"]["PS"] = (frozen_fab["arms_inductive_strong"]["PS"],
                                                                                     frozen_fab["arms_inductive_strong"]["S"])
    ok("parity_fails_swapped_arms", not H.baseline_parity(table, {"A": folds["A"], "C": folds["C"]}, scorer, FAST, swapped)[1])

    # -- portable loaders: the user's own P1 aggregate, download receipts and canonical folder -------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        saved = (H.P1_RESULT, H.P1_RESULT_SHA256, H.RECEIPT_FILES, H.CANONICAL_DIR)
        try:
            p1_file = tmp / "p1_strong_models.json"
            p1_file.write_text(json.dumps(frozen_fab))
            digest = hashlib.sha256(p1_file.read_bytes()).hexdigest()
            H.P1_RESULT, H.P1_RESULT_SHA256 = None, None
            ok("p1_loader_refuses_when_unconfigured", raises(H.frozen_p1, C.SourceError))
            H.P1_RESULT, H.P1_RESULT_SHA256 = p1_file, digest
            ok("p1_loader_reads_user_p1_pinned_by_sha256", H.frozen_p1() == frozen_fab)
            ok("frozen_pin_refuses_other_hash", raises(lambda: H.frozen_p1(want="0" * 64), C.SourceError))
            p1_file.write_text(json.dumps(frozen_fab) + " ")
            ok("p1_loader_refuses_a_changed_file", raises(H.frozen_p1, C.SourceError))
            rows = [{"cohort": c, "publisher_datafile": G.ROOT_PINS[c][1], "md5_pinned": G.ROOT_PINS[c][3],
                     "sha256": H.ROOT_SHA256[c], "integrity_ok": True} for c in C.COHORTS]
            first, rest = tmp / "fetch_originals_first.json", tmp / "fetch_originals_rest.json"
            first.write_text(json.dumps({"status": H.DOWNLOAD_PASS, "files": rows[:1]}))
            rest.write_text(json.dumps({"status": H.DOWNLOAD_PASS, "files": rows[1:] + [{"cohort": "x", "stop": "y"}]}))
            H.RECEIPT_FILES = (first, rest)
            ok("receipts_from_user_aggregates_match_sha256_pins", set(H.receipt_rows()) == set(C.COHORTS)
               and all(H.receipt_rows()[c]["sha256"] == H.ROOT_SHA256[c] for c in C.COHORTS))
            H.RECEIPT_FILES = (first,)
            ok("receipts_missing_cohort_refused", raises(H.receipt_rows, C.SourceError))
            H.RECEIPT_FILES = (first, tmp / "absent.json")
            ok("receipts_missing_file_refused", raises(H.receipt_rows, C.SourceError))
            rest.write_text(json.dumps({"status": "G_ROOT_DOWNLOAD_FAILED", "files": rows[1:]}))
            H.RECEIPT_FILES = (first, rest)
            ok("receipts_not_passed_refused", raises(H.receipt_rows, C.SourceError))
            rest.write_text(json.dumps({"status": H.DOWNLOAD_PASS, "files": [dict(r, sha256="0" * 64) for r in rows[1:]]}))
            ok("receipts_with_other_sha256_refused", raises(H.receipt_rows, C.SourceError))
            rest.write_text("{not json")
            ok("receipts_unreadable_refused", raises(H.receipt_rows, C.SourceError))
            H.CANONICAL_DIR = None
            ok("canonical_loader_refuses_when_unconfigured", raises(H.load_canonical, C.SourceError))
            H.CANONICAL_DIR = tmp / "absent"
            ok("canonical_loader_refuses_missing_files", raises(H.load_canonical, C.SourceError))
            prov = H.input_provenance()
            ok("provenance_records_user_digests_not_paths", prov["frozen_p1_summary"]["sha256"] == digest
               and str(tmp) not in json.dumps(prov) and set(prov["download_receipts_sha256"]) == {first.name, rest.name})
        finally:
            H.P1_RESULT, H.P1_RESULT_SHA256, H.RECEIPT_FILES, H.CANONICAL_DIR = saved
    sections = [("arms_inductive_strong", "S"), ("arms_inductive_strong", "PS"), ("per_config_macro_f1_exploratory", "S"),
                ("per_config_macro_f1_exploratory", "PS"), ("selected_configs", "S"), ("selected_configs", "PS"),
                ("selection_stability", "S"), ("selection_stability", "PS"), ("confusions_by_cohort", "S_strong"),
                ("confusions_by_cohort", "PS_strong"), ("primary", "contrast"), ("totals",), ("candidate_configs",)]
    ok("engine_p1_has_every_parity_section", all(H._dig(frozen_fab, s) is not None for s in sections))
    ok("engine_p1_selection_policies_match_engine",
       all(set(frozen_fab["selected_configs"][a][n]) == set(Q.policy_members(FAST)) for a in ("S", "PS") for n in C.COHORTS))

    # -- determinism across processes ----------------------------------------------------------------------------------------
    code = ("import sys; sys.path[:0] = [%r, %r]\n"
            "import root_speed_sensitivity_tests as T, root_speed_sensitivity as H, p1_core as Q\n"
            "c = T.make_canonical(); t = Q.build_table(c); r = T.make_root(c); s, _, f = H.join_root(t, c, r)\n"
            "X = H.arm_inputs(t, s)\n"
            "print(f, T.fold_digest(Q.outer_run(X['D'], t['label'], t['cohort'], t['subject'], 'walk', T.FAST, [0])))\n"
            % (str(SCRIPTS), str(HERE)))
    child = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=900,
                           env=dict(os.environ, PYTHONHASHSEED="0"))
    ok("cross_process_determinism", child.returncode == 0 and child.stdout.split() == ["None", fold_digest(folds["D"])])

    # -- end-to-end run on fabricated inputs ---------------------------------------------------------------------------------
    calls = []

    def counting_fit(*a, **k):
        calls.append(1)
        return Q.outer_run(*a, **k)

    common = dict(frozen_loader=lambda: frozen_fab, receipts=lambda: None, configs=FAST, n_boot=NB,
                  check_stops=fab_check_stops, refit_seeds=(1, 2), progress=lambda m: None)
    out = H.run(META, canonical_loader=lambda: canon, root_loader=lambda: root, fit=counting_fit, **common)
    ok("end_to_end_complete", out["status"] == H.STATUS_COMPLETE and out["scored_root_arms"] and len(calls) == 4
       and out["baseline_parity"]["verdict"] == "PASS" and out["input_gate"]["verdict"] == "PASS"
       and out["substitution_gate"]["verdict"] == "PASS")
    prim = out["primary_source_only"]
    mf = prim["contrasts"]["macro_f1"]
    arm_m = {a: prim["arms"][a]["macro_f1"]["M"]["point"] for a in H.ARMS}
    ok("end_to_end_did_arithmetic", abs(mf["difference_in_differences"]["M"]["point"]
                                        - ((arm_m["D"] - arm_m["B"]) - (arm_m["C"] - arm_m["A"]))) < 1e-12
       and abs(mf["delta_canonical_C_minus_A"]["M"]["point"] - frozen_fab["primary"]["contrast"]["M"]["point"]) <= 5e-7)
    ok("end_to_end_reports_both_policies_and_metrics",
       set(prim["contrasts"]) == set(H.CONTRAST_METRICS) and set(out["secondary_locked_policy"]["arms"]) == {"B", "D"}
       and out["secondary_locked_policy"]["locked_configs"]["B"] == {n: prim["selection"]["A"]["by_held_out_cohort"][n]["selected"]
                                                                     for n in C.COHORTS}
       and prim["decision"]["verdict"] in ("CLAIM_ROBUST_TO_ROOT_SPEED", "CLAIM_SENSITIVE_TO_ROOT_SPEED", "INDETERMINATE")
       and len(out["refit_seed_sensitivity"]["did_by_seed"]) == 2)
    ok("end_to_end_grade3_excluded_from_scoring", out["totals"]["walks_all_labels"] == n
       and out["totals"]["walks_eligible_012"] == n - 2)
    out["compute"] = IC.Receipt().block()
    fin = H.finalize(out)
    text = IC.render(fin)
    ok("aggregate_renders_and_validates", fin["status"] == H.STATUS_COMPLETE and not G.disclosure_problems(fin)
       and len(text.encode()) < IC.MAX_BYTES)
    ok("aggregate_has_no_identifier_strings", SECRET_SUBJ not in text and SECRET_WALK not in text and '"770' not in text)
    ok("aggregate_has_no_row_level_list", max_list_len(fin) <= 16)
    with tempfile.TemporaryDirectory() as tmp:
        import run_provenance as RP
        dest = Path(tmp) / "root_speed.json"
        RP.save_to(dest, fin)
        ok("save_to_writes_validated_aggregate_once", dest.read_text(encoding="utf-8") == text
           and raises(lambda: RP.save_to(dest, fin), FileExistsError))
    for label, inject in (("per_walk_key", lambda o: o["primary_source_only"].__setitem__("per_walk_predictions", [0, 1])),
                          ("long_list", lambda o: o.__setitem__("speeds", list(range(100)))),
                          ("small_count", lambda o: o.__setitem__("n_walks_flagged", 3)),
                          ("exact_small_cell", lambda o: o.__setitem__("x", {"mode": "exact", "cells": {"a": 2, "b": 9}})),
                          ("subject_key", lambda o: o.__setitem__("subject_ids", ["a"]))):
        bad = copy.deepcopy(out)
        inject(bad)
        fb = H.finalize(bad)
        ok(f"privacy_blocks_{label}", fb["status"] == H.STATUS_PRIVACY and "primary_source_only" not in fb
           and IC.render(fb) is not None)

    # -- blocked paths: nothing scored, statuses recorded ------------------------------------------------------------------
    calls.clear()
    bad_root = copy.deepcopy(root)
    drop(bad_root)
    o = H.run(META, canonical_loader=lambda: canon, root_loader=lambda: bad_root, fit=counting_fit, **common)
    ok("input_blocked_no_fit", o["status"] == H.STATUS_INPUT and o["failed_gate"] == "INPUT_JOIN" and not calls
       and not o["scored_root_arms"] and "primary_source_only" not in o and o["stop_reason"] == "RECORD_SETS:BMCLab")
    fz = copy.deepcopy(frozen_fab)
    bump(["arms_inductive_strong", "PS", "macro_f1", "M", "point"], 5e-5)(fz)
    o = H.run(META, canonical_loader=lambda: canon, root_loader=lambda: root, fit=counting_fit,
              **dict(common, frozen_loader=lambda: fz))
    ok("parity_blocked_root_arms_never_fitted", o["status"] == H.STATUS_PARITY and len(calls) == 2
       and o["baseline_parity"]["verdict"] == "FAIL" and "primary_source_only" not in o and not o["scored_root_arms"])

    def raiser(code):
        def f():
            raise C.SourceError(code)
        return f

    ok("root_file_problem_blocks", H.run(META, canonical_loader=lambda: canon, root_loader=raiser("ROOT_FILE_PIN:BMCLab"),
                                         **common)["failed_gate"] == "ROOT_INPUTS")
    ok("canonical_file_problem_blocks", H.run(META, canonical_loader=raiser("SHA256_MISMATCH:BMCLab"), root_loader=lambda: root,
                                              **common)["status"] == H.STATUS_INPUT)
    ok("frozen_summary_problem_blocks", H.run(META, canonical_loader=lambda: canon, root_loader=lambda: root,
                                              **dict(common, frozen_loader=raiser("FROZEN_P1_SUMMARY_PIN")))["status"] == H.STATUS_PARITY)
    ok("p1_pre_score_check_blocks", H.run(META, canonical_loader=lambda: canon, root_loader=lambda: root,
                                          **dict(common, check_stops=lambda t: (_ for _ in ()).throw(C.SourceError("P1_WALK_TOTALS"))))
       ["failed_gate"] == "P1_PRE_SCORE_CHECKS")

    # -- root loader pins (temporary files only) -----------------------------------------------------------------------------
    orig = G.root_path
    try:
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "root.pkl"
            G.root_path = lambda name: Path(tmp) / "absent.pkl"
            ok("root_loader_missing_file", raises(H.load_roots, C.SourceError))
            fake.write_bytes(b"\x80\x04N.")
            G.root_path = lambda name: fake
            ok("root_loader_pin_mismatch", raises(H.load_roots, C.SourceError))
    finally:
        G.root_path = orig
    ok("root_paths_point_to_configured_sentinel_folder", all(str(G.root_path(c)).startswith(DATA) for c in C.COHORTS))
    test_cli(canon, root, frozen_fab)
    print(f"ROOT_SPEED_SENSITIVITY_TESTS_PASS n={PASS}")


def test_cli(canon, root, frozen_fab):
    """root-speed through the command line on fabricated pinned files, with the learners reduced as above."""
    import provenance_fixtures as FX
    import release_context as X
    import run_provenance as RP
    with tempfile.TemporaryDirectory() as d:
        data, out = Path(d) / "data", Path(d) / "out"
        argv = ["root-speed", "--data-dir", str(data), "--out-dir", str(out)]
        code, text = FX.quiet_main(RP, argv)
        ok("cli_refuses_without_user_p0_p1_before_data", code == 20 and "RELEASE_UPSTREAM_MISSING" in text
           and not (out / "root_speed.json").exists())
        saved = FX.snapshot()
        saved_sha, saved_configs, saved_run = dict(H.ROOT_SHA256), Q.CONFIGS, H.run
        saved_attrs = (H.P1_RESULT, H.P1_RESULT_SHA256, H.RECEIPT_FILES, H.CANONICAL_DIR)
        try:
            FX.layout(data)
            FX.pin_canonical(data, canon)
            shas = FX.pin_roots(root)
            H.ROOT_SHA256.update(shas)
            FX.receipt(out, RP.result_name("fetch-originals", "first"), ("3DGait",), shas)
            FX.receipt(out, RP.result_name("fetch-originals", "rest"), ("BMCLab", "T-SDU-PD", "PD-GaM"), shas)
            Q.CONFIGS = FAST            # the upstream contract compares candidates with the engine list
            p0, _ = FX.upstream(out)
            files = {s: {"name": f[0], "bytes": f[1], "sha256": f[2], "md5": f[3]} for s, f in C.FILES.items()}
            p1 = dict(frozen_fab, status="P1_COMPLETE", n_boot=P.N_BOOT, seed=P.SEED,
                      provenance={"hf_commit": C.HF_COMMIT, "files": files, "p0_result_sha256": X.digest(out / "p0.json")})
            (out / "p1_strong_models.json").write_text(json.dumps(p1))
            H.run = lambda meta: saved_run(meta, configs=FAST, n_boot=NB, check_stops=fab_check_stops, refit_seeds=(1, 2),
                                           progress=lambda m: None)
            code, text = FX.quiet_main(RP, argv)
            res = out / "root_speed.json"
            agg = json.loads(res.read_text()) if res.is_file() else {}
            ok("cli_end_to_end_complete", code == 0 and agg.get("status") == H.STATUS_COMPLETE
               and agg["baseline_parity"]["verdict"] == "PASS" and agg["input_gate"]["verdict"] == "PASS"
               and agg.get("card") == "root-speed" and agg.get("release_scope") == RP.SCOPE)
            ok("cli_records_user_upstream_digests", agg["provenance"]["frozen_p1_summary"]["sha256"]
               == X.digest(out / "p1_strong_models.json")
               and agg["release_provenance"]["upstream_sha256"]["p0"] == X.digest(out / "p0.json"))
            text = res.read_text()
            ok("cli_aggregate_has_no_identifier_or_path", SECRET_SUBJ not in text and SECRET_WALK not in text and d not in text)
            code, text = FX.quiet_main(RP, argv)
            ok("cli_existing_result_not_overwritten", code == 10)
            res.unlink()
            (out / RP.result_name("fetch-originals", "rest")).unlink()
            code, text = FX.quiet_main(RP, argv)
            agg = json.loads(res.read_text()) if res.is_file() else {}
            ok("cli_missing_receipt_blocks_with_exit_20", code == 20 and agg.get("status") == H.STATUS_INPUT
               and agg.get("failed_gate") == "CANONICAL_INPUTS" and not agg.get("scored_root_arms"))
        finally:
            H.ROOT_SHA256.clear()
            H.ROOT_SHA256.update(saved_sha)
            Q.CONFIGS, H.run = saved_configs, saved_run
            H.P1_RESULT, H.P1_RESULT_SHA256, H.RECEIPT_FILES, H.CANONICAL_DIR = saved_attrs
            FX.restore(saved)


if __name__ == "__main__":
    main()
