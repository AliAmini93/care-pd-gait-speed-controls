"""P0 analysis core: features, LOCO arms, bootstrap, verdict.

See docs/protocol.md. Subject identity is reduced to an anonymous integer index on
load and nothing returned by run_analysis is participant-level.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common as C

SEED = 20261030
N_BOOT = 2000
N_PERM = 200
PERM_BUDGET_S = 2400
FITTED_ARMS = ("D", "S", "Sz", "P", "PS")
ARMS = ("C", "D", "S", "Sz", "P", "PS")
RGB_COHORTS = ("3DGait", "PD-GaM", "T-SDU-PD")
S1_MIN = 0.05
C1_MIN = 0.05


# ---- features -----------------------------------------------------------------
def speed_and_duration(trans, fps):
    w = max(1, round(fps / 2))
    xz = trans[:, [0, 2]]
    if w > 1:
        left = (w - 1) // 2
        padded = np.pad(xz, ((left, w - 1 - left), (0, 0)), mode="edge")
        kernel = np.ones(w) / w
        xz = np.column_stack([np.convolve(padded[:, j], kernel, mode="valid") for j in (0, 1)])
    path = float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum())
    duration = trans.shape[0] / fps
    return path / duration, duration


def pose_stats(pose):
    p = pose[:, 3:]
    return np.concatenate([p.std(axis=0), np.percentile(p, 95, axis=0) - np.percentile(p, 5, axis=0)])


def build_table(cohort_objs):
    """Per-walk features from {cohort: nested dict}. Subject keys become anonymous indices."""
    cohort, subject, label, speed, duration, poses = [], [], [], [], [], []
    nonfinite = {c: 0 for c in C.COHORTS}
    next_subject = 0
    for ci, name in enumerate(C.COHORTS):
        index = {}
        for key, rec in C.walks_of(cohort_objs[name]):
            lab = C.read_label(rec["UPDRS_GAIT"])
            pose, trans = C.read_arrays(rec)
            if not (np.isfinite(pose).all() and np.isfinite(trans).all()):
                nonfinite[name] += 1
                continue
            if key not in index:
                index[key] = next_subject
                next_subject += 1
            s, d = speed_and_duration(trans, C.read_fps(rec["fps"]))
            cohort.append(ci)
            subject.append(index[key])
            label.append(-1 if lab is None else lab)
            speed.append(s)
            duration.append(d)
            poses.append(pose_stats(pose))
    return {
        "cohort": np.asarray(cohort), "subject": np.asarray(subject), "label": np.asarray(label),
        "speed": np.asarray(speed), "duration": np.asarray(duration), "pose": np.asarray(poses),
        "nonfinite": nonfinite,
    }


def check_stops(t):
    """Stop checks on the feature table. Runs before any model is fitted."""
    for ci, name in enumerate(C.COHORTS):
        if t["nonfinite"][name]:
            raise C.SourceError(f"NON_FINITE:{name}")
        mask = t["cohort"] == ci
        if len(np.unique(t["subject"][mask])) != C.EXPECTED_SUBJECTS[name]:
            raise C.SourceError(f"SUBJECT_COUNT:{name}")
        got = [int((t["label"][mask] == k).sum()) for k in range(4)]
        if any(abs(g - e) > C.CLASS_CELL_TOLERANCE for g, e in zip(got, C.EXPECTED_CLASS_WALKS[name])):
            raise C.SourceError(f"CLASS_COUNT:{name}")
        if (t["speed"][mask] <= C.LOW_SPEED).mean() > C.MAX_LOW_SPEED_SHARE:
            raise C.SourceError(f"LOW_SPEED_SHARE:{name}")
        if iqr(t["speed"][mask]) <= 0:
            raise C.SourceError(f"ZERO_SPEED_IQR:{name}")
    total = int((t["label"] >= 0).sum())
    lo, hi = C.TOTAL_WALKS_RANGE
    if not lo <= total <= hi:
        raise C.SourceError("TOTAL_WALKS")


def iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def scale_within_cohort(speed, cohort):
    """Median/IQR scaling within each cohort. Takes no label argument, so it cannot use one."""
    out = np.empty_like(speed, dtype=float)
    for ci in np.unique(cohort):
        m = cohort == ci
        out[m] = (speed[m] - np.median(speed[m])) / iqr(speed[m])
    return out


# ---- metrics --------------------------------------------------------------------
def confusion_by_subject(y, pred, subject, subjects):
    """(n_subjects, 3, 3) confusion counts; `subjects` fixes the row order."""
    pos = {s: i for i, s in enumerate(subjects)}
    out = np.zeros((len(subjects), 3, 3))
    for yi, pi, si in zip(y, pred, subject):
        out[pos[si], int(yi), int(pi)] += 1
    return out


def macro_f1_conf(conf):
    """Macro-F1 over {0,1,2} from (...,3,3) confusion. Empty classes score 0."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    denom = conf.sum(axis=-1) + conf.sum(axis=-2)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return f1.mean(axis=-1)


def plugin_floor(true_counts, q=None):
    """Plug-in expected macro-F1 of an independent guesser with class shares q (uniform by default)."""
    true_counts = np.asarray(true_counts, dtype=float)
    p = true_counts / true_counts.sum(axis=-1, keepdims=True)
    q = np.full(3, 1 / 3) if q is None else np.asarray(q, dtype=float)
    denom = p + q
    f1 = np.divide(2 * p * q, denom, out=np.zeros_like(denom), where=denom > 0)
    return f1.mean(axis=-1)


# ---- LOCO arms --------------------------------------------------------------------
def fit_predict(Xtr, ytr, Xte, warned):
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lr.fit(scaler.transform(Xtr), ytr)
    warned[0] += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    proba = np.zeros((Xte.shape[0], 3))
    proba[:, lr.classes_.astype(int)] = lr.predict_proba(scaler.transform(Xte))
    return proba.argmax(axis=1), proba


def arm_features(t):
    sz = scale_within_cohort(t["speed"], t["cohort"])
    return {
        "D": t["duration"][:, None], "S": t["speed"][:, None], "Sz": sz[:, None],
        "P": t["pose"], "PS": np.column_stack([t["pose"], t["speed"]]),
    }


def loco_predictions(t):
    """Per held-out cohort: predictions on eligible walks. Held-out labels are never passed in."""
    eligible = np.isin(t["label"], (0, 1, 2))
    feats = arm_features(t)
    out, warned = {}, [0]
    for ci, name in enumerate(C.COHORTS):
        te = eligible & (t["cohort"] == ci)
        tr = eligible & (t["cohort"] != ci)
        ytr = t["label"][tr]
        rec = {"test_mask": te, "pred": {}, "proba": {},
               "train_counts": np.bincount(ytr, minlength=3)}
        rec["pred"]["C"] = np.full(te.sum(), int(np.bincount(ytr, minlength=3).argmax()))
        for arm, X in feats.items():
            rec["pred"][arm], rec["proba"][arm] = fit_predict(X[tr], ytr, X[te], warned)
        out[name] = rec
    return out, warned[0]


def subject_averaged_pred(proba, subject):
    avg = np.empty_like(proba)
    for s in np.unique(subject):
        m = subject == s
        avg[m] = proba[m].mean(axis=0)
    return avg.argmax(axis=1)


# ---- bootstrap --------------------------------------------------------------------
def draw_counts(sizes, n_boot, seed):
    """Per cohort, an (n_boot, n_subjects) array of how often each subject was drawn."""
    rng = np.random.default_rng(seed)
    out = [np.zeros((n_boot, n), dtype=int) for n in sizes]
    for b in range(n_boot):
        for c, n in enumerate(sizes):
            out[c][b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return out


def apply_counts(counts, conf):
    """(n_boot, 3, 3) pooled confusion for one cohort and arm."""
    return np.einsum("bs,sij->bij", counts, conf)


def ci95(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def summary(point, draws):
    return {"point": float(point), "ci95": ci95(draws)}


# ---- verdict ------------------------------------------------------------------------
def decide(d_s, d_s_lo, d_ps, d_ps_lo, d_sz, d_sz_lo):
    s1 = bool(d_s >= S1_MIN and d_s_lo > 0)
    c1 = bool(d_ps >= C1_MIN and d_ps_lo > 0)
    verdict = {(True, False): "SPEED_FLOOR_HOLDS", (True, True): "POSE_ADDS_BEYOND_SPEED",
               (False, True): "POSE_ONLY_SIGNAL", (False, False): "AT_FLOOR"}[(s1, c1)]
    sz_would_pass = bool(d_sz >= S1_MIN and d_sz_lo > 0)
    return {"S1": s1, "C1": c1, "verdict": verdict,
            "wording_qualifier": "SPEED_SITE_DEPENDENT" if (not s1 and sz_would_pass) else None}


# ---- within-subject control -----------------------------------------------------------
def within_subject_control(t, n_boot=N_BOOT):
    eligible = np.isin(t["label"], (0, 1, 2))
    rhos, cohorts, undefined = [], [], 0
    for s in np.unique(t["subject"][eligible]):
        m = eligible & (t["subject"] == s)
        if m.sum() < 3 or len(np.unique(t["label"][m])) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            rho = stats.spearmanr(t["speed"][m], t["label"][m]).statistic
        if not np.isfinite(rho):
            undefined += 1
            continue
        rhos.append(float(rho))
        cohorts.append(int(t["cohort"][m][0]))
    rhos, cohorts = np.asarray(rhos), np.asarray(cohorts)
    if rhos.size == 0:
        return {"n_subjects": 0, "n_undefined": undefined, "supports_negative": False}
    neg, pos = int((rhos < 0).sum()), int((rhos > 0).sum())
    p = float(stats.binomtest(neg, neg + pos, 0.5).pvalue) if neg + pos else 1.0
    rng = np.random.default_rng([SEED, 2])
    groups = [np.flatnonzero(cohorts == c) for c in np.unique(cohorts)]
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = np.concatenate([rhos[rng.choice(g, g.size)] for g in groups]).mean()
    per_cohort = {}
    for ci, name in enumerate(C.COHORTS):
        r = rhos[cohorts == ci]
        per_cohort[name] = {"n_subjects": int(r.size), "n_negative": int((r < 0).sum()),
                            "n_positive": int((r > 0).sum()),
                            "mean_rho": float(r.mean()) if r.size else None}
    return {
        "n_subjects": int(rhos.size), "n_undefined": undefined,
        "n_negative": neg, "n_positive": pos, "n_zero": int((rhos == 0).sum()),
        "share_negative": neg / (neg + pos) if neg + pos else None,
        "sign_test_two_sided_p": p,
        "mean_rho": float(rhos.mean()), "mean_rho_ci95": ci95(means),
        "supports_negative": bool(neg > pos and p < 0.05),
        "per_cohort": per_cohort,
    }


# ---- descriptive null calibration --------------------------------------------------------
def null_calibration(t, observed, n_perm=N_PERM, budget_s=PERM_BUDGET_S):
    """Refit the arms with training labels permuted within each cohort. Never part of the verdict.

    Keeps each cohort's class mix and the real held-out labels, and removes any walk-level
    link between features and training labels. Shows what the frozen margins look like when
    the fitted arms have learned nothing beyond cohort composition. `observed` holds the
    unrounded M per arm plus F_U. Stops early on the time budget and reports the count done.
    """
    rng = np.random.default_rng([SEED, 3])
    eligible = np.isin(t["label"], (0, 1, 2))
    feats = arm_features(t)
    names = list(C.COHORTS)
    masks = [(eligible & (t["cohort"] == c), eligible & (t["cohort"] != c)) for c in range(len(names))]
    pools = [np.flatnonzero(eligible & (t["cohort"] == c)) for c in range(len(names))]
    scores = {a: [] for a in FITTED_ARMS}
    warned, start, done = [0], time.monotonic(), 0
    for _ in range(n_perm):
        if time.monotonic() - start > budget_s:
            break
        shuffled = t["label"].copy()
        for pool in pools:
            shuffled[pool] = rng.permutation(shuffled[pool])
        row = {a: [] for a in FITTED_ARMS}
        for te, tr in masks:
            for a in FITTED_ARMS:
                pred, _ = fit_predict(feats[a][tr], shuffled[tr], feats[a][te], warned)
                cm = np.zeros((3, 3))
                np.add.at(cm, (t["label"][te], pred), 1)
                row[a].append(macro_f1_conf(cm))
        for a in FITTED_ARMS:
            scores[a].append(np.mean(row[a]))
        done += 1
    if done == 0:
        return {"n_permutations_done": 0, "n_permutations_planned": n_perm, "seed": [SEED, 3]}
    scores = {a: np.asarray(v) for a, v in scores.items()}

    def describe(null, obs):
        return {"null_mean": float(null.mean()), "null_p2_5_p97_5": ci95(null),
                "observed": float(obs),
                "share_null_at_or_above_observed": float((null >= obs).mean()),
                "rank_p_one_sided": float((1 + (null >= obs).sum()) / (null.size + 1))}

    f_u = observed["F_U"]
    return {
        "n_permutations_done": done, "n_permutations_planned": n_perm, "seed": [SEED, 3],
        "n_convergence_warnings": warned[0],
        "arms_M": {a: describe(scores[a], observed[a]) for a in FITTED_ARMS},
        "S_minus_F_U": dict(describe(scores["S"] - f_u, observed["S"] - f_u),
                            share_null_at_or_above_margin=float((scores["S"] - f_u >= S1_MIN).mean())),
        "PS_minus_S": dict(describe(scores["PS"] - scores["S"], observed["PS"] - observed["S"]),
                           share_null_at_or_above_margin=float((scores["PS"] - scores["S"] >= C1_MIN).mean())),
    }


# ---- full analysis ---------------------------------------------------------------------
def run_analysis(t, n_boot=N_BOOT):
    loco, n_conv = loco_predictions(t)
    names = list(C.COHORTS)
    subjects = {}
    conf = {}
    true_counts = {}
    for name in names:
        rec = loco[name]
        te = rec["test_mask"]
        subj, y = t["subject"][te], t["label"][te]
        subjects[name] = np.unique(subj)
        conf[name] = {a: confusion_by_subject(y, rec["pred"][a], subj, subjects[name]) for a in ARMS}
        for a in ("S", "PS"):
            conf[name][a + "_subj_avg"] = confusion_by_subject(
                y, subject_averaged_pred(rec["proba"][a], subj), subj, subjects[name])
        true_counts[name] = np.bincount(y, minlength=3)

    all_arms = ARMS + ("S_subj_avg", "PS_subj_avg")
    point = {a: np.array([macro_f1_conf(conf[n][a].sum(axis=0)) for n in names]) for a in all_arms}
    point["F_U"] = np.array([plugin_floor(true_counts[n]) for n in names])

    counts = draw_counts([len(subjects[n]) for n in names], n_boot, SEED)
    draws, absent = {a: np.empty((n_boot, len(names))) for a in all_arms + ("F_U",)}, {}
    for c, name in enumerate(names):
        pooled = {a: apply_counts(counts[c], conf[name][a]) for a in all_arms}
        for a in all_arms:
            draws[a][:, c] = macro_f1_conf(pooled[a])
        drawn_true = pooled["S"].sum(axis=2)
        draws["F_U"][:, c] = plugin_floor(drawn_true)
        absent[name] = int((drawn_true == 0).any(axis=1).sum())

    rgb = [names.index(n) for n in RGB_COHORTS]
    arms_out, deltas = {}, {}
    for a in ("F_U",) + all_arms:
        arms_out[a] = {
            "per_cohort": {n: summary(point[a][c], draws[a][:, c]) for c, n in enumerate(names)},
            "M": summary(point[a].mean(), draws[a].mean(axis=1)),
            "M_rgb_only": summary(point[a][rgb].mean(), draws[a][:, rgb].mean(axis=1)),
        }

    def delta(a, b):
        return {
            "per_cohort": {n: summary(point[a][c] - point[b][c], draws[a][:, c] - draws[b][:, c])
                           for c, n in enumerate(names)},
            "M": summary(point[a].mean() - point[b].mean(),
                         draws[a].mean(axis=1) - draws[b].mean(axis=1)),
            "M_rgb_only": summary(point[a][rgb].mean() - point[b][rgb].mean(),
                                  draws[a][:, rgb].mean(axis=1) - draws[b][:, rgb].mean(axis=1)),
        }

    pairs = [("S", "F_U"), ("PS", "S"), ("Sz", "F_U"), ("D", "S"), ("D", "F_U"), ("P", "F_U"),
             ("PS", "F_U"), ("C", "F_U"), ("Sz", "S"), ("P", "S"),
             ("S_subj_avg", "F_U"), ("PS_subj_avg", "S_subj_avg")]
    for a, b in pairs:
        deltas[f"{a}_minus_{b}"] = delta(a, b)

    d_s = deltas["S_minus_F_U"]["M"]
    d_ps = deltas["PS_minus_S"]["M"]
    d_sz = deltas["Sz_minus_F_U"]["M"]
    decision = decide(d_s["point"], d_s["ci95"][0], d_ps["point"], d_ps["ci95"][0],
                      d_sz["point"], d_sz["ci95"][0])

    # Descriptive reference guessers: never part of the verdict.
    rng = np.random.default_rng([SEED, 1])
    runs = np.empty((n_boot, len(names)))
    for c, name in enumerate(names):
        y = t["label"][loco[name]["test_mask"]]
        for b in range(n_boot):
            guess = rng.integers(0, 3, y.size)
            cm = np.zeros((3, 3))
            np.add.at(cm, (y, guess), 1)
            runs[b, c] = macro_f1_conf(cm)
    prior = np.array([plugin_floor(true_counts[n], loco[n]["train_counts"] / loco[n]["train_counts"].sum())
                      for n in names])
    reference = {
        "observed_uniform_runs": {
            "per_cohort": {n: {"mean": float(runs[:, c].mean()), "p2_5_p97_5": ci95(runs[:, c])}
                           for c, n in enumerate(names)},
            "M": {"mean": float(runs.mean()), "p2_5_p97_5": ci95(runs.mean(axis=1))},
        },
        "train_prior_plugin": {"per_cohort": dict(zip(names, map(float, prior))), "M": float(prior.mean())},
    }

    eligible = np.isin(t["label"], (0, 1, 2))
    cohort_info = {}
    for c, name in enumerate(names):
        all_mask = t["cohort"] == c
        el = all_mask & eligible
        mixed = sum(len(np.unique(t["label"][el & (t["subject"] == s)])) >= 2 for s in subjects[name])
        cohort_info[name] = {
            "subjects_all_labels": int(len(np.unique(t["subject"][all_mask]))),
            "subjects_eligible": int(len(subjects[name])),
            "subjects_with_mixed_labels_012": int(mixed),
            "walks_all_labels": int(all_mask.sum()),
            "walks_eligible": int(el.sum()),
            "class_walks_0123": [int((t["label"][all_mask] == k).sum()) for k in range(4)],
            "median_speed": float(np.median(t["speed"][all_mask])),
            "speed_iqr": iqr(t["speed"][all_mask]),
            "median_duration_s": float(np.median(t["duration"][all_mask])),
            "walks_speed_le_0_01": int((t["speed"][all_mask] <= C.LOW_SPEED).sum()),
            "bootstrap_draws_with_absent_class": absent[name],
        }

    return {
        "status": "P0_COMPLETE",
        "decision": decision,
        "thresholds": {"S1_min": S1_MIN, "C1_min": C1_MIN},
        "totals": {"walks_all_labels": int((t["label"] >= 0).sum()),
                   "walks_eligible_012": int(eligible.sum()),
                   "walks_class_3": int((t["label"] == 3).sum()),
                   "subjects_all": int(len(np.unique(t["subject"]))),
                   "subjects_eligible": int(len(np.unique(t["subject"][eligible])))},
        "cohorts": cohort_info,
        "arms": arms_out,
        "paired_deltas": deltas,
        "within_subject_control": within_subject_control(t, n_boot),
        "reference_guessers_descriptive": reference,
        "n_convergence_warnings": n_conv,
        "n_boot": n_boot,
        "seed": SEED,
    }
