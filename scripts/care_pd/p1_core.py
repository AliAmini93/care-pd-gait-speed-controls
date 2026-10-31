"""P1 analysis core: stronger learners with a matched speed-only control.

See docs/protocol.md. The P0 modules are imported and never changed. Selection functions take source-site arrays only, so a
held-out cohort cannot reach them. Subject identity stays an anonymous integer in
memory, and nothing returned by run_analysis is walk-level or person-level.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import common as C
import p0_core as P

SEED = P.SEED
N_BOOT = P.N_BOOT
N_NULL = 100
NULL_BUDGET_S = 5400
MIN_NULL_FOR_GO = 39
REFIT_SEEDS = (1, 2, 3, 4, 5)
SCREEN_MIN = 0.05
NULL_ALPHA = 0.05
TIE_EPS = 1e-12
PARITY_TOL = 1e-9
RGB_COHORTS = P.RGB_COHORTS
EXPECTED_TOTALS = {"walks_all": 2953, "walks_eligible": 2908, "subjects_eligible": 107}
VERSIONS = {"numpy": "1.26.4", "scipy": "1.16.0", "sklearn": "1.6.1"}

LR_BASE = {"solver": "lbfgs", "max_iter": 2000}
RF_BASE = {"n_estimators": 300, "max_features": "sqrt", "bootstrap": True, "n_jobs": 1}
HGB_BASE = {"learning_rate": 0.05, "early_stopping": False}
# Frozen candidate list. Its order is the tie-break order, simplest first.
CONFIGS = (
    ("LR_C1", "LR", dict(LR_BASE, C=1.0)),
    ("LR_C001", "LR", dict(LR_BASE, C=0.01)),
    ("RF_leaf25", "RF", dict(RF_BASE, min_samples_leaf=25)),
    ("RF_leaf5", "RF", dict(RF_BASE, min_samples_leaf=5)),
    ("HGB_L7_it200", "HGB", dict(HGB_BASE, max_leaf_nodes=7, max_iter=200, min_samples_leaf=20, l2_regularization=1.0)),
    ("HGB_L15_it100", "HGB", dict(HGB_BASE, max_leaf_nodes=15, max_iter=100, min_samples_leaf=50, l2_regularization=10.0)),
)
FAMILIES = ("LR", "RF", "HGB")
INDUCTIVE_ARMS = ("S", "S_alt", "D", "S_D", "P", "PS", "PS_alt", "K", "KS")
TRANSDUCTIVE_ARMS = ("Sz", "PzSz")
ARMS = INDUCTIVE_ARMS + TRANSDUCTIVE_ARMS
PERSON_SCHEME_ARMS = ("S", "PS", "KS")
PARITY_ARMS = ("D", "S", "Sz", "P", "PS")

# SMPL joint indices (official 24-joint order). Left then right.
JOINTS = (("hip", 1, 2), ("knee", 4, 5), ("ankle", 7, 8), ("shoulder", 16, 17), ("elbow", 18, 19))
SPINE = (3, 6, 9)
KIN_RATE_HZ = 25
FREQ_GRID_HZ = 0.025
FREQ_BAND = (0.3, 3.0)
FREQ_PEAK_HALF_WIDTH = 0.1
KIN_NAMES = tuple(
    [f"amp_{j[0]}" for j in JOINTS] + [f"asym_{j[0]}" for j in JOINTS] + [f"angspeed_{j[0]}" for j in JOINTS]
    + ["stride_freq_hz", "stride_regularity", "vertical_bob", "lateral_sway", "trunk_mean", "trunk_range"])


# ---- features -----------------------------------------------------------------
def speed_alt(trans, fps):
    """Median frame speed of the smoothed (x, z) path, using only fully covered windows.

    Same window as P0 S. No edge padding and no division by whole-walk duration, so the
    P0 endpoint shrinkage cannot enter. Not assumed unbiased; units are those of `trans`.
    """
    w = max(1, round(fps / 2))
    if trans.shape[0] < w + 1:
        raise C.SourceError("S_ALT_UNDEFINED")
    xz = trans[:, [0, 2]]
    if w > 1:
        kernel = np.ones(w) / w
        xz = np.column_stack([np.convolve(xz[:, j], kernel, mode="valid") for j in (0, 1)])
    return float(np.median(np.linalg.norm(np.diff(xz, axis=0), axis=1)) * fps)


def _rotvec(pose, joint):
    return pose[:, 3 * joint:3 * joint + 3]


def _spread(x):
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def _geodesic_rate(rotvec, dt):
    """Mean geodesic angle between consecutive rotations, per second."""
    r = Rotation.from_rotvec(rotvec)
    return float(np.linalg.norm((r[:-1].inv() * r[1:]).as_rotvec(), axis=1).mean() / dt)


def _stride_frequency(signal, dt):
    """Peak of the band-limited spectrum on a fixed 0.025 Hz grid, and its power share."""
    x = signal - signal.mean()
    n = max(x.size, int(round(1 / (FREQ_GRID_HZ * dt))))
    power = np.abs(np.fft.rfft(x * np.hanning(x.size), n=n)) ** 2
    freq = np.fft.rfftfreq(n, dt)
    band = (freq >= FREQ_BAND[0]) & (freq <= FREQ_BAND[1])
    total = power[band].sum()
    if total <= 0:
        return 0.0, 0.0
    peak = freq[band][int(np.argmax(power[band]))]
    near = band & (np.abs(freq - peak) <= FREQ_PEAK_HALF_WIDTH)
    return float(peak), float(power[near].sum() / total)


def _detrended_spread(y, t):
    slope, intercept = np.polyfit(t, y, 1)
    return _spread(y - (slope * t + intercept))


def kinematic_summary(pose, trans, fps):
    """21 frozen features from joint-rotation magnitudes and detrended trans. See KIN_NAMES.

    Uses no body model and no axis convention for joints. Holds no path length, speed or
    duration term: forward progression is removed from the trans features by detrending.
    """
    step = max(1, round(fps / KIN_RATE_HZ))
    dt = step / fps
    idx = np.arange(0, pose.shape[0], step)
    amp, asym, rate = [], [], []
    for _, left, right in JOINTS:
        a_l = _spread(np.linalg.norm(_rotvec(pose, left), axis=1))
        a_r = _spread(np.linalg.norm(_rotvec(pose, right), axis=1))
        amp.append((a_l + a_r) / 2)
        asym.append(abs(a_l - a_r) / (a_l + a_r + 1e-9))
        rate.append((_geodesic_rate(_rotvec(pose, left)[idx], dt) + _geodesic_rate(_rotvec(pose, right)[idx], dt)) / 2)
    alternation = (np.linalg.norm(_rotvec(pose, JOINTS[0][1]), axis=1)
                   - np.linalg.norm(_rotvec(pose, JOINTS[0][2]), axis=1))[idx]
    freq, regularity = _stride_frequency(alternation, dt)
    t = np.arange(trans.shape[0]) / fps
    trunk = sum(np.linalg.norm(_rotvec(pose, j), axis=1) for j in SPINE)
    return np.asarray(amp + asym + rate + [freq, regularity, _detrended_spread(trans[:, 1], t),
                                           _detrended_spread(trans[:, 0], t), float(trunk.mean()), _spread(trunk)])


def build_table(cohort_objs):
    """P0 feature table plus S_alt and K, in the same walk order as P0 build_table."""
    t = P.build_table(cohort_objs)
    alt, kin = [], []
    for name in C.COHORTS:
        for _, rec in C.walks_of(cohort_objs[name]):
            pose, trans = C.read_arrays(rec)
            if not (np.isfinite(pose).all() and np.isfinite(trans).all()):
                continue
            fps = C.read_fps(rec["fps"])
            alt.append(speed_alt(trans, fps))
            kin.append(kinematic_summary(pose, trans, fps))
    t["speed_alt"], t["kin"] = np.asarray(alt), np.asarray(kin)
    return t


def check_stops(t, totals=EXPECTED_TOTALS):
    """P0 stop checks plus P1 feature checks. Runs before any model is fitted."""
    P.check_stops(t)
    if not (np.isfinite(t["speed_alt"]).all() and np.isfinite(t["kin"]).all()):
        raise C.SourceError("P1_FEATURE_NON_FINITE")
    if t["kin"].shape != (t["label"].size, len(KIN_NAMES)) or t["speed_alt"].size != t["label"].size:
        raise C.SourceError("P1_FEATURE_SHAPE")
    eligible = np.isin(t["label"], (0, 1, 2))
    if int((t["label"] >= 0).sum()) != totals["walks_all"] or int(eligible.sum()) != totals["walks_eligible"]:
        raise C.SourceError("P1_WALK_TOTALS")
    if len(np.unique(t["subject"][eligible])) != totals["subjects_eligible"]:
        raise C.SourceError("P1_SUBJECT_TOTALS")
    for ci in range(len(C.COHORTS)):
        if not all((t["label"][eligible & (t["cohort"] == ci)] == k).any() for k in (0, 1, 2)):
            raise C.SourceError("P1_COHORT_MISSING_CLASS")


def check_versions(numpy_version, scipy_version, sklearn_version):
    got = {"numpy": numpy_version, "scipy": scipy_version, "sklearn": sklearn_version}
    for name, want in VERSIONS.items():
        if got[name] != want:
            raise C.SourceError(f"P1_VERSION:{name}")


def scale_columns_within_cohort(X, cohort):
    """Median/IQR per column within each cohort. No label argument. A zero IQR only centres."""
    out = np.empty_like(X, dtype=float)
    for ci in np.unique(cohort):
        m = cohort == ci
        med = np.median(X[m], axis=0)
        q75, q25 = np.percentile(X[m], [75, 25], axis=0)
        iqr = q75 - q25
        out[m] = (X[m] - med) / np.where(iqr > 0, iqr, 1.0)
    return out


def arm_features(t):
    s, d, alt, pose, kin = t["speed"], t["duration"], t["speed_alt"], t["pose"], t["kin"]
    sz = P.scale_within_cohort(s, t["cohort"])
    return {
        "S": s[:, None], "S_alt": alt[:, None], "D": d[:, None], "S_D": np.column_stack([s, d]),
        "P": pose, "PS": np.column_stack([pose, s]), "PS_alt": np.column_stack([pose, alt]),
        "K": kin, "KS": np.column_stack([kin, s]),
        "Sz": sz[:, None], "PzSz": np.column_stack([scale_columns_within_cohort(pose, t["cohort"]), sz]),
    }


# ---- fitting ---------------------------------------------------------------------
def train_weights(y, subject, scheme):
    """Class-balanced sample weights. Depends only on labels and subjects, never on the input arm.

    `walk` equals scikit-learn's class_weight="balanced". `person` first gives each subject
    equal total weight, then balances classes with the same formula.
    """
    if scheme == "walk":
        w = np.ones(y.size, dtype=float)
    elif scheme == "person":
        _, inverse, counts = np.unique(subject, return_inverse=True, return_counts=True)
        w = 1.0 / counts[inverse]
    else:
        raise ValueError(scheme)
    present = np.unique(y)
    total = w.sum()
    out = w.copy()
    for k in present:
        out[y == k] *= total / (present.size * w[y == k].sum())
    return out


def make_model(config, seed=SEED):
    _, family, params = config
    if family == "LR":
        return LogisticRegression(**params)
    if family == "RF":
        return RandomForestClassifier(random_state=seed, **params)
    if family == "HGB":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(family)


def fit_predict_proba(config, Xtr, ytr, wtr, Xte, warned, seed=SEED):
    """Train-only scaling, then the model. Returns (n_test, 3) probabilities in class order 0, 1, 2."""
    scaler = StandardScaler().fit(Xtr)
    model = make_model(config, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(Xtr), ytr, sample_weight=wtr)
    warned[0] += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    proba = np.zeros((Xte.shape[0], 3))
    proba[:, model.classes_.astype(int)] = model.predict_proba(scaler.transform(Xte))
    return proba


def walk_macro_f1(y, pred):
    conf = np.zeros((3, 3))
    np.add.at(conf, (y, pred), 1)
    return float(P.macro_f1_conf(conf))


def inner_scores(X, y, site, subject, scheme, configs, warned):
    """(n_configs, n_source_sites) leave-one-source-site-out macro-F1. Source arrays only."""
    sites = np.unique(site)
    out = np.zeros((len(configs), sites.size))
    for c, config in enumerate(configs):
        for s, v in enumerate(sites):
            tr, va = site != v, site == v
            proba = fit_predict_proba(config, X[tr], y[tr], train_weights(y[tr], subject[tr], scheme), X[va], warned)
            out[c, s] = walk_macro_f1(y[va], proba.argmax(axis=1))
    return out


def select_config(mean_scores, allowed):
    """Index of the best allowed config. A tie within 1e-12 keeps the earlier index."""
    best = None
    for i in allowed:
        if best is None or mean_scores[i] > mean_scores[best] + TIE_EPS:
            best = i
    return best


def policy_members(configs):
    names = {"all": list(range(len(configs))),
             "nonlinear": [i for i, c in enumerate(configs) if c[1] != "LR"]}
    for fam in FAMILIES:
        names[fam] = [i for i, c in enumerate(configs) if c[1] == fam]
    return {k: v for k, v in names.items() if v}


def outer_run(X, label, cohort, subject, scheme, configs, warned):
    """All four outer folds for one input arm.

    Selection sees the three source sites only. Every candidate is refit on all sources and
    predicts held-out features, which are never paired with held-out labels in this function.
    """
    eligible = np.isin(label, (0, 1, 2))
    members = policy_members(configs)
    folds = []
    for ci in range(len(C.COHORTS)):
        tr = eligible & (cohort != ci)
        te = eligible & (cohort == ci)
        scores = inner_scores(X[tr], label[tr], cohort[tr], subject[tr], scheme, configs, warned)
        means = scores.mean(axis=1)
        selected = {name: select_config(means, idx) for name, idx in members.items()}
        wtr = train_weights(label[tr], subject[tr], scheme)
        proba = [fit_predict_proba(c, X[tr], label[tr], wtr, X[te], warned) for c in configs]
        folds.append({"inner": scores, "selected": selected, "proba": proba, "test_mask": te})
    return folds


def proba_for(folds, ci, policy=None, config_index=None):
    f = folds[ci]
    return f["proba"][f["selected"][policy] if policy is not None else config_index]


# ---- metrics ---------------------------------------------------------------------
def balanced_accuracy_conf(conf):
    true = conf.sum(axis=-1)
    recall = np.divide(np.diagonal(conf, axis1=-2, axis2=-1), true, out=np.zeros_like(true, dtype=float), where=true > 0)
    return recall.mean(axis=-1)


def ordinal_mae_conf(conf):
    dist = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
    return (conf * dist).sum(axis=(-2, -1)) / conf.sum(axis=(-2, -1))


METRICS = {"macro_f1": P.macro_f1_conf, "balanced_accuracy": balanced_accuracy_conf, "ordinal_mae": ordinal_mae_conf}


class Scorer:
    """Pooled metrics and paired bootstrap draws. One set of subject draws serves every arm."""

    def __init__(self, t, n_boot):
        eligible = np.isin(t["label"], (0, 1, 2))
        self.names = list(C.COHORTS)
        self.n_boot = n_boot
        self.y, self.subject, self.subjects, self.masks = {}, {}, {}, {}
        for ci, name in enumerate(self.names):
            m = eligible & (t["cohort"] == ci)
            self.masks[name] = m
            self.y[name], self.subject[name] = t["label"][m], t["subject"][m]
            self.subjects[name] = np.unique(self.subject[name])
        self.counts = P.draw_counts([len(self.subjects[n]) for n in self.names], n_boot, SEED)
        self.rgb = [self.names.index(n) for n in RGB_COHORTS]

    def conf(self, preds, where=None):
        out = {}
        for ci, name in enumerate(self.names):
            keep = np.ones(self.y[name].size, bool) if where is None else where[name]
            out[name] = P.confusion_by_subject(self.y[name][keep], preds[ci][keep], self.subject[name][keep],
                                               self.subjects[name])
        return out

    def stats(self, conf, metric="macro_f1", person=False):
        fn = METRICS[metric]
        point, draws = np.empty(len(self.names)), np.empty((self.n_boot, len(self.names)))
        for c, name in enumerate(self.names):
            cm = conf[name]
            if person:
                totals = cm.sum(axis=(1, 2), keepdims=True)
                cm = np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)
            point[c] = fn(cm.sum(axis=0))
            draws[:, c] = fn(P.apply_counts(self.counts[c], cm))
        return point, draws

    def block(self, point, draws):
        return {
            "per_cohort": {n: P.summary(point[c], draws[:, c]) for c, n in enumerate(self.names)},
            "M": P.summary(point.mean(), draws.mean(axis=1)),
            "M_rgb_only": P.summary(point[self.rgb].mean(), draws[:, self.rgb].mean(axis=1)),
        }

    def all_metrics(self, conf):
        out = {m: self.block(*self.stats(conf, m)) for m in METRICS}
        out["macro_f1_person_weighted_eval"] = self.block(*self.stats(conf, "macro_f1", person=True))
        return out

    def contrast(self, conf_a, conf_b, person=False):
        pa, da = self.stats(conf_a, person=person)
        pb, db = self.stats(conf_b, person=person)
        return self.block(pa - pb, da - db)

    def floor_stats(self):
        point, draws = np.empty(len(self.names)), np.empty((self.n_boot, len(self.names)))
        for c, name in enumerate(self.names):
            true = np.zeros((len(self.subjects[name]), 3))
            pos = {s: i for i, s in enumerate(self.subjects[name])}
            for yi, si in zip(self.y[name], self.subject[name]):
                true[pos[si], int(yi)] += 1
            point[c] = P.plugin_floor(true.sum(axis=0))
            draws[:, c] = P.plugin_floor(self.counts[c] @ true)
        return point, draws


# ---- verdict ---------------------------------------------------------------------
def decide(delta, ci_lo, ci_hi, per_site, null_rank_p, null_done, guard_delta, guard_lo):
    """Frozen rule. The screen, then the null gate, then the handicapped-control guard."""
    nonneg = int(sum(float(d) >= 0 for d in per_site))
    screen = bool(delta >= SCREEN_MIN and ci_lo > 0 and nonneg >= 2)
    qualifier = None
    if screen:
        if null_done < MIN_NULL_FOR_GO:
            verdict = "P1_SCREEN_PASS_NULL_UNAVAILABLE"
        elif null_rank_p <= NULL_ALPHA:
            verdict = "P1_GO_BEYOND_SPEED"
            if not (guard_delta >= SCREEN_MIN and guard_lo > 0):
                qualifier = "GAIN_DEPENDS_ON_WEAKER_CONTROL"
        else:
            verdict = "P1_SCREEN_PASS_NOT_ABOVE_NULL"
    else:
        verdict = "P1_NO_GO_NOT_BEYOND_SPEED"
        qualifier = "POSE_ARM_WORSE" if ci_hi < 0 else ("SMALL_POSITIVE" if ci_lo > 0 else "INCONCLUSIVE")
    return {"screen_pass": screen, "sites_nonnegative": nonneg, "verdict": verdict, "qualifier": qualifier}


# ---- real-data null ----------------------------------------------------------------
def subject_block_labels(label, cohort, subject, rng):
    """Null labels that keep subject clustering and cohort mix but break the feature link.

    Within a cohort, subjects are permuted, and each subject's eligible walks draw labels
    with replacement from the walks of the subject it was mapped to. Class 3 stays put.
    """
    out = label.copy()
    eligible = np.isin(label, (0, 1, 2))
    for ci in np.unique(cohort):
        m = eligible & (cohort == ci)
        subjects = np.unique(subject[m])
        donors = rng.permutation(subjects)
        pools = {s: label[m & (subject == s)] for s in subjects}
        for s, d in zip(subjects, donors):
            where = m & (subject == s)
            out[where] = rng.choice(pools[d], where.sum(), replace=True)
    return out


def null_distribution(t, feats, folds_s, folds_ps, configs, scorer, observed, n_null=N_NULL, budget_s=NULL_BUDGET_S):
    """Refit the selected S and PS configs on null training labels; score against real held-out labels.

    Selection is not repeated for each null draw, so this is conditional on the real selection.
    Descriptive for the arms. It gates only the go verdict in decide().
    """
    rng = np.random.default_rng([SEED, 13])
    eligible = np.isin(t["label"], (0, 1, 2))
    warned, start, s_all, ps_all = [0], time.monotonic(), [], []
    for _ in range(n_null):
        if time.monotonic() - start > budget_s:
            break
        shuffled = subject_block_labels(t["label"], t["cohort"], t["subject"], rng)
        row_s, row_ps = [], []
        for ci in range(len(C.COHORTS)):
            tr = eligible & (t["cohort"] != ci)
            te = eligible & (t["cohort"] == ci)
            wtr = train_weights(shuffled[tr], t["subject"][tr], "walk")
            for arm, folds, row in (("S", folds_s, row_s), ("PS", folds_ps, row_ps)):
                config = configs[folds[ci]["selected"]["all"]]
                proba = fit_predict_proba(config, feats[arm][tr], shuffled[tr], wtr, feats[arm][te], warned)
                row.append(walk_macro_f1(t["label"][te], proba.argmax(axis=1)))
        s_all.append(np.mean(row_s))
        ps_all.append(np.mean(row_ps))
    done = len(s_all)
    result = {"n_draws_done": done, "n_draws_planned": n_null, "seed": [SEED, 13], "n_convergence_warnings": warned[0]}
    if done == 0:
        return result

    def describe(null, obs):
        return {"null_mean": float(null.mean()), "null_p2_5_p97_5": P.ci95(null), "observed": float(obs),
                "share_null_at_or_above_observed": float((null >= obs).mean()),
                "share_null_at_or_above_margin": float((null >= SCREEN_MIN).mean()),
                "rank_p_one_sided": float((1 + (null >= obs).sum()) / (null.size + 1))}

    s_all, ps_all = np.asarray(s_all), np.asarray(ps_all)
    result["M_S"] = describe(s_all, observed["S"])
    result["M_PS"] = describe(ps_all, observed["PS"])
    result["delta_PS_minus_S"] = describe(ps_all - s_all, observed["PS"] - observed["S"])
    return result


# ---- diagnostics -----------------------------------------------------------------
def _partial_rank_corr(y, score, covars):
    """Rank correlation after removing quantile-bin effects of each covariate from both sides.

    Bins per covariate: walks // 30, kept within 3 to 20. On fabricated speed-only data a smooth
    cubic adjustment left the label's sharp speed steps behind and credited sharper models, so
    finer bins are used. A small leak remains; read arms as differences from the speed arm.
    """
    ry, rs = stats.rankdata(y).astype(float), stats.rankdata(score).astype(float)
    ry -= ry.mean()
    rs -= rs.mean()
    if covars is not None:
        n_bins = int(np.clip(y.size // 30, 3, 20))
        terms = [np.ones(y.size)]
        for c in covars:
            bins = np.minimum(((stats.rankdata(c) - 0.5) / y.size * n_bins).astype(int), n_bins - 1)
            terms += [(bins == k).astype(float) for k in range(1, n_bins)]
        A = np.column_stack(terms)
        ry = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        rs = rs - A @ np.linalg.lstsq(A, rs, rcond=None)[0]
    sy, ss = ry.std(), rs.std()
    if sy < 1e-8:
        return np.nan  # the label has no variation left, so the value is undefined
    if ss < 1e-8:
        return 0.0  # the score is fully explained by the covariates, so it adds nothing beyond them
    return float(((ry - ry.mean()) * (rs - rs.mean())).mean() / (sy * ss))


def speed_conditional_partial(t, scorer, scores_by_arm):
    """Rank correlation of label and expected-label score, given S, S_alt and duration ranks.

    A threshold-free read of information beyond speed. The speed arm is the negative control:
    its value shows how much speed the adjustment leaves behind, so richer arms are read as a
    paired difference from it. Bootstraps subjects within cohorts, using the same draws as
    every other interval, and counts undefined draws.
    """
    keys = ("unconditional", "given_speed_duration")
    points, all_draws = {}, {}
    for arm, scores in scores_by_arm.items():
        point = {k: np.empty(len(scorer.names)) for k in keys}
        draws = {k: np.empty((scorer.n_boot, len(scorer.names))) for k in keys}
        for c, name in enumerate(scorer.names):
            m = scorer.masks[name]
            y, sc = t["label"][m], scores[c]
            cov = (t["speed"][m], t["speed_alt"][m], t["duration"][m])
            point["unconditional"][c] = _partial_rank_corr(y, sc, None)
            point["given_speed_duration"][c] = _partial_rank_corr(y, sc, cov)
            rows = [np.flatnonzero(scorer.subject[name] == s) for s in scorer.subjects[name]]
            for b in range(scorer.n_boot):
                idx = np.concatenate([np.repeat(r, k) for r, k in zip(rows, scorer.counts[c][b])])
                draws["unconditional"][b, c] = _partial_rank_corr(y[idx], sc[idx], None)
                draws["given_speed_duration"][b, c] = _partial_rank_corr(y[idx], sc[idx], tuple(v[idx] for v in cov))
        points[arm], all_draws[arm] = point, draws

    def block(point, d):
        ok = np.isfinite(d).all(axis=1)
        if not ok.any():
            return {"per_cohort": None, "M": None, "n_undefined_draws": int((~ok).sum())}
        return {"per_cohort": {n: P.summary(point[c], d[ok, c]) for c, n in enumerate(scorer.names)},
                "M": P.summary(point.mean(), d[ok].mean(axis=1)), "n_undefined_draws": int((~ok).sum())}

    out = {arm: {k: block(points[arm][k], all_draws[arm][k]) for k in keys} for arm in scores_by_arm}
    k = "given_speed_duration"
    for arm in scores_by_arm:
        if arm != "S":
            out[f"{arm}_minus_S"] = {k: block(points[arm][k] - points["S"][k], all_draws[arm][k] - all_draws["S"][k])}
    return out


def speed_terciles(t, scorer):
    """Label-free speed terciles within each held-out cohort, over all that cohort's walks."""
    where = []
    for ci, name in enumerate(scorer.names):
        all_speed = t["speed"][t["cohort"] == ci]
        q1, q2 = np.percentile(all_speed, [100 / 3, 200 / 3])
        s = t["speed"][scorer.masks[name]]
        where.append({"slow": s <= q1, "mid": (s > q1) & (s <= q2), "fast": s > q2})
    return where


def tercile_contrasts(scorer, terciles, preds_a, preds_b):
    out = {}
    for level in ("slow", "mid", "fast"):
        where = {name: terciles[c][level] for c, name in enumerate(scorer.names)}
        conf_a, conf_b = scorer.conf(preds_a, where), scorer.conf(preds_b, where)
        out[level] = {"a": scorer.block(*scorer.stats(conf_a)), "b": scorer.block(*scorer.stats(conf_b)),
                      "a_minus_b": scorer.contrast(conf_a, conf_b),
                      "walks_per_cohort": {n: int(where[n].sum()) for n in scorer.names}}
    return out


def site_identifiability(t, feats, arms=("S", "S_alt", "D", "P", "K", "PS")):
    """Balanced accuracy of predicting the cohort from features. Severity labels are not used."""
    eligible = np.isin(t["label"], (0, 1, 2))
    cohort, subject = t["cohort"][eligible], t["subject"][eligible]
    out = {}
    warned = [0]
    for arm in arms:
        X = feats[arm][eligible]
        recalls = []
        for tr, va in GroupKFold(n_splits=5).split(X, cohort, subject):
            scaler = StandardScaler().fit(X[tr])
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight="balanced")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(scaler.transform(X[tr]), cohort[tr])
            warned[0] += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
            pred = model.predict(scaler.transform(X[va]))
            recalls.append(np.mean([(pred[cohort[va] == c] == c).mean() for c in np.unique(cohort[va])]))
        out[arm] = float(np.mean(recalls))
    return {"balanced_accuracy_5fold_subject_grouped": out, "chance": 0.25, "n_convergence_warnings": warned[0]}


def selection_stability(folds_by_arm, scorer, configs):
    """How far inner-site scores tracked outer scores. Read after selection; it selects nothing."""
    out = {}
    for arm, folds in folds_by_arm.items():
        rows = {}
        for c, name in enumerate(scorer.names):
            f = folds[c]
            inner = f["inner"].mean(axis=1)
            outer = np.array([walk_macro_f1(scorer.y[name], p.argmax(axis=1)) for p in f["proba"]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", stats.ConstantInputWarning)
                rho = stats.spearmanr(inner, outer).statistic
            rows[name] = {"selected": configs[f["selected"]["all"]][0],
                          "selected_inner_mean": float(inner[f["selected"]["all"]]),
                          "selected_outer": float(outer[f["selected"]["all"]]),
                          "inner_mean_by_config": {cfg[0]: float(v) for cfg, v in zip(configs, inner)},
                          "outer_by_config": {cfg[0]: float(v) for cfg, v in zip(configs, outer)},
                          "spearman_inner_outer": None if not np.isfinite(rho) else float(rho)}
        out[arm] = rows
    return out


def speed_definition_summary(t):
    """Label-free comparison of S and S_alt, per cohort."""
    out = {}
    for ci, name in enumerate(C.COHORTS):
        m = t["cohort"] == ci
        ratio = t["speed"][m] / np.where(t["speed_alt"][m] > 0, t["speed_alt"][m], np.nan)
        keep = np.isfinite(ratio)
        out[name] = {
            "median_S": float(np.median(t["speed"][m])), "median_S_alt": float(np.median(t["speed_alt"][m])),
            "spearman_S_S_alt": float(stats.spearmanr(t["speed"][m], t["speed_alt"][m]).statistic),
            "median_ratio_S_over_S_alt": float(np.median(ratio[keep])),
            "spearman_ratio_duration": float(stats.spearmanr(ratio[keep], t["duration"][m][keep]).statistic),
            "n_S_alt_zero": int((~keep).sum()),
        }
    return out


# ---- P0 parity -------------------------------------------------------------------
def parity_check(t, expected_m, configs=CONFIGS):
    """Refit the P0 model through the P1 path and compare with the five M values in the P0 result.

    Reveals nothing new: every number compared is already in the P0 result. Any mismatch
    raises before a new metric exists.
    """
    feats = arm_features(t)
    eligible = np.isin(t["label"], (0, 1, 2))
    warned, found = [0], {}
    for arm in PARITY_ARMS:
        scores = []
        for ci in range(len(C.COHORTS)):
            tr, te = eligible & (t["cohort"] != ci), eligible & (t["cohort"] == ci)
            wtr = train_weights(t["label"][tr], t["subject"][tr], "walk")
            proba = fit_predict_proba(configs[0], feats[arm][tr], t["label"][tr], wtr, feats[arm][te], warned)
            scores.append(walk_macro_f1(t["label"][te], proba.argmax(axis=1)))
        found[arm] = float(np.mean(scores))
        if abs(found[arm] - expected_m[arm]) > PARITY_TOL:
            raise C.SourceError(f"P0_PARITY:{arm}")
    return found


# ---- full analysis ---------------------------------------------------------------
def run_analysis(t, configs=CONFIGS, n_boot=N_BOOT, n_null=N_NULL, null_budget_s=NULL_BUDGET_S):
    feats = arm_features(t)
    warned = [0]
    folds = {arm: outer_run(feats[arm], t["label"], t["cohort"], t["subject"], "walk", configs, warned) for arm in ARMS}
    person = {arm: outer_run(feats[arm], t["label"], t["cohort"], t["subject"], "person", configs, warned)
              for arm in PERSON_SCHEME_ARMS}
    scorer = Scorer(t, n_boot)
    n_folds = len(scorer.names)
    members = policy_members(configs)

    def preds(run, policy):
        return [proba_for(run, c, policy=policy).argmax(axis=1) for c in range(n_folds)]

    def conf_of(run, policy):
        return scorer.conf(preds(run, policy))

    lr_fixed = {arm: scorer.conf([proba_for(folds[arm], c, config_index=0).argmax(axis=1) for c in range(n_folds)])
                for arm in ARMS}
    strong = {arm: conf_of(folds[arm], "all") for arm in ARMS}
    person_strong = {arm: conf_of(person[arm], "all") for arm in PERSON_SCHEME_ARMS}

    majority = []
    for name in scorer.names:
        train = np.isin(t["label"], (0, 1, 2)) & ~scorer.masks[name] & (t["cohort"] != scorer.names.index(name))
        majority.append(np.full(scorer.y[name].size, int(np.bincount(t["label"][train], minlength=3).argmax())))
    conf_c = scorer.conf(majority)
    f_u = scorer.floor_stats()

    primary = scorer.contrast(strong["PS"], strong["S"])
    guard_ps_lr = scorer.contrast(strong["PS"], lr_fixed["S"])
    per_site = [primary["per_cohort"][n]["point"] for n in scorer.names]

    terciles = speed_terciles(t, scorer)
    scores_by_arm = {arm: [(proba_for(folds[arm], c, policy="all") * np.arange(3)).sum(axis=1) for c in range(n_folds)]
                     for arm in ("S", "PS", "KS")}

    exploratory = {f"{a}_minus_{b}": scorer.contrast(strong[a], strong[b])
                   for a, b in (("S_alt", "S"), ("S_D", "S"), ("D", "S"), ("P", "S"), ("K", "S"), ("KS", "S"),
                                ("PS_alt", "S_alt"), ("KS", "PS"))}
    p_s, d_s = scorer.stats(strong["S"])
    exploratory["S_minus_F_U"] = scorer.block(p_s - f_u[0], d_s - f_u[1])

    per_policy = {}
    for policy in members:
        if policy == "all":
            continue
        confs = {arm: conf_of(folds[arm], policy) for arm in ("S", "PS", "KS")}
        per_policy[policy] = {"S": scorer.block(*scorer.stats(confs["S"])),
                              "PS": scorer.block(*scorer.stats(confs["PS"])),
                              "KS": scorer.block(*scorer.stats(confs["KS"])),
                              "PS_minus_S": scorer.contrast(confs["PS"], confs["S"]),
                              "KS_minus_S": scorer.contrast(confs["KS"], confs["S"])}

    fixed_table = {}
    for arm in ARMS:
        fixed_table[arm] = {}
        for i, cfg in enumerate(configs):
            scores = [walk_macro_f1(scorer.y[n], proba_for(folds[arm], c, config_index=i).argmax(axis=1))
                      for c, n in enumerate(scorer.names)]
            fixed_table[arm][cfg[0]] = {"per_cohort": dict(zip(scorer.names, scores)), "M": float(np.mean(scores))}

    selected = {arm: {n: {policy: configs[folds[arm][c]["selected"][policy]][0] for policy in members}
                      for c, n in enumerate(scorer.names)} for arm in ARMS}
    selected_person = {arm: {n: configs[person[arm][c]["selected"]["all"]][0] for c, n in enumerate(scorer.names)}
                       for arm in PERSON_SCHEME_ARMS}

    subj_avg = {}
    for arm in ("S", "PS"):
        avg = [P.subject_averaged_pred(proba_for(folds[arm], c, policy="all"), scorer.subject[n])
               for c, n in enumerate(scorer.names)]
        subj_avg[arm] = scorer.conf(avg)

    refit = {}
    for arm in ("S", "PS"):
        rows = []
        for seed in REFIT_SEEDS:
            scores = []
            for c, name in enumerate(scorer.names):
                config = configs[folds[arm][c]["selected"]["all"]]
                if config[1] == "RF":
                    tr = np.isin(t["label"], (0, 1, 2)) & (t["cohort"] != c)
                    wtr = train_weights(t["label"][tr], t["subject"][tr], "walk")
                    proba = fit_predict_proba(config, feats[arm][tr], t["label"][tr], wtr, feats[arm][scorer.masks[name]],
                                              warned, seed)
                else:
                    proba = proba_for(folds[arm], c, policy="all")
                scores.append(walk_macro_f1(scorer.y[name], proba.argmax(axis=1)))
            rows.append(float(np.mean(scores)))
        refit[arm] = {"seeds": list(REFIT_SEEDS), "M_by_seed": rows, "range": [min(rows), max(rows)]}

    eligible = np.isin(t["label"], (0, 1, 2))
    result = {
        "status": "P1_COMPLETE",
        "primary": {"estimand": "M(PS_strong) - M(S_strong), walk-level macro-F1, equal-site mean, walk training weights",
                    "contrast": primary},
        "guards": {"PS_strong_minus_S_P0_model": guard_ps_lr,
                   "S_strong_minus_S_P0_model": scorer.contrast(strong["S"], lr_fixed["S"])},
        "candidate_configs": [{"name": c[0], "family": c[1], "params": c[2]} for c in configs],
        "arms_inductive_strong": {a: scorer.all_metrics(strong[a]) for a in INDUCTIVE_ARMS},
        "arms_transductive_strong": {a: scorer.all_metrics(strong[a]) for a in TRANSDUCTIVE_ARMS},
        "arms_P0_model_fixed": {a: scorer.block(*scorer.stats(lr_fixed[a])) for a in ARMS},
        "baselines": {"F_U": scorer.block(*f_u), "C": scorer.block(*scorer.stats(conf_c))},
        "contrasts_exploratory_inductive": exploratory,
        "contrasts_transductive": {
            "PzSz_minus_Sz": scorer.contrast(strong["PzSz"], strong["Sz"]),
            "Sz_minus_S_differs_in_deployment_assumption": scorer.contrast(strong["Sz"], strong["S"]),
        },
        "per_policy_exploratory": per_policy,
        "per_config_macro_f1_exploratory": fixed_table,
        "selected_configs": selected,
        "selection_stability": selection_stability({a: folds[a] for a in ("S", "PS", "KS")}, scorer, configs),
        "person_weighted_training": {
            "selected": selected_person,
            "arms": {a: scorer.all_metrics(person_strong[a]) for a in PERSON_SCHEME_ARMS},
            "PS_minus_S": scorer.contrast(person_strong["PS"], person_strong["S"]),
            "KS_minus_S": scorer.contrast(person_strong["KS"], person_strong["S"]),
            "PS_minus_S_person_weighted_eval": scorer.contrast(person_strong["PS"], person_strong["S"], person=True),
        },
        "walk_training_person_weighted_eval": {
            "PS_minus_S": scorer.contrast(strong["PS"], strong["S"], person=True),
        },
        "speed_conditional": {
            "partial_rank_correlation": speed_conditional_partial(t, scorer, scores_by_arm),
            "terciles_PS_vs_S": tercile_contrasts(scorer, terciles, preds(folds["PS"], "all"), preds(folds["S"], "all")),
            "terciles_KS_vs_S": tercile_contrasts(scorer, terciles, preds(folds["KS"], "all"), preds(folds["S"], "all")),
        },
        "subject_averaged_posteriors_uses_test_grouping": {
            a: scorer.block(*scorer.stats(subj_avg[a])) for a in ("S", "PS")},
        "confusions_by_cohort": {
            label: {n: conf[n].sum(axis=0).astype(int).tolist() for n in scorer.names}
            for label, conf in (("S_P0_model", lr_fixed["S"]), ("S_strong", strong["S"]),
                                ("PS_strong", strong["PS"]), ("KS_strong", strong["KS"]))},
        "site_identifiability": site_identifiability(t, feats),
        "speed_definition": speed_definition_summary(t),
        "within_subject_control_S_alt": P.within_subject_control(dict(t, speed=t["speed_alt"]), n_boot),
        "refit_seed_sensitivity": refit,
        "n_convergence_warnings": warned[0],
        "n_boot": n_boot,
        "seed": SEED,
        "totals": {"walks_all_labels": int((t["label"] >= 0).sum()), "walks_eligible_012": int(eligible.sum()),
                   "subjects_eligible": int(len(np.unique(t["subject"][eligible])))},
    }
    observed = {a: result["arms_inductive_strong"][a]["macro_f1"]["M"]["point"] for a in ("S", "PS")}
    state = {"feats": feats, "folds_s": folds["S"], "folds_ps": folds["PS"], "scorer": scorer, "observed": observed,
             "per_site": per_site, "configs": configs}
    return result, state


def finish_with_null(t, result, state, n_null=N_NULL, null_budget_s=NULL_BUDGET_S):
    """Run the null, then apply the frozen rule. The arms are already written before this runs."""
    null = null_distribution(t, state["feats"], state["folds_s"], state["folds_ps"], state["configs"], state["scorer"],
                             state["observed"], n_null, null_budget_s)
    primary = result["primary"]["contrast"]["M"]
    guard = result["guards"]["PS_strong_minus_S_P0_model"]["M"]
    rank_p = null.get("delta_PS_minus_S", {}).get("rank_p_one_sided", 1.0)
    result["null_subject_block"] = null
    result["decision"] = decide(primary["point"], primary["ci95"][0], primary["ci95"][1], state["per_site"], rank_p,
                                null["n_draws_done"], guard["point"], guard["ci95"][0])
    return result


# ---- one-run ledger ----------------------------------------------------------------
def ledger_states(path):
    if not path.exists():
        return []
    return [line.split("\t")[-1] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ledger_append(path, commit, state):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{commit}\t{state}\n")


def ledger_refusal(states):
    """A new attempt is refused after any scored attempt, or after one failed pre-score attempt."""
    if "SCORED" in states:
        return "SCORES_EXIST"
    if states.count("START") >= 2:
        return "PRE_SCORE_FIX_ALREADY_USED"
    return None
