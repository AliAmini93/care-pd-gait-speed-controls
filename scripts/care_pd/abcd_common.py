"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import p0_core as P
import p1_core as Q
import representation_core as RC
DEV_SITES = ('BMCLab', 'PD-GaM', 'T-SDU-PD')
FINAL_SITES = C.COHORTS
RESERVED_SITE = '3DGait'
ELIGIBLE = (0, 1, 2)
RIDGE_LAMBDA = 0.03
LOG_TAU_BOUNDS = (-3.0, 3.0)
SPEED_FLOOR = 0.001
MARGIN = 0.03
SITE_FLOOR = -0.03
DEV_MIN_POSITIVE = 2
FINAL_MIN_POSITIVE = 2
N_BOOT = 2000
BOOT_SEED = Q.SEED
DEV_EXPECTED = {'BMCLab': {'subjects': 23, 'walks_eligible': 781}, 'PD-GaM': {'subjects': 30, 'walks_eligible': 1666}, 'T-SDU-PD': {'subjects': 14, 'walks_eligible': 381}}

class AbcdError(C.SourceError):
    pass

def one_hot(y):
    Y = np.zeros((y.size, 3))
    Y[np.arange(y.size), y.astype(int)] = 1.0
    return Y

def class_weights(y):
    w = Q.train_weights(y, np.zeros(y.size, dtype=int), 'walk')
    return w / w.mean()

class OffsetResidual:

    def __init__(self, lam=RIDGE_LAMBDA, penalty=None, beta=0.0):
        if penalty not in (None, 'consensus', 'vrex'):
            raise AbcdError('ABCD_PENALTY')
        self.lam, self.penalty, self.beta = (float(lam), penalty, float(beta))
        self.W = None
        self.tau = 1.0
        self.diag = {}

    @staticmethod
    def _forward(theta, Phi, O, d):
        tau = float(np.exp(theta[0]))
        W = theta[1:].reshape(d, 3)
        Z = tau * O + Phi @ W
        Z = Z - Z.max(axis=1, keepdims=True)
        logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
        return (tau, W, logp)

    def fit(self, Phi, O, y, w=None, env=None):
        Phi, O = (np.asarray(Phi, dtype=np.float64), np.asarray(O, dtype=np.float64))
        y = np.asarray(y).astype(int)
        n, d = Phi.shape
        if O.shape != (n, 3) or y.size != n or (not (np.isfinite(Phi).all() and np.isfinite(O).all())):
            raise AbcdError('ABCD_RESIDUAL_INPUT')
        w = class_weights(y) if w is None else np.asarray(w, dtype=np.float64) / np.mean(w)
        Y = one_hot(y)
        groups = None
        if self.penalty is not None:
            if env is None or np.unique(env).size < 2:
                raise AbcdError('ABCD_PENALTY_NEEDS_TWO_ENVS')
            groups = [np.flatnonzero(env == e) for e in np.unique(env)]
            tau0 = OffsetResidual(lam=self.lam).fit_offset_only(O, y, w)
            self.diag['tau0'] = tau0
            Z0 = tau0 * O
            Z0 = Z0 - Z0.max(axis=1, keepdims=True)
            lp0 = Z0 - np.log(np.exp(Z0).sum(axis=1, keepdims=True))
            r0 = np.array([-(w[g] * (lp0[g] * Y[g]).sum(axis=1)).sum() / w[g].sum() for g in groups])
            self.diag['env_offset_risk'] = r0.tolist()

        def objective(theta):
            tau, W, logp = self._forward(theta, Phi, O, d)
            ce = -(logp * Y).sum(axis=1)
            P_ = np.exp(logp)
            G = (P_ - Y) * (w / n)[:, None]
            loss = float((w * ce).sum() / n)
            if groups is not None:
                r = np.array([(w[g] * ce[g]).sum() / w[g].sum() for g in groups])
                k = len(groups)
                if self.penalty == 'consensus':
                    gain = r0 - r
                    dev = gain - gain.mean()
                    loss += self.beta * float((dev ** 2).mean())
                    coef = -2.0 * self.beta * dev / k
                else:
                    dev = r - r.mean()
                    loss += self.beta * float((dev ** 2).mean())
                    coef = 2.0 * self.beta * dev / k
                for e, g in enumerate(groups):
                    G[g] += coef[e] * (P_[g] - Y[g]) * (w[g] / w[g].sum())[:, None]
            gW = Phi.T @ G + self.lam * W
            loss += 0.5 * self.lam * float((W ** 2).sum())
            gt = tau * float((G * O).sum())
            return (loss, np.concatenate([[gt], gW.ravel()]))
        theta0 = np.zeros(1 + 3 * d)
        bounds = [LOG_TAU_BOUNDS] + [(None, None)] * (3 * d)
        res = minimize(objective, theta0, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'gtol': 1e-07})
        if not np.isfinite(res.fun) or not np.isfinite(res.x).all():
            raise AbcdError('ABCD_RESIDUAL_NONFINITE')
        self.tau = float(np.exp(res.x[0]))
        self.W = res.x[1:].reshape(d, 3)
        self.diag.update({'converged': bool(res.success), 'n_iter': int(res.nit), 'tau': self.tau, 'w_frobenius': float(np.sqrt((self.W ** 2).sum())), 'loss': float(res.fun)})
        return self

    def fit_offset_only(self, O, y, w):
        Y = one_hot(y)

        def f(lt):
            tau = float(np.exp(lt[0]))
            Z = tau * O
            Z = Z - Z.max(axis=1, keepdims=True)
            logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
            G = (np.exp(logp) - Y) * (w / y.size)[:, None]
            return (float(-(w * (logp * Y).sum(axis=1)).sum() / y.size), np.array([tau * float((G * O).sum())]))
        res = minimize(f, np.zeros(1), jac=True, method='L-BFGS-B', bounds=[LOG_TAU_BOUNDS])
        return float(np.exp(res.x[0]))

    def logits(self, Phi, O):
        if self.W is None:
            raise AbcdError('ABCD_RESIDUAL_NOT_FITTED')
        return self.tau * np.asarray(O, dtype=np.float64) + np.asarray(Phi, dtype=np.float64) @ self.W

class SpeedConditionalScaler:

    def fit(self, F, s):
        F = np.asarray(F, dtype=np.float64)
        ls = np.log(np.maximum(np.asarray(s, dtype=np.float64), SPEED_FLOOR))
        self.coef = np.zeros((2, F.shape[1]))
        self.sd = np.ones(F.shape[1])
        for j in range(F.shape[1]):
            ok = np.isfinite(F[:, j])
            if ok.sum() < 3:
                self.coef[:, j] = 0.0
                self.sd[j] = 1.0
                continue
            A = np.column_stack([np.ones(ok.sum()), ls[ok]])
            c, *_ = np.linalg.lstsq(A, F[ok, j], rcond=None)
            r = F[ok, j] - A @ c
            self.coef[:, j] = c
            sd = float(r.std())
            self.sd[j] = sd if sd > 1e-09 else 1.0
        return self

    def transform(self, F, s):
        F = np.asarray(F, dtype=np.float64)
        ls = np.log(np.maximum(np.asarray(s, dtype=np.float64), SPEED_FLOOR))
        A = np.column_stack([np.ones(ls.size), ls])
        out = (F - A @ self.coef) / self.sd
        return np.where(np.isfinite(out), out, 0.0)

class SiteScorer:

    def __init__(self, y_by_site, subj_by_site, sites, n_boot=N_BOOT, seed=BOOT_SEED):
        self.sites = list(sites)
        self.n_boot = n_boot
        self.y = {s: np.asarray(y_by_site[s]).astype(int) for s in self.sites}
        self.subj = {s: np.asarray(subj_by_site[s]) for s in self.sites}
        self.subjects = {s: np.unique(self.subj[s]) for s in self.sites}
        self.counts = P.draw_counts([len(self.subjects[s]) for s in self.sites], n_boot, seed)

    def conf(self, pred_by_site):
        return {s: P.confusion_by_subject(self.y[s], np.asarray(pred_by_site[s]).astype(int), self.subj[s], self.subjects[s]) for s in self.sites}

    def stats(self, conf, metric='macro_f1', person=False):
        fn = Q.METRICS[metric]
        point, draws = (np.empty(len(self.sites)), np.empty((self.n_boot, len(self.sites))))
        for c, s in enumerate(self.sites):
            cm = conf[s]
            if person:
                tot = cm.sum(axis=(1, 2), keepdims=True)
                cm = np.divide(cm, tot, out=np.zeros_like(cm), where=tot > 0)
            point[c] = fn(cm.sum(axis=0))
            draws[:, c] = fn(P.apply_counts(self.counts[c], cm))
        return (point, draws)

    def block(self, point, draws):
        return {'per_site': {s: P.summary(point[c], draws[:, c]) for c, s in enumerate(self.sites)}, 'M': P.summary(point.mean(), draws.mean(axis=1))}

    def metrics(self, conf):
        out = {m: self.block(*self.stats(conf, m)) for m in Q.METRICS}
        out['macro_f1_person_weighted_eval'] = self.block(*self.stats(conf, 'macro_f1', person=True))
        return out

    def contrast(self, conf_a, conf_b, person=False):
        pa, da = self.stats(conf_a, person=person)
        pb, db = self.stats(conf_b, person=person)
        return self.block(pa - pb, da - db)

def macro_f1(y, pred):
    return Q.walk_macro_f1(np.asarray(y).astype(int), np.asarray(pred).astype(int))

def dev_screen(contrast):
    sites = [float(v['point']) for v in contrast['per_site'].values()]
    m = float(contrast['M']['point'])
    cond = {'M_delta': m, 'margin_ge_0_03': bool(m >= MARGIN), 'sites_positive': int(sum((d > 0 for d in sites))), 'sites_positive_ge_2_of_3': bool(sum((d > 0 for d in sites)) >= DEV_MIN_POSITIVE), 'worst_site_delta': float(min(sites)), 'no_site_below_minus_0_03': bool(min(sites) >= SITE_FLOOR)}
    cond['dev_screen_pass'] = bool(cond['margin_ge_0_03'] and cond['sites_positive_ge_2_of_3'] and cond['no_site_below_minus_0_03'])
    return cond

def synthetic_screen(delta_by_world):
    w1, w2, w3 = (float(delta_by_world[k]) for k in ('W1', 'W2', 'W3'))
    cond = {'W1_delta': w1, 'W2_delta': w2, 'W3_delta': w3, 'W1_or_W2_gain_ge_0_03': bool(max(w1, w2) >= MARGIN), 'W3_no_degradation_gt_0_03': bool(w3 >= -MARGIN)}
    cond['synthetic_screen_pass'] = bool(cond['W1_or_W2_gain_ge_0_03'] and cond['W3_no_degradation_gt_0_03'])
    return cond

def stage_status(validity_pass, synth, dev, blocked_reason=None):
    if blocked_reason:
        return {'status': 'DEV_BLOCKED', 'reason': blocked_reason}
    if validity_pass and synth['synthetic_screen_pass'] and dev['dev_screen_pass']:
        return {'status': 'DEV_PROMISING', 'reason': 'all synthetic validity checks, synthetic screen and the three-cohort development screen passed'}
    failed = [k for k in ('synthetic_screen_pass',) if not synth[k]] + [k for k in ('margin_ge_0_03', 'sites_positive_ge_2_of_3', 'no_site_below_minus_0_03') if not dev[k]]
    if not validity_pass:
        failed = ['synthetic_validity'] + failed
    if dev['dev_screen_pass'] and (not (validity_pass and synth['synthetic_screen_pass'])):
        return {'status': 'DEV_INCONCLUSIVE', 'failed': failed, 'reason': 'development screen passed but synthetic validity/screen did not, so the development gain cannot be attributed to the hypothesised mechanism'}
    return {'status': 'DEV_NO_GAIN', 'failed': failed, 'reason': 'valid and tested; screen not passed'}

def final_decision(contrast):
    sites = [float(v['point']) for v in contrast['per_site'].values()]
    m = contrast['M']
    cond = {'M_delta': float(m['point']), 'margin_ge_0_03': bool(m['point'] >= MARGIN), 'ci95_lower_gt_0': bool(m['ci95'][0] > 0), 'sites_positive': int(sum((d > 0 for d in sites))), 'sites_positive_ge_2_of_4': bool(sum((d > 0 for d in sites)) >= FINAL_MIN_POSITIVE), 'worst_site_delta': float(min(sites)), 'no_site_below_minus_0_03': bool(min(sites) >= SITE_FLOOR)}
    if not cond['margin_ge_0_03']:
        status = 'NO_DETECTED_GAIN'
    elif cond['ci95_lower_gt_0'] and cond['sites_positive_ge_2_of_4'] and cond['no_site_below_minus_0_03']:
        status = 'EMPIRICAL_GAIN_EXPLORATORY'
    else:
        status = 'INCONCLUSIVE'
    return {'status': status, 'conditions': cond}

def local_matrices(store, i):
    return RC.sixd_to_matrix(store.locals[i].astype(np.float64))

def chordal_mean(R):
    M = R.mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.ones((M.shape[0], 3))
    D[:, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ (D[:, :, None] * Vt)

def geodesic_angle(Ra, Rb):
    tr = np.einsum('...ji,...ji->...', Ra, Rb)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))

def build_store(poses, fpss):
    roots, locs = ([], [])
    for pose, fps in zip(poses, fpss):
        r, l6 = RC.walk_features(pose, float(fps))
        roots.append(r)
        locs.append(l6)
    return RC.WalkStore(roots, locs)

def synthetic_bundle(data):
    speed, duration = ([], [])
    for tr, fps in zip(data['trans'], data['fps']):
        s, d = P.speed_and_duration(tr, float(fps))
        speed.append(s)
        duration.append(d)
    return {'label': data['label'].astype(int), 'site': data['env'].astype(int), 'subject': data['person'].astype(int), 'speed': np.asarray(speed), 'duration': np.asarray(duration), 'store': build_store(data['pose'], data['fps']), 'fps': np.asarray(data['fps'], dtype=float)}

def build_bundle(cohort_objs, sites):
    label, site, subject, speed, duration, poses, fpss = ([], [], [], [], [], [], [])
    nxt = 0
    for name in sites:
        index = {}
        ci = C.COHORTS.index(name)
        for key, rec in C.walks_of(cohort_objs[name]):
            lab = C.read_label(rec['UPDRS_GAIT'])
            pose, trans = C.read_arrays(rec)
            if not (np.isfinite(pose).all() and np.isfinite(trans).all()):
                raise AbcdError('ABCD_NONFINITE_INPUT')
            if lab is None or lab not in ELIGIBLE:
                continue
            if key not in index:
                index[key] = nxt
                nxt += 1
            fps = C.read_fps(rec['fps'])
            s, d = P.speed_and_duration(trans, fps)
            label.append(lab)
            site.append(ci)
            subject.append(index[key])
            speed.append(s)
            duration.append(d)
            poses.append(pose)
            fpss.append(fps)
    out = {'label': np.asarray(label), 'site': np.asarray(site), 'subject': np.asarray(subject), 'speed': np.asarray(speed), 'duration': np.asarray(duration), 'fps': np.asarray(fpss)}
    out['store'] = build_store(poses, fpss)
    return out

def check_dev_bundle(b):
    for name, want in DEV_EXPECTED.items():
        m = b['site'] == C.COHORTS.index(name)
        if np.unique(b['subject'][m]).size != want['subjects'] or int(m.sum()) != want['walks_eligible']:
            raise AbcdError('ABCD_DEV_ELIGIBILITY:' + name)
        if not all(((b['label'][m] == k).any() for k in ELIGIBLE)):
            raise AbcdError('ABCD_DEV_CLASS_MISSING:' + name)
    if np.isin(b['site'], [C.COHORTS.index(RESERVED_SITE)]).any():
        raise AbcdError('ABCD_RESERVED_SITE_IN_DEV')

def take(b, idx):
    out = {k: v[idx] for k, v in b.items() if k != 'store'}
    out['store'] = RC.WalkStore([b['store'].roots[i] for i in idx], [b['store'].locals[i] for i in idx])
    return out
