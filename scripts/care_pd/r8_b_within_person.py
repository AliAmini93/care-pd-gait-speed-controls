"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import json
import os
import platform
import sys
from pathlib import Path
import numpy as np
import scipy
from scipy import stats
import sklearn
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import p0_core as P
import p1_core as Q
import r8_core as R
CARD_ID = 'b_within_person'
STATUS = 'R8_B_WITHIN_PERSON_COMPLETE'
SEED = 20261104
N_BOOT = 1000
MIN_WALKS_W = 4
N_FEATURES = 24
BONFERRONI_P = 0.05 / N_FEATURES
MIN_ABS_MEAN = 0.1
SITE_P = 0.05
SPEED_LIKE_REFERENCES = ('S_alt', 'duration')
LEAK_MARGIN = 0.05
SPEED_PROXY_ABS_RHO = 0.8
LIMITS = ['Descriptive within-person associations on previously inspected cohorts; exploratory; not causal, not longitudinal.', 'PD-GaM segments of one recording share a label and a bout; within-person n overstates independent events; persons are the independent units.', 'BMCLab within-person grade variation coincides with the ON/OFF session; it is a state contrast, not a treatment effect.', '3DGait two-walk differences are not speed-adjusted.', 'No model, threshold or feature was chosen on these results; the criterion was frozen in the card.', 'No novelty: partial rank correlation and sign tests are standard.']

def feature_matrix(t, seed=SEED):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(t['label'].size)
    X = np.column_stack([t['kin'], t['duration'], t['speed_alt'], noise])
    names = list(Q.KIN_NAMES) + ['duration', 'S_alt', 'noise_control']
    if X.shape[1] != N_FEATURES or len(names) != N_FEATURES:
        raise C.SourceError('R8_B_FEATURE_COUNT')
    return (X, names)

def rank_residual(y, x):
    ry, rx = (stats.rankdata(y).astype(float), stats.rankdata(x).astype(float))
    A = np.column_stack([np.ones(ry.size), rx])
    return ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]

def partial_spearman(grade, speed, feat):
    grade, speed, feat = (np.asarray(v, dtype=float) for v in (grade, speed, feat))
    n = grade.size
    if n < MIN_WALKS_W or np.unique(grade).size < 2 or np.unique(speed).size < 2 or (np.unique(feat).size < 2):
        return None
    eg, ef = (rank_residual(grade, speed), rank_residual(feat, speed))
    if eg.std() < 1e-10 or ef.std() < 1e-10:
        return None
    r = float(np.corrcoef(eg, ef)[0, 1])
    return r if np.isfinite(r) else None

def plain_spearman(a, b):
    a, b = (np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    r = stats.spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else None

def sign_test(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    pos, neg = (int((v > 0).sum()), int((v < 0).sum()))
    p = float(stats.binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else None
    return {'n_defined': int(v.size), 'n_positive': pos, 'n_negative': neg, 'n_zero': int((v == 0).sum()), 'sign_test_two_sided_p': p, 'mean': float(v.mean()) if v.size else None, 'median': float(np.median(v)) if v.size else None}

def boot_mean(groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
    if not groups:
        return np.full(n_boot, np.nan)
    out = np.empty(n_boot)
    for b in range(n_boot):
        out[b] = np.concatenate([g[rng.integers(0, g.size, g.size)] for g in groups]).mean()
    return out

def unit_w(t, X, names):
    eligible = np.isin(t['label'], (0, 1, 2))
    per_site = {}
    for ci, site in enumerate(C.COHORTS):
        rows = {name: [] for name in names}
        rows_unadj = {name: [] for name in names}
        rows_speedlike = {name: [] for name in names}
        undefined = {name: 0 for name in names}
        n_persons_eligible = 0
        n_persons_used = 0
        speed_rho = []
        for s in np.unique(t['subject'][eligible & (t['cohort'] == ci)]):
            m = eligible & (t['subject'] == s)
            g, sp = (t['label'][m], t['speed'][m])
            n_persons_eligible += 1
            if m.sum() < MIN_WALKS_W or np.unique(g).size < 2 or np.unique(sp).size < 2:
                continue
            n_persons_used += 1
            speed_rho.append(plain_spearman(g, sp))
            for j, name in enumerate(names):
                r = partial_spearman(g, sp, X[m, j])
                if r is None:
                    undefined[name] += 1
                else:
                    rows[name].append(r)
                u = plain_spearman(g, X[m, j])
                if u is not None:
                    rows_unadj[name].append(u)
                v = plain_spearman(sp, X[m, j])
                if v is not None:
                    rows_speedlike[name].append(v)
        per_site[site] = {'n_persons_eligible': n_persons_eligible, 'n_persons_used': n_persons_used, 'speed_grade_spearman': sign_test(speed_rho), 'rows': rows, 'rows_unadj': rows_unadj, 'rows_speedlike': rows_speedlike, 'undefined': undefined}
    return per_site

def summarise_w(per_site, names, n_boot=N_BOOT, seed=SEED):
    out_site, pooled = ({}, {})
    for site, rec in per_site.items():
        feats = {}
        for name in names:
            st = sign_test(rec['rows'][name])
            draws = boot_mean([rec['rows'][name]], n_boot, seed)
            st['mean_ci95'] = P.ci95(draws) if np.isfinite(draws).all() and st['n_defined'] >= 2 else None
            st['n_undefined'] = rec['undefined'][name]
            st['unadjusted'] = sign_test(rec['rows_unadj'][name])
            st['speed_likeness_within_person'] = sign_test(rec['rows_speedlike'][name])
            feats[name] = st
        out_site[site] = {'n_persons_eligible': rec['n_persons_eligible'], 'n_persons_used': rec['n_persons_used'], 'speed_grade_spearman': rec['speed_grade_spearman'], 'features': feats}
    for name in names:
        groups = [per_site[site]['rows'][name] for site in per_site]
        allv = [v for g in groups for v in g]
        st = sign_test(allv)
        draws = boot_mean(groups, n_boot, seed)
        st['mean_ci95'] = P.ci95(draws) if np.isfinite(draws).all() and st['n_defined'] >= 2 else None
        st['by_site_n'] = {site: len(per_site[site]['rows'][name]) for site in per_site}
        st['speed_likeness_within_person'] = sign_test([v for site in per_site for v in per_site[site]['rows_speedlike'][name]])
        pooled[name] = st
    return (out_site, pooled)

def apply_criterion(out_site, pooled, names):
    verdict = {}
    ref = max((abs(pooled[r]['mean']) if pooled[r]['mean'] is not None else 0.0 for r in SPEED_LIKE_REFERENCES))
    for name in names:
        p = pooled[name]
        ci = p['mean_ci95']
        c1 = bool(p['sign_test_two_sided_p'] is not None and p['sign_test_two_sided_p'] < BONFERRONI_P and (p['mean'] is not None) and (abs(p['mean']) >= MIN_ABS_MEAN) and (ci is not None) and (ci[0] > 0 or ci[1] < 0))
        c4 = bool(p['mean'] is not None and (name in SPEED_LIKE_REFERENCES or abs(p['mean']) > ref + LEAK_MARGIN))
        sl = p['speed_likeness_within_person']['mean']
        speed_proxy = bool(sl is not None and abs(sl) > SPEED_PROXY_ABS_RHO)
        c2 = True
        for site in ('PD-GaM', 'T-SDU-PD'):
            f = out_site[site]['features'][name]
            ok = f['mean'] is not None and p['mean'] is not None and (np.sign(f['mean']) == np.sign(p['mean'])) and (f['sign_test_two_sided_p'] is not None) and (f['sign_test_two_sided_p'] < SITE_P)
            c2 = c2 and bool(ok)
        verdict[name] = {'pooled_criterion': c1, 'site_recurrence': c2, 'beats_speed_like_references': c4, 'speed_proxy_suspect': speed_proxy, 'survives': bool(c1 and c2 and c4 and (name not in SPEED_LIKE_REFERENCES) and (name != 'noise_control'))}
    noise_ok = not verdict['noise_control']['pooled_criterion']
    survivors = [n for n in names if verdict[n]['survives']]
    clean_survivors = [n for n in survivors if not verdict[n]['speed_proxy_suspect']]
    if not noise_ok:
        reading = 'NEGATIVE_CONTROL_FAILED'
    elif clean_survivors:
        reading = 'WITHIN_PERSON_CANDIDATE'
    elif survivors:
        reading = 'SPEED_PROXY_SUSPECT_ONLY'
    else:
        reading = 'NO_WITHIN_PERSON_SIGNAL_BEYOND_SPEED'
    return {'per_feature': verdict, 'survivors': survivors, 'clean_survivors': clean_survivors, 'negative_control_clean': noise_ok, 'reading': reading, 'speed_like_reference_abs_mean': ref, 'thresholds': {'bonferroni_p': BONFERRONI_P, 'min_abs_mean': MIN_ABS_MEAN, 'site_p': SITE_P, 'leak_margin_over_speed_like_references': LEAK_MARGIN, 'speed_proxy_abs_rho': SPEED_PROXY_ABS_RHO}}

def unit_p(t, X, names, med_site, n_boot=N_BOOT, seed=SEED):
    ci = C.COHORTS.index('BMCLab')
    eligible = np.isin(t['label'], (0, 1, 2))
    site_rows = np.flatnonzero(t['cohort'] == ci)
    if len(med_site) != site_rows.size:
        raise C.SourceError('R8_B_MEDICATION_ROW_ALIGNMENT')
    keep = eligible[site_rows]
    rows = site_rows[keep]
    med = [med_site[i] for i in range(site_rows.size) if keep[i]]
    cells = R.build_cells(t['subject'][rows], med, t['label'][rows])
    persons = cells['persons']
    two = np.flatnonzero(cells['cells_per_person'] == 2)
    if two.size != 19:
        raise C.SourceError('R8_B_TWO_STATE_PERSONS')
    d_grade, d_speed, d_feat = ([], [], [])
    for i in two:
        on = np.flatnonzero((cells['person_idx'] == i) & (cells['state'] == 'ON'))
        off = np.flatnonzero((cells['person_idx'] == i) & (cells['state'] == 'OFF'))
        if on.size != 1 or off.size != 1:
            raise C.SourceError('R8_B_CELL_PAIR')
        m_on, m_off = (cells['cell'] == on[0], cells['cell'] == off[0])
        d_grade.append(int(cells['grade'][off[0]] - cells['grade'][on[0]]))
        d_speed.append(float(t['speed'][rows][m_off].mean() - t['speed'][rows][m_on].mean()))
        d_feat.append(X[rows][m_off].mean(axis=0) - X[rows][m_on].mean(axis=0))
    d_grade, d_speed, d_feat = (np.asarray(d_grade), np.asarray(d_speed), np.asarray(d_feat))
    disc = d_grade != 0
    concordant_sign = int(((d_grade > 0) & (d_speed < 0)).sum() + ((d_grade < 0) & (d_speed > 0)).sum())
    rng = np.random.default_rng(seed)
    n = d_grade.size

    def boot(fn):
        vals = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            v = fn(idx)
            vals[b] = np.nan if v is None else v
        return vals
    out = {'n_two_state_people': int(n), 'n_discordant': int(disc.sum()), 'delta_grade_counts_off_minus_on': {str(k): int((d_grade == k).sum()) for k in range(-2, 3)}, 'discordant_sign_concordance_grade_vs_minus_speed': {'n_concordant': concordant_sign, 'n_discordant_people': int(disc.sum()), 'binomial_p_two_sided': float(stats.binomtest(concordant_sign, int(disc.sum()), 0.5).pvalue) if disc.sum() else None}, 'delta_speed_sign_among_discordant': {'n_speed_lower_in_worse_state': concordant_sign}, 'spearman_delta_grade_delta_speed': R.summary(plain_spearman(d_grade, d_speed), boot(lambda idx: plain_spearman(d_grade[idx], d_speed[idx]))), 'features': {}}
    for j, name in enumerate(names):
        f = d_feat[:, j]
        out['features'][name] = {'spearman_delta_grade_delta_feature': R.summary(plain_spearman(d_grade, f), boot(lambda idx: plain_spearman(d_grade[idx], f[idx]))), 'partial_given_delta_speed': R.summary(partial_spearman(d_grade, d_speed, f), boot(lambda idx: partial_spearman(d_grade[idx], d_speed[idx], f[idx])))}
    return out

def unit_t(t, X, names):
    ci = C.COHORTS.index('3DGait')
    eligible = np.isin(t['label'], (0, 1, 2))
    d_speed, d_feat, n_two, n_two_mixed, n_other = ([], [], 0, 0, 0)
    for s in np.unique(t['subject'][eligible & (t['cohort'] == ci)]):
        m = np.flatnonzero(eligible & (t['subject'] == s))
        if m.size != 2:
            n_other += 1
            continue
        n_two += 1
        g = t['label'][m]
        if g[0] == g[1]:
            continue
        n_two_mixed += 1
        hi, lo = (m[0], m[1]) if g[0] > g[1] else (m[1], m[0])
        d_speed.append(float(t['speed'][hi] - t['speed'][lo]))
        d_feat.append(X[hi] - X[lo])
    d_feat = np.asarray(d_feat) if d_feat else np.zeros((0, len(names)))
    return {'n_persons_with_two_walks': n_two, 'n_two_walk_persons_with_two_grades': n_two_mixed, 'n_persons_other_walk_count': n_other, 'delta_speed_higher_minus_lower_grade': sign_test(d_speed), 'features_not_speed_adjusted': {name: sign_test(d_feat[:, j].tolist()) for j, name in enumerate(names)}}


def collect_medication(obj):
    med = []
    for _, rec in C.walks_of(obj):
        pose, trans = C.read_arrays(rec)
        if not (np.isfinite(pose).all() and np.isfinite(trans).all()):
            continue
        med.append(R.normalized_medication(rec.get('medication')))
    return med

def analyze(t, med_site, n_boot=N_BOOT, seed=SEED):
    X, names = feature_matrix(t, seed)
    if not np.isfinite(X).all():
        raise C.SourceError('R8_B_FEATURE_NON_FINITE')
    per_site = unit_w(t, X, names)
    if per_site['BMCLab']['n_persons_used'] != 8 or per_site['PD-GaM']['n_persons_used'] + per_site['T-SDU-PD']['n_persons_used'] < 30:
        raise C.SourceError('R8_B_UNIT_W_PERSON_COUNTS')
    out_site, pooled = summarise_w(per_site, names, n_boot, seed)
    criterion = apply_criterion(out_site, pooled, names)
    return {'feature_names': names, 'unit_W_within_person_partial': {'per_site': out_site, 'pooled': pooled}, 'criterion': criterion, 'unit_P_bmclab_paired_cells': unit_p(t, X, names, med_site, n_boot, seed), 'unit_T_3dgait_two_walk_descriptive': unit_t(t, X, names), 'settings': {'min_walks_unit_W': MIN_WALKS_W, 'n_boot': n_boot, 'seed': seed, 'n_features': N_FEATURES}}
