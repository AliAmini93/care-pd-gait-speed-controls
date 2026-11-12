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
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import p0_core as P
import p1_core as Q
SEEDS = (20261103, 20261104, 20261105)
N_BOOT = 500
TOL = 1.1e-06
MAX_SOURCE_GRID = 65
ARMS = ('S', 'PS')
POLICIES = ('all', 'LR', 'HGB', 'RF')

def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return round(v, 6) if np.isfinite(v) else None
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    raise TypeError('RESULT_NOT_AGGREGATE_JSON')

def selected_conf(scorer, runs, policy='all'):
    return scorer.conf([Q.proba_for(runs, ci, policy=policy).argmax(axis=1) for ci in range(len(C.COHORTS))])

def point_metrics(scorer, conf):
    return scorer.all_metrics(conf)

def check_baseline_parity(t, results, p1, scorer):
    expected = p1['arms_inductive_strong']
    for arm in ARMS:
        found = selected_conf(scorer, results[arm])
        metrics = point_metrics(scorer, found)['macro_f1']
        for ci, site in enumerate(C.COHORTS):
            config = Q.CONFIGS[results[arm][ci]['selected']['all']][0]
            want_config = p1['selected_configs'][arm][site]['all']
            if config != want_config:
                raise C.SourceError('R1C_P1_CONFIG_PARITY:' + arm + ':' + site)
            point = metrics['per_cohort'][site]['point']
            want = expected[arm]['macro_f1']['per_cohort'][site]['point']
            if abs(point - want) > TOL:
                raise C.SourceError('R1C_P1_SITE_PARITY:' + arm + ':' + site)
        point = metrics['M']['point']
        want = expected[arm]['macro_f1']['M']['point']
        if abs(point - want) > TOL:
            raise C.SourceError('R1C_P1_MEAN_PARITY:' + arm)
    return True

def ordinal_q(proba):
    if proba.ndim != 2 or proba.shape[1] != 3:
        raise C.SourceError('R1C_BAD_PROB_SHAPE')
    return proba[:, 1] + 2.0 * proba[:, 2]

def candidate_thresholds(q, max_grid=None):
    uq = np.unique(np.asarray(q, dtype=float))
    if max_grid is not None and uq.size > max_grid:
        uq = np.unique(np.quantile(uq, np.linspace(0, 1, max_grid)))
    mids = (uq[:-1] + uq[1:]) / 2
    return np.concatenate(([-np.inf], mids, [np.inf]))

def best_thresholds(y_by_site, q_by_site, max_grid=None):
    if len(y_by_site) != len(q_by_site) or not y_by_site:
        raise C.SourceError('R1C_THRESHOLD_INPUT')
    q_all = np.concatenate(q_by_site)
    cuts = candidate_thresholds(q_all, max_grid)
    k = len(cuts)
    cumul = np.zeros((len(y_by_site), k, 3), dtype=np.int64)
    totals = np.zeros((len(y_by_site), 3), dtype=np.int64)
    for site, (y, q) in enumerate(zip(y_by_site, q_by_site)):
        y = np.asarray(y, dtype=int)
        q = np.asarray(q, dtype=float)
        if len(y) != len(q) or not np.isfinite(q).all():
            raise C.SourceError('R1C_THRESHOLD_NONFINITE')
        for cls in range(3):
            vals = np.sort(q[y == cls])
            totals[site, cls] = len(vals)
            cumul[site, :, cls] = np.searchsorted(vals, cuts, side='left')
    best = (-np.inf, 0, 0)
    for i in range(k):
        mid = cumul[:, i:, :] - cumul[:, i:i + 1, :]
        right = totals[:, None, :] - cumul[:, i:, :]
        left = np.broadcast_to(cumul[:, i:i + 1, :], mid.shape)
        tp = np.stack((left[..., 0], mid[..., 1], right[..., 2]), axis=-1)
        predicted = np.stack((left.sum(axis=-1), mid.sum(axis=-1), right.sum(axis=-1)), axis=-1)
        denom = totals[:, None, :] + predicted
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
        scores = f1.mean(axis=-1).mean(axis=0)
        j_off = int(np.argmax(scores))
        val = float(scores[j_off])
        if val > best[0] + 1e-12:
            best = (val, i, i + j_off)
    return (float(cuts[best[1]]), float(cuts[best[2]]), float(best[0]))

def threshold_pred(q, t1, t2):
    if t1 > t2:
        raise C.SourceError('R1C_THRESHOLDS_OUT_OF_ORDER')
    return np.searchsorted([t1, t2], np.asarray(q), side='right').astype(int)

def rank_metrics(y, q):
    out = {}
    for threshold in (1, 2):
        binary = np.asarray(y) >= threshold
        out['auc_y_ge_' + str(threshold)] = float(roc_auc_score(binary, q)) if np.unique(binary).size == 2 else None
    if np.unique(y).size >= 2 and np.unique(q).size >= 2:
        rho = stats.spearmanr(y, q).statistic
        out['spearman_grade_q'] = float(rho) if np.isfinite(rho) else None
    else:
        out['spearman_grade_q'] = None
    return out

def source_cutpoints(t, feat, baseline_folds, outer_ci, configs):
    valid = np.isin(t['label'], (0, 1, 2))
    source = t['cohort'] != outer_ci
    choose = baseline_folds[outer_ci]['selected']['all']
    ys, qs = ([], [])
    warnings = [0]
    for inner_ci in range(len(C.COHORTS)):
        if inner_ci == outer_ci:
            continue
        tr = valid & source & (t['cohort'] != inner_ci)
        va = valid & (t['cohort'] == inner_ci)
        proba = Q.fit_predict_proba(configs[choose], feat[tr], t['label'][tr], Q.train_weights(t['label'][tr], t['subject'][tr], 'walk'), feat[va], warnings)
        ys.append(t['label'][va])
        qs.append(ordinal_q(proba))
    return best_thresholds(ys, qs, max_grid=MAX_SOURCE_GRID)

def d1_analysis(t, feats, baseline_folds, scorer):
    d1 = {'status': 'D1_EXPLORATORY', 'n_boot': N_BOOT, 'rank_scalar': 'q=P(Y=1)+2*P(Y=2)', 'cutpoints_source': 'selected on 3 source-site OOF predictions; grid 65; unweighted site F1', 'oracle': 'in-sample target-labelled best two cutpoints; diagnostic only, not independent', 'arms': {}}
    source_preds = {}
    argmax_preds = {}
    for arm in ARMS:
        site_records, oracle_preds = ({}, [])
        source_preds[arm], argmax_preds[arm] = ([], [])
        for ci, site in enumerate(C.COHORTS):
            f = baseline_folds[arm][ci]
            q = ordinal_q(Q.proba_for(baseline_folds[arm], ci, policy='all'))
            mask = f['test_mask']
            y = t['label'][mask]
            orig = Q.proba_for(baseline_folds[arm], ci, policy='all').argmax(axis=1)
            t1, t2, source_cv_max = source_cutpoints(t, feats[arm], baseline_folds[arm], ci, Q.CONFIGS)
            source_pred = threshold_pred(q, t1, t2)
            o1, o2, oracle_best = best_thresholds([y], [q])
            oracle_pred = threshold_pred(q, o1, o2)
            source_preds[arm].append(source_pred)
            argmax_preds[arm].append(orig)
            oracle_preds.append(oracle_pred)
            site_records[site] = {'rank': rank_metrics(y, q), 'source_thresholds': [t1, t2], 'source_cv_macro_f1': source_cv_max, 'in_sample_oracle_thresholds': [o1, o2], 'in_sample_oracle_macro_f1': oracle_best, 'source_macro_f1': Q.walk_macro_f1(y, source_pred), 'original_argmax_macro_f1': Q.walk_macro_f1(y, orig), 'threshold_regret_vs_oracle': oracle_best - Q.walk_macro_f1(y, source_pred)}
        conf_original = scorer.conf(argmax_preds[arm])
        conf_source = scorer.conf(source_preds[arm])
        conf_oracle = scorer.conf(oracle_preds)
        d1['arms'][arm] = {'per_site': site_records, 'argmax': point_metrics(scorer, conf_original), 'source_threshold': point_metrics(scorer, conf_source), 'target_label_oracle': {'macro_f1': scorer.block(*scorer.stats(conf_oracle))}, 'source_threshold_minus_argmax': scorer.contrast(conf_source, conf_original)}
    d1['PS_minus_S_argmax'] = scorer.contrast(scorer.conf(argmax_preds['PS']), scorer.conf(argmax_preds['S']))
    d1['PS_minus_S_source_threshold'] = scorer.contrast(scorer.conf(source_preds['PS']), scorer.conf(source_preds['S']))
    return d1

def permute_pose_cross_person(pose, cohort, subject, seed):
    rng = np.random.default_rng(seed)
    dest = np.empty_like(pose)
    for ci in np.unique(cohort):
        site_idx = np.flatnonzero(cohort == ci)
        ids = np.unique(subject[site_idx])
        if len(ids) < 2:
            raise C.SourceError('R1C_SITE_LT2_SUBJECTS')
        shuffled = rng.permutation(ids)
        donors = np.roll(shuffled, 1)
        for who, donor in zip(shuffled, donors):
            dest_idx = site_idx[subject[site_idx] == who]
            donor_idx = site_idx[subject[site_idx] == donor]
            dest[dest_idx] = pose[rng.choice(donor_idx, size=len(dest_idx), replace=True)]
    return dest

def d0_analysis(t, feats, baseline_folds, scorer):
    res = {'status': 'D0_EXPLORATORY', 'seeds': list(SEEDS), 'controls': {'S_plus_IID138': '138 standard-normal columns independent of labels and sites; source-only', 'S_plus_permutedPose': 'pose sourced from different people within same site, label-free; target feature permutation is transductive diagnostic'}, 'families_reported': list(POLICIES), 'seeds_results': []}
    original = {arm: {p: selected_conf(scorer, baseline_folds[arm], p) for p in POLICIES} for arm in ARMS}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((len(t['label']), feats['PS'].shape[1] - 1))
        perm = permute_pose_cross_person(t['pose'], t['cohort'], t['subject'], seed)
        controls = {'S_plus_IID138': np.column_stack((noise, t['speed'])), 'S_plus_permutedPose': np.column_stack((perm, t['speed']))}
        one = {'seed': seed, 'arms': {}}
        for name, X in controls.items():
            if X.shape != feats['PS'].shape or not np.isfinite(X).all():
                raise C.SourceError('R1C_D0_BAD_FEATURE_SHAPE')
            folds = Q.outer_run(X, t['label'], t['cohort'], t['subject'], 'walk', Q.CONFIGS, [0])
            policies = {}
            for pol in POLICIES:
                conf = selected_conf(scorer, folds, pol)
                policies[pol] = {'model_by_heldout_site': {site: Q.CONFIGS[folds[ci]['selected'][pol]][0] for ci, site in enumerate(C.COHORTS)}, 'scores': point_metrics(scorer, conf), 'delta_vs_S': scorer.contrast(conf, original['S'][pol]), 'delta_vs_PS': scorer.contrast(conf, original['PS'][pol])}
            one['arms'][name] = policies
        res['seeds_results'].append(one)
    return res
