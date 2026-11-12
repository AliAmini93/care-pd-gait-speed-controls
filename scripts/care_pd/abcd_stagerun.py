"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
import sklearn
sys.path.insert(0, str(Path(__file__).resolve().parent))
import abcd_baseline as AB
import abcd_common as AC
import abcd_privacy as AP
import abcd_worlds as W
import common as C
import p1_core as Q
TRAIN_BUDGET_S = 2700
LIMITS = ['Development layer only: three previously inspected CARE-PD cohorts (BMCLab, PD-GaM, T-SDU-PD), leave-one-development-cohort-out, two source cohorts per fold; a triage screen, not confirmation.', '3DGait was not opened by this route; its historical aggregates are known from earlier work, so the later four-cohort score is a pseudo-holdout, not independent validation.', "Only each site's reported 0/1/2 label is predicted (R6/A0: no common clinical construct identified).", 'Walks are repeated measures of few persons (67 persons, 2,828 walks); bootstrap intervals resample persons within fixed sites with predictions fixed; they cover neither refits nor new sites.', 'Synthetic worlds are author-designed and can privilege a mechanism; synthetic success is not clinical evidence.', 'Stages are adapted sequentially from earlier stage lessons, so every clinical reading is exploratory.']

def _rows(F, b, idx, with_labels):
    out = {'F': F[idx], 'speed': b['speed'][idx], 'site': b['site'][idx]}
    if with_labels:
        out['label'] = b['label'][idx]
        out['subject'] = b['subject'][idx]
    return out

def _fold(stage, F, b, src, tgt, warned):
    X = b['speed'][:, None]
    s_head = AB.s_fold(X[src], b['label'][src], b['site'][src], b['subject'][src], X[tgt], warned)
    out = stage.method_fold(_rows(F, b, src, True), _rows(F, b, tgt, False), s_head, warned)
    preds = {'S': s_head['proba'].argmax(axis=1)}
    for arm, z in out['arms'].items():
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (tgt.size, 3) or not np.isfinite(z).all():
            raise AC.AbcdError('ABCD_ARM_OUTPUT:' + arm)
        preds[arm] = z.argmax(axis=1)
    return (s_head, preds, out['diag'])

def run_synthetic(stage, seed_set='test'):
    worlds = list(W.WORLDS) + list(stage.SYNTH_FALSIFIERS)
    res = {}
    for world in worlds:
        reps = []
        for r in range(W.REPLICATES[seed_set]):
            data = W.generate(world, seed_set, r)
            b = AC.synthetic_bundle(data)
            src, tgt = W.split(data)
            F = stage.features(b['store'])
            warned = [0]
            s_head, preds, diag = _fold(stage, F, b, src, tgt, warned)
            y = b['label'][tgt]
            f1 = {arm: AC.macro_f1(y, p) for arm, p in preds.items()}
            reps.append({'macro_f1': f1, 'delta_vs_S': {a: f1[a] - f1['S'] for a in f1 if a != 'S'}, 'S_selected': s_head['name'], 'diag': diag, 'n_convergence_warnings': warned[0]})
            del data, b
        arms = [a for a in reps[0]['macro_f1']]
        res[world] = {'replicates': reps, 'mean_macro_f1': {a: float(np.mean([x['macro_f1'][a] for x in reps])) for a in arms}, 'mean_delta_vs_S': {a: float(np.mean([x['delta_vs_S'][a] for x in reps])) for a in arms if a != 'S'}}
    cand = stage.CANDIDATE
    screen = AC.synthetic_screen({w: res[w]['mean_delta_vs_S'][cand] for w in ('W1', 'W2', 'W3')})
    return {'seed_set': seed_set, 'replicates': W.REPLICATES[seed_set], 'worlds': res, 'screen': screen}

def run_dev(stage, objs, t_start):
    if set(objs) != set(AC.DEV_SITES):
        raise AC.AbcdError('ABCD_DEV_SITES')
    b = AC.build_bundle(objs, AC.DEV_SITES)
    AC.check_dev_bundle(b)
    t_feat = time.monotonic()
    F = stage.features(b['store'])
    if F.shape[0] != b['label'].size:
        raise AC.AbcdError('ABCD_FEATURE_ROWS')
    feat_seconds = time.monotonic() - t_feat
    codes = [C.COHORTS.index(s) for s in AC.DEV_SITES]
    preds, folds, warned = ({}, {}, [0])
    for site, code in zip(AC.DEV_SITES, codes):
        if time.monotonic() - t_start > TRAIN_BUDGET_S:
            raise AC.AbcdError('ABCD_TIME_BUDGET')
        src, tgt = (np.flatnonzero(b['site'] != code), np.flatnonzero(b['site'] == code))
        if np.intersect1d(b['subject'][src], b['subject'][tgt]).size:
            raise AC.AbcdError('ABCD_PERSON_OVERLAP')
        s_head, p, diag = _fold(stage, F, b, src, tgt, warned)
        for arm, v in p.items():
            preds.setdefault(arm, {})[site] = v
        folds[site] = {'S_selected': s_head['name'], 'S_inner_mean_selected': float(s_head['inner'].mean(axis=1).max()), 'source_sites': [s for s in AC.DEV_SITES if s != site], 'diag': diag}
    y = {s: b['label'][b['site'] == c] for s, c in zip(AC.DEV_SITES, codes)}
    subj = {s: b['subject'][b['site'] == c] for s, c in zip(AC.DEV_SITES, codes)}
    scorer = AC.SiteScorer(y, subj, AC.DEV_SITES)
    conf = {arm: scorer.conf(preds[arm]) for arm in preds}
    arms = {arm: scorer.metrics(conf[arm]) for arm in preds}
    contrasts = {f'{arm}_minus_S': {'walk_eval': scorer.contrast(conf[arm], conf['S']), 'person_weighted_eval': scorer.contrast(conf[arm], conf['S'], person=True)} for arm in preds if arm != 'S'}
    screen = AC.dev_screen(contrasts[f'{stage.CANDIDATE}_minus_S']['walk_eval'])
    changed = {arm: {s: float(np.mean(preds[arm][s] != preds['S'][s])) for s in AC.DEV_SITES} for arm in preds if arm != 'S'}
    denominators = {s: {'walks_eligible': int(y[s].size), 'persons_eligible': int(np.unique(subj[s]).size)} for s in AC.DEV_SITES}
    return {'arms': arms, 'contrasts': contrasts, 'screen': screen, 'folds': folds, 'share_target_predictions_changed_vs_S': changed, 'pooled_per_grade_recall_suppressed': {arm: AP.pooled_recall_rows(conf[arm]) for arm in preds}, 'denominators': denominators, 'n_convergence_warnings_downstream': warned[0], 'feature_seconds': round(feat_seconds, 1), 'n_boot': scorer.n_boot}
