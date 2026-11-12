"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
import sklearn
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import p0_core as P
import p1_core as Q
import r8_core as R
CARD_ID = 'c_insite_ceiling'
STATUS = 'R8_C_INSITE_CEILING_COMPLETE'
ARMS = ('S', 'PS', 'KS')
N_OUTER = 5
N_INNER = 3
N_BOOT = 2000
TOL = 1.1e-06
READ_MARGIN_1 = 0.03
READ_MARGIN_2 = 0.05
LIMITS = ['In-site arms are target-supervised ceilings on previously inspected cohorts; exploratory; never generalisation.', 'GroupKFold is deterministic given person order; folds are not stratified by grade, so small sites may lack a grade in a fold.', 'LOCO arms are the frozen P1 refits (exact parity); in-site and LOCO predictions are paired on the same walks.', 'Bootstrap intervals resample persons within site with predictions fixed; refit variance is not covered.', 'BMCLab cell-grain metrics use the person x medication cell (card A); other sites have no occasion key.', 'No novelty: nested cross-validation and in-domain versus out-of-domain comparison are standard.']

def insite_cv(X, y, subject, configs, warned, n_outer=N_OUTER, n_inner=N_INNER, scheme='walk'):
    n = y.size
    if X.shape[0] != n or subject.size != n or n == 0:
        raise C.SourceError('R8_C_INSITE_INPUT')
    if len(np.unique(subject)) < n_outer:
        raise C.SourceError('R8_C_TOO_FEW_PERSONS')
    proba = np.full((n, 3), np.nan)
    folds = []
    covered = np.zeros(n, dtype=bool)
    for tr, te in GroupKFold(n_splits=n_outer).split(X, y, subject):
        if np.intersect1d(subject[tr], subject[te]).size:
            raise C.SourceError('R8_C_PERSON_OVERLAP')
        inner = np.zeros((len(configs), n_inner))
        for k, (itr, iva) in enumerate(GroupKFold(n_splits=n_inner).split(X[tr], y[tr], subject[tr])):
            a, b = (tr[itr], tr[iva])
            w = Q.train_weights(y[a], subject[a], scheme)
            for c, cfg in enumerate(configs):
                p = Q.fit_predict_proba(cfg, X[a], y[a], w, X[b], warned)
                inner[c, k] = Q.walk_macro_f1(y[b], p.argmax(axis=1))
        best = Q.select_config(inner.mean(axis=1), list(range(len(configs))))
        w = Q.train_weights(y[tr], subject[tr], scheme)
        proba[te] = Q.fit_predict_proba(configs[best], X[tr], y[tr], w, X[te], warned)
        covered[te] = True
        folds.append({'selected': configs[best][0], 'inner_mean_by_config': {cfg[0]: float(v) for cfg, v in zip(configs, inner.mean(axis=1))}, 'n_train_persons': int(np.unique(subject[tr]).size), 'n_test_persons': int(np.unique(subject[te]).size), 'test_grades_present': [bool((y[te] == k).any()) for k in range(3)]})
    if not covered.all() or not np.isfinite(proba).all():
        raise C.SourceError('R8_C_INSITE_COVERAGE')
    return (proba, folds)

def check_loco_parity(folds_loco, scorer, p1):
    out = {}
    for arm in ARMS:
        preds = [R.selected_proba(folds_loco[arm][ci]).argmax(axis=1) for ci in range(len(scorer.names))]
        point, _ = scorer.stats(scorer.conf(preds))
        out[arm] = {}
        for ci, site in enumerate(scorer.names):
            name = Q.CONFIGS[folds_loco[arm][ci]['selected']['all']][0]
            want_name = p1['selected_configs'][arm][site]['all']
            want = p1['arms_inductive_strong'][arm]['macro_f1']['per_cohort'][site]['point']
            if name != want_name:
                raise C.SourceError('R8_C_LOCO_PARITY_CONFIG:' + arm + ':' + site)
            if abs(point[ci] - want) > TOL:
                raise C.SourceError('R8_C_LOCO_PARITY_VALUE:' + arm + ':' + site)
            out[arm][site] = {'selected_config': name, 'refit_macro_f1': float(point[ci]), 'archived_macro_f1': want}
    return out

def bmclab_cells(t, med_site, scorer):
    ci = C.COHORTS.index('BMCLab')
    site_rows = np.flatnonzero(t['cohort'] == ci)
    if len(med_site) != site_rows.size:
        raise C.SourceError('R8_C_MEDICATION_ROW_ALIGNMENT')
    keep = scorer.masks['BMCLab'][site_rows]
    med = [med_site[i] for i in range(site_rows.size) if keep[i]]
    cells = R.build_cells(scorer.subject['BMCLab'], med, scorer.y['BMCLab'])
    if cells['n_cells'] != 42 or len(cells['persons']) != 23:
        raise C.SourceError('R8_C_BMCLAB_CELLS')
    return cells

def cell_metrics(proba, cells, counts):
    cp = R.cell_predictions(proba, cells)
    pc = R.cell_person_confusions(cells, cp['pred_mean'])
    point = pc.sum(axis=0)
    boot = R.boot_from_person_conf(pc, counts)
    return ({m: R.summary(fn(point), boot[m]) for m, fn in R.METRICS.items()}, boot)

def analyze(t, feats, folds_loco, insite_proba, insite_folds, scorer, cells, warned):
    names = scorer.names
    preds = {}
    for arm in ARMS:
        preds['loco', arm] = [R.selected_proba(folds_loco[arm][ci]).argmax(axis=1) for ci in range(len(names))]
        preds['insite', arm] = [insite_proba[arm][name].argmax(axis=1) for name in names]
    conf = {k: scorer.conf(v) for k, v in preds.items()}
    arms_out = {f'{reg}_{arm}': scorer.all_metrics(conf[reg, arm]) for reg, arm in conf}
    contrasts = {}
    pairs = [(('insite', 'PS'), ('insite', 'S')), (('insite', 'KS'), ('insite', 'S')), (('insite', 'S'), ('loco', 'S')), (('insite', 'PS'), ('loco', 'PS')), (('insite', 'KS'), ('loco', 'KS')), (('loco', 'PS'), ('loco', 'S')), (('loco', 'KS'), ('loco', 'S'))]
    for a, b in pairs:
        key = f'{a[0]}_{a[1]}_minus_{b[0]}_{b[1]}'
        contrasts[key] = {'macro_f1': scorer.contrast(conf[a], conf[b]), 'macro_f1_person_weighted': scorer.contrast(conf[a], conf[b], person=True)}
        pa, da = scorer.stats(conf[a], 'ordinal_mae')
        pb, db = scorer.stats(conf[b], 'ordinal_mae')
        contrasts[key]['ordinal_mae'] = scorer.block(pa - pb, da - db)
    counts = scorer.counts[names.index('BMCLab')]
    cell_out, cell_boot = ({}, {})
    for reg in ('loco', 'insite'):
        for arm in ARMS:
            proba = R.selected_proba(folds_loco[arm][names.index('BMCLab')]) if reg == 'loco' else insite_proba[arm]['BMCLab']
            cell_out[f'{reg}_{arm}'], cell_boot[reg, arm] = cell_metrics(proba, cells, counts)
    cell_contrasts = {}
    for a, b in pairs:
        key = f'{a[0]}_{a[1]}_minus_{b[0]}_{b[1]}'
        cell_contrasts[key] = {m: R.summary(cell_out[f'{a[0]}_{a[1]}'][m]['point'] - cell_out[f'{b[0]}_{b[1]}'][m]['point'], cell_boot[a][m] - cell_boot[b][m]) for m in R.METRICS}
    ps = contrasts['insite_PS_minus_insite_S']['macro_f1']
    ks = contrasts['insite_KS_minus_insite_S']['macro_f1']

    def small(c):
        return bool(c['M']['point'] < READ_MARGIN_1 or c['M']['ci95'][0] <= 0)

    def sites_clear(c):
        return sum((1 for n in names if c['per_cohort'][n]['point'] >= READ_MARGIN_2 and c['per_cohort'][n]['ci95'][0] > 0))
    loco_ps_neg = bool(contrasts['loco_PS_minus_loco_S']['macro_f1']['M']['point'] < 0)
    r1 = small(ps) and small(ks)
    r2 = bool(sites_clear(ps) >= 2 and loco_ps_neg)
    reading = 'NO_INSITE_INFORMATION_BEYOND_SPEED' if r1 and (not r2) else 'INSITE_INFORMATION_DOES_NOT_TRANSPORT' if r2 and (not r1) else 'MIXED'
    transport = {arm: {'per_cohort': {n: contrasts[f'insite_{arm}_minus_loco_{arm}']['macro_f1']['per_cohort'][n]['point'] for n in names}, 'M': contrasts[f'insite_{arm}_minus_loco_{arm}']['macro_f1']['M']['point']} for arm in ARMS}
    return {'arms': arms_out, 'contrasts': contrasts, 'bmclab_cell_grain': {'arms': cell_out, 'contrasts': cell_contrasts, 'n_cells': int(cells['n_cells'])}, 'insite_folds': {arm: {name: insite_folds[arm][name] for name in names} for arm in ARMS}, 'loco_selected': {arm: {name: Q.CONFIGS[folds_loco[arm][ci]['selected']['all']][0] for ci, name in enumerate(names)} for arm in ARMS}, 'transport_loss_insite_minus_loco_macro_f1': transport, 'reading': {'value': reading, 'rule1_no_insite_info': r1, 'rule2_info_not_transported': r2, 'sites_with_clear_PS_gain': sites_clear(ps), 'sites_with_clear_KS_gain': sites_clear(ks), 'margins': {'rule1': READ_MARGIN_1, 'rule2': READ_MARGIN_2}}, 'n_convergence_warnings': warned[0]}

def collect_medication(obj):
    med = []
    for _, rec in C.walks_of(obj):
        pose, trans = C.read_arrays(rec)
        if not (np.isfinite(pose).all() and np.isfinite(trans).all()):
            continue
        med.append(R.normalized_medication(rec.get('medication')))
    return med

def run(t, med_site, configs=Q.CONFIGS, n_boot=N_BOOT, p1=None):
    feats = Q.arm_features(t)
    warned = [0]
    scorer = Q.Scorer(t, n_boot)
    names = scorer.names
    t0 = time.monotonic()
    folds_loco = {arm: [R.single_outer_fold(feats[arm], t['label'], t['cohort'], t['subject'], 'walk', configs, warned, ci) for ci in range(len(names))] for arm in ARMS}
    parity = check_loco_parity(folds_loco, scorer, p1) if p1 is not None else 'SKIPPED_FABRICATED'
    t1 = time.monotonic()
    insite_proba, insite_folds = ({arm: {} for arm in ARMS}, {arm: {} for arm in ARMS})
    for arm in ARMS:
        for name in names:
            m = scorer.masks[name]
            insite_proba[arm][name], insite_folds[arm][name] = insite_cv(feats[arm][m], scorer.y[name], scorer.subject[name], configs, warned)
    t2 = time.monotonic()
    cells = bmclab_cells(t, med_site, scorer)
    body = analyze(t, feats, folds_loco, insite_proba, insite_folds, scorer, cells, warned)
    body['loco_parity'] = parity
    body['timing_s'] = {'loco_refits': round(t1 - t0, 1), 'insite_cv': round(t2 - t1, 1)}
    body['protocol'] = {'n_outer_person_folds': N_OUTER, 'n_inner_person_folds': N_INNER, 'candidates': [c[0] for c in configs], 'selection': 'inner mean walk macro-F1, earliest tie', 'weights': 'P1 walk class-balanced', 'n_boot': n_boot, 'bootstrap_seed': Q.SEED}
    return body
