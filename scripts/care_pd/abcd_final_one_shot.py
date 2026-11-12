"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import abcd_baseline as AB
import abcd_common as AC
import abcd_privacy as AP
import abcd_stage_a as SA
import abcd_stage_b as SB
import abcd_stage_c as SC
import abcd_stage_d as SD
import common as C
import p1_core as Q
SCHEMA = 'gait_abcd_final_v1'
S_PARITY_TOL = 1.1e-06
FINAL_BUDGET_S = 2700
CANDIDATES = ('S_plus_A', 'S_plus_B', 'S_plus_C', 'S_plus_D')
CONTROLS = ('ERM_RES_A', 'ERM_RES_B', 'ERM_RES_C', 'VREX_RES_C', 'ERM_RES_D')
B_FEATURE_ARMS = ('S_plus_B', 'ERM_RES_B', 'S_plus_C', 'ERM_RES_C', 'VREX_RES_C', 'S_plus_D', 'ERM_RES_D')
LIMITS = ["All four CARE-PD cohorts have been inspected repeatedly; 3DGait's historical aggregates were known before this cycle, so the four-cohort score is a pseudo-holdout and every reading is EXPLORATORY, not independent validation.", 'Four registered candidates (A, B, C, D) were developed sequentially on three of the cohorts; the analysis is exploratory; no multiplicity adjustment is applied and no best-of-four claim is made.', "Only each cohort's reported 0/1/2 label is predicted (R6/A0: no common clinical construct identified).", 'Intervals resample persons within the four fixed cohorts with predictions fixed; they cover neither refits nor new sites. 107 persons behind 2,908 walks.', 'S uses a speed proxy in unverified trans units; B/C/D features are an observable hip-rotation cycle proxy, not gait events.']

class FinalError(C.SourceError):
    pass

def build_inputs(objs):
    t = Q.build_table(objs)
    Q.check_stops(t)
    b = AC.build_bundle(objs, C.COHORTS)
    el = np.isin(t['label'], (0, 1, 2))
    if not (np.array_equal(b['label'], t['label'][el]) and np.array_equal(b['speed'], t['speed'][el]) and np.array_equal(b['duration'], t['duration'][el]) and np.array_equal(b['site'], t['cohort'][el])):
        raise FinalError('ABCD_FINAL_ROW_ALIGNMENT')
    return (t, b)

def s_parity(selected, point_s, p1):
    ref = p1['arms_inductive_strong']['S']['macro_f1']
    rows, ok = ({}, True)
    for ci, site in enumerate(C.COHORTS):
        want_sel = p1['selected_configs']['S'][site]['all']
        want = ref['per_cohort'][site]['point']
        good = selected[ci] == want_sel and abs(point_s[ci] - want) <= S_PARITY_TOL
        rows[site] = {'selected': selected[ci], 'archived_selected': want_sel, 'macro_f1': float(point_s[ci]), 'archived': want, 'pass': bool(good)}
        ok &= good
    m_ok = abs(float(point_s.mean()) - ref['M']['point']) <= S_PARITY_TOL
    return {'status': 'PASS' if ok and m_ok else 'FAIL', 'tolerance': S_PARITY_TOL, 'per_site': rows, 'M': float(point_s.mean()), 'archived_M': ref['M']['point']}

def observability(FB, b):
    out = {}
    for ci, s in enumerate(C.COHORTS):
        m = b['site'] == ci
        out[s] = {'walks': int(m.sum()), 'observable_share': float(np.isfinite(FB[m, 0]).mean())}
    return out

def support(b):
    out, ok = ({}, True)
    for ci, s in enumerate(C.COHORTS):
        m = b['site'] == ci
        persons = [int(np.unique(b['subject'][m & (b['label'] == k)]).size) for k in range(3)]
        walks = [int((m & (b['label'] == k)).sum()) for k in range(3)]
        good = min(persons) >= SC.MIN_PERSONS_PER_GRADE and min(walks) >= SC.MIN_WALKS_PER_GRADE
        out[s] = {'support_pass': bool(good)}
        ok &= good
    return (ok, out)

def site_decodability(F, b, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    X = np.where(np.isfinite(F), F, np.nan)
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isfinite(X), X, col_mean)
    pred = np.empty(b['site'].size, dtype=int)
    for tr, va in GroupKFold(n_splits=5).split(X, b['site'], b['subject']):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(sc.transform(X[tr]), b['site'][tr])
        pred[va] = lr.predict(sc.transform(X[va]))
    recalls = [float((pred[b['site'] == k] == k).mean()) for k in range(4)]
    return float(np.mean(recalls))

def _rows(F, b, idx, labels):
    out = {'F': F[idx], 'speed': b['speed'][idx], 'site': b['site'][idx]}
    if labels:
        out['label'], out['subject'] = (b['label'][idx], b['subject'][idx])
    return out

def c_controls(FB, b, src, tgt, s_head, warned):
    s_rows, t_rows = (_rows(FB, b, src, True), _rows(FB, b, tgt, False))
    _, phi_s, phi_t, O_s, o_tgt, _ = SC.prepare(s_rows, t_rows, s_head, warned)
    y, env = (s_rows['label'], s_rows['site'])
    erm = AC.OffsetResidual().fit(phi_s, O_s, y)
    vrex = AC.OffsetResidual(penalty='vrex', beta=SC.BETA).fit(phi_s, O_s, y, env=env)
    return ({'ERM_RES_C': erm.logits(phi_t, o_tgt), 'VREX_RES_C': vrex.logits(phi_t, o_tgt)}, {'vrex_tau': vrex.tau, 'erm_tau': erm.tau})

def run_arms(FA, FB, b, run_b, run_c, budget, warned):
    X = b['speed'][:, None]
    preds = {'S': []}
    folds, selected = ({}, [])
    for ci, site in enumerate(C.COHORTS):
        budget()
        src, tgt = (np.flatnonzero(b['site'] != ci), np.flatnonzero(b['site'] == ci))
        if np.intersect1d(b['subject'][src], b['subject'][tgt]).size:
            raise FinalError('ABCD_FINAL_PERSON_OVERLAP')
        s_head = AB.s_fold(X[src], b['label'][src], b['site'][src], b['subject'][src], X[tgt], warned)
        selected.append(s_head['name'])
        preds['S'].append(s_head['proba'].argmax(axis=1))
        arms, diag = ({}, {'S_selected': s_head['name']})
        out = SA.method_fold(_rows(FA, b, src, True), _rows(FA, b, tgt, False), s_head, warned)
        arms.update(out['arms'])
        diag['A'] = {'gate_open': out['diag']['gate_open'], 'inner_deltas': out['diag']['inner_deltas']}
        if run_b:
            out = SB.method_fold(_rows(FB, b, src, True), _rows(FB, b, tgt, False), s_head, warned)
            arms.update(out['arms'])
            diag['B'] = {'gate_open': out['diag']['gate_open'], 'inner_deltas': out['diag']['inner_deltas']}
            ctl, cd = c_controls(FB, b, src, tgt, s_head, warned)
            arms.update(ctl)
            diag['C_controls'] = cd
            if run_c:
                out = SC.method_fold(_rows(FB, b, src, True), _rows(FB, b, tgt, False), s_head, warned)
                arms['S_plus_C'] = out['arms']['S_plus_C']
                diag['C'] = {'gate_open': out['diag']['gate_open'], 'held_out_gain_by_env': out['diag']['fallback_held_out_macro_f1_gain_by_env']}
            out = SD.method_fold(_rows(FB, b, src, True), _rows(FB, b, tgt, False), s_head, warned)
            arms.update(out['arms'])
            diag['D'] = {'c_star': out['diag']['c_star'], 'held_out_person_ce': out['diag']['held_out_person_ce']}
        for arm, z in arms.items():
            z = np.asarray(z, dtype=np.float64)
            if z.shape != (tgt.size, 3) or not np.isfinite(z).all():
                raise FinalError('ABCD_FINAL_ARM_OUTPUT:' + arm)
            preds.setdefault(arm, []).append(z.argmax(axis=1))
        diag['share_changed_vs_S'] = {a: float(np.mean(preds[a][-1] != preds['S'][-1])) for a in arms}
        folds[site] = diag
    return (preds, folds, selected)

def verdict(decisions, parity_ok):
    if not parity_ok:
        return 'FINAL_BLOCKED'
    run = [d['status'] for d in decisions.values() if d['status'] not in ('NOT_RUN_ENV_SUPPORT', 'NOT_RUN_PHASE_NOT_OBSERVABLE')]
    if any((s == 'EMPIRICAL_GAIN_EXPLORATORY' for s in run)):
        return 'FINAL_EMPIRICAL_GAIN_EXPLORATORY'
    if any((s == 'INCONCLUSIVE' for s in run)):
        return 'FINAL_INCONCLUSIVE'
    if not run:
        return 'FINAL_BLOCKED'
    return 'FINAL_NO_DETECTED_GAIN'

def run_final(t, b, FA, FB, p1, n_boot=Q.N_BOOT, budget=lambda: None):
    warned = [0]
    obs = observability(FB, b)
    sup_ok, sup = support(b)
    run_b = all((v['observable_share'] >= SB.OBSERVABILITY_MIN_SHARE for v in obs.values()))
    run_c = run_b and sup_ok
    scorer = Q.Scorer(t, n_boot)
    X = b['speed'][:, None]
    s_preds, s_sel = ([], [])
    for ci in range(len(C.COHORTS)):
        src, tgt = (np.flatnonzero(b['site'] != ci), np.flatnonzero(b['site'] == ci))
        s_head = AB.s_fold(X[src], b['label'][src], b['site'][src], b['subject'][src], X[tgt], warned)
        s_preds.append(s_head['proba'].argmax(axis=1))
        s_sel.append(s_head['name'])
    point_s, _ = scorer.stats(scorer.conf(s_preds))
    parity = s_parity(s_sel, point_s, p1) if p1 is not None else {'status': 'NOT_CHECKED_FABRICATED'}
    if parity['status'] == 'FAIL':
        return {'S_parity': parity, 'terminal_verdict': 'FINAL_BLOCKED', 'candidates_scored': False, 'observability': obs, 'support': sup}
    preds, folds, selected = run_arms(FA, FB, b, run_b, run_c, budget, warned)
    if selected != s_sel or not all((np.array_equal(a, c) for a, c in zip(preds['S'], s_preds))):
        raise FinalError('ABCD_FINAL_S_NOT_REPRODUCED_IN_ARM_PASS')
    conf = {a: scorer.conf(p) for a, p in preds.items()}
    arms = {a: scorer.all_metrics(conf[a]) for a in preds}
    contrasts = {f'{a}_minus_S': {'walk_eval': scorer.contrast(conf[a], conf['S']), 'person_weighted_eval': scorer.contrast(conf[a], conf['S'], person=True)} for a in preds if a != 'S'}
    decisions = {}
    for cand in CANDIDATES:
        if cand not in preds:
            decisions[cand] = {'status': 'NOT_RUN_ENV_SUPPORT' if cand == 'S_plus_C' and run_b else 'NOT_RUN_PHASE_NOT_OBSERVABLE', 'conditions': None}
            continue
        blk = contrasts[f'{cand}_minus_S']['walk_eval']
        decisions[cand] = AC.final_decision({'per_site': blk['per_cohort'], 'M': blk['M']})
    not_run = [a for a in CANDIDATES + CONTROLS if a not in preds]
    return {'S_parity': parity, 'candidates_scored': True, 'observability': obs, 'support': sup, 'arms': arms, 'contrasts': contrasts, 'decisions': decisions, 'terminal_verdict': verdict(decisions, parity['status'] != 'FAIL'), 'not_run_arms': not_run, 'folds': folds, 'pooled_per_grade_recall_suppressed': {a: AP.pooled_recall_rows(conf[a]) for a in preds}, 'denominators': {s: {'walks_eligible': int(scorer.masks[s].sum()), 'persons_eligible': int(len(scorer.subjects[s]))} for s in scorer.names}, 'n_convergence_warnings_downstream': warned[0], 'n_boot': n_boot}
