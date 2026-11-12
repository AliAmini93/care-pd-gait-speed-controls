"""Fabricated-only tests for R8 card c_insite_ceiling (never reads real data; values are not evidence)."""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'care_pd'))
sys.path.insert(1, str(Path(__file__).resolve().parent))
import common as C
import p0_core as P
import p1_core as Q
import r8_core as R
import r8_c_insite_ceiling as CC
import r8_a_occasion_grain_synthetic_tests as AT
FAILS = []
MARK = AT.MARK
FAST = AT.FAST

def check(cond, name):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond:
        FAILS.append(name)

def raises_source(fn):
    try:
        fn()
    except C.SourceError:
        return True
    return False

def pose_world(seed):
    """Fabricated world where labels drive knee amplitude only and speed is noise (P1 smoke 'pose' mode)."""
    objs = AT.synth_objs('speed', seed=seed)
    rng = np.random.default_rng(seed + 7)
    for site in C.COHORTS:
        for walks in objs[site].values():
            for rec in walks.values():
                lab = rec['UPDRS_GAIT']
                T = rec['pose'].shape[0]
                tt = np.arange(T) / rec['fps']
                pose = rec['pose'].astype(np.float64)
                pose[:, 3 * 4] = 0.0
                pose[:, 3 * 5] = 0.0
                knee = 0.9 - 0.3 * min(lab, 2) + rng.normal(0, 0.06)
                pose[:, 3 * 4] += knee * np.sin(2 * np.pi * tt)
                pose[:, 3 * 5] += knee * np.sin(2 * np.pi * tt + np.pi)
                rec['pose'] = pose.astype(np.float32)
                trans = rec['trans'].astype(np.float64)
                trans[:, 2] = max(0.05, 1.0 + rng.normal(0, 0.12)) * tt
                rec['trans'] = trans.astype(np.float32)
    return objs

def test_insite_cv_mechanics():
    rng = np.random.default_rng(2)
    n = 300
    subject = np.repeat(np.arange(30), 10)
    y = rng.integers(0, 3, n)
    X = np.column_stack([y + rng.normal(0, 0.3, n), rng.normal(size=n)])
    proba, folds = CC.insite_cv(X, y, subject, FAST, [0])
    check(proba.shape == (n, 3) and np.isfinite(proba).all() and np.allclose(proba.sum(axis=1), 1), 'out-of-fold probabilities cover every walk once')
    check(len(folds) == CC.N_OUTER and all((f['selected'] in [c[0] for c in FAST] for f in folds)), 'five outer folds each select one candidate')
    check(Q.walk_macro_f1(y, proba.argmax(axis=1)) > 0.8, 'a label-driven feature is recovered in-site')
    proba2, _ = CC.insite_cv(X, y, subject, FAST, [0])
    check(np.allclose(proba, proba2), 'in-site CV is deterministic')
    check(raises_source(lambda: CC.insite_cv(X, y, np.zeros(n, int), FAST, [0])), 'fewer persons than folds is refused')
    check(raises_source(lambda: CC.insite_cv(X[:-1], y, subject, FAST, [0])), 'row mismatch is refused')
    y_alt = y.copy()
    s0 = subject == 0
    y_alt[s0] = (y_alt[s0] + 1) % 3
    proba_alt, _ = CC.insite_cv(X, y_alt, subject, FAST, [0])
    check(np.allclose(proba[s0], proba_alt[s0]), "a person's own labels never influence that person's out-of-fold prediction (person isolation)")

def test_end_to_end():
    objs = AT.synth_objs('speed', seed=61)
    t = Q.build_table(objs)
    med = CC.collect_medication(objs['BMCLab'])
    body = CC.run(t, med, configs=FAST, n_boot=40, p1=None)
    s = body['arms']['insite_S']['macro_f1']['M']['point']
    check(s > 0.5, f'speed world: in-site speed arm recovers the signal ({s:.3f})')
    rd = body['reading']['value']
    check(rd == 'NO_INSITE_INFORMATION_BEYOND_SPEED', f'speed world: reading is no in-site information beyond speed ({rd})')
    check(body['bmclab_cell_grain']['n_cells'] == 42, 'BMCLab cell grain uses 42 cells')
    text = R.render(body)
    check(MARK not in text, 'no fabricated identifier reaches the output')
    check(len(text.encode()) < 262144 * 0.6, f'output well under the summary cap ({len(text)} bytes)')
    for arm in CC.ARMS:
        for site in C.COHORTS:
            check(len(body['insite_folds'][arm][site]) == CC.N_OUTER, f'{arm}/{site}: five in-site folds recorded')
    objs_p = pose_world(62)
    t_p = Q.build_table(objs_p)
    med_p = CC.collect_medication(objs_p['BMCLab'])
    body_p = CC.run(t_p, med_p, configs=FAST, n_boot=40, p1=None)
    gain = body_p['contrasts']['insite_PS_minus_insite_S']['macro_f1']['M']['point']
    check(gain > 0.1, f'pose world: in-site PS beats in-site S ({gain:+.3f})')
    check(body_p['reading']['value'] != 'NO_INSITE_INFORMATION_BEYOND_SPEED', f"pose world: reading is not the null ({body_p['reading']['value']})")
    feats = Q.arm_features(t)
    scorer = Q.Scorer(t, 10)
    folds = {arm: [R.single_outer_fold(feats[arm], t['label'], t['cohort'], t['subject'], 'walk', FAST, [0], ci) for ci in range(4)] for arm in CC.ARMS}
    fake = {'selected_configs': {arm: {site: {'all': 'NOT_A_CONFIG'} for site in C.COHORTS} for arm in CC.ARMS}, 'arms_inductive_strong': {arm: {'macro_f1': {'per_cohort': {site: {'point': 0.5} for site in C.COHORTS}}} for arm in CC.ARMS}}
    check(raises_source(lambda: CC.check_loco_parity(folds, scorer, fake)), 'LOCO parity refuses a mismatching archive')
    good = {'selected_configs': {arm: {site: {'all': FAST[folds[arm][ci]['selected']['all']][0]} for ci, site in enumerate(C.COHORTS)} for arm in CC.ARMS}, 'arms_inductive_strong': {}}
    for arm in CC.ARMS:
        preds = [R.selected_proba(folds[arm][ci]).argmax(axis=1) for ci in range(4)]
        pt, _ = scorer.stats(scorer.conf(preds))
        good['arms_inductive_strong'][arm] = {'macro_f1': {'per_cohort': {site: {'point': float(pt[ci])} for ci, site in enumerate(C.COHORTS)}}}
    saved = Q.CONFIGS
    Q.CONFIGS = FAST
    try:
        par = CC.check_loco_parity(folds, scorer, good)
        check(all((par[arm][site]['refit_macro_f1'] == par[arm][site]['archived_macro_f1'] for arm in CC.ARMS for site in C.COHORTS)), 'LOCO parity passes on a matching archive')
    finally:
        Q.CONFIGS = saved
    bad_med = list(med)
    bad_med[0] = 'unknown'
    check(raises_source(lambda: CC.bmclab_cells(t, bad_med, scorer)), 'BMCLab cell construction stops on an unrecognised medication value')

def main():
    test_insite_cv_mechanics()
    test_end_to_end()
    if FAILS:
        print(f'R8_C_SYNTHETIC_FAIL {len(FAILS)}: ' + '; '.join(FAILS))
        return 1
    print('R8_C_SYNTHETIC_PASS all fabricated checks passed (no real data read)')
    return 0
if __name__ == '__main__':
    sys.exit(main())
