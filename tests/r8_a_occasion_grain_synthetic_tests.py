"""Fabricated-only tests for R8 card a_occasion_grain (never reads real data; values are not evidence)."""
from __future__ import annotations
import json
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
import r8_a_occasion_grain as A
FAILS = []
MARK = 'SYNTHID'
FPS = {'3DGait': 30, 'BMCLab': 150, 'PD-GaM': 25, 'T-SDU-PD': 30}
FAST = tuple(((n, f, dict(p, n_estimators=10) if f == 'RF' else dict(p, max_iter=8) if f == 'HGB' else p) for n, f, p in Q.CONFIGS))

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

def bmclab_cells(rng):
    """23 people, 42 cells (19 ON/OFF pairs + 4 singles), 8 discordant pairs, 781 walks with
    class totals (341, 276, 164). Returns per-walk (person, state, grade)."""
    n_cells_per_grade = (18, 15, 9)
    totals = C.EXPECTED_CLASS_WALKS['BMCLab'][:3]
    grades = np.repeat(np.arange(3), n_cells_per_grade)
    cells = []
    g = list(grades)
    rng.shuffle(g)
    pairs, singles = ([], [])
    pool = sorted(g)
    conc = []
    for k in range(3):
        same = [x for x in pool if x == k]
        while len(conc) < 11 and len(same) >= 2:
            conc.append((k, k))
            same = same[2:]
            pool.remove(k)
            pool.remove(k)
    disc = []
    while len(disc) < 8:
        a, b = rng.choice(len(pool), 2, replace=False)
        if pool[a] != pool[b]:
            disc.append((pool[a], pool[b]))
            for x in sorted([a, b], reverse=True):
                pool.pop(x)
    assert len(conc) == 11 and len(disc) == 8 and (len(pool) == 4)
    pairs = conc + disc
    singles = pool
    person, state, grade = ([], [], [])
    cell_rows = []
    pid = 0
    for a, b in pairs:
        cell_rows.append((pid, 'ON', a))
        cell_rows.append((pid, 'OFF', b))
        pid += 1
    for a in singles:
        cell_rows.append((pid, rng.choice(['ON', 'OFF']), a))
        pid += 1
    assert pid == 23 and len(cell_rows) == 42
    sizes = {}
    for k in range(3):
        idx = [i for i, (_, _, gg) in enumerate(cell_rows) if gg == k]
        n = totals[k]
        base = np.full(len(idx), 3)
        extra = rng.multinomial(n - 3 * len(idx), np.ones(len(idx)) / len(idx))
        for i, s in zip(idx, base + extra):
            sizes[i] = int(s)
    target_on = 410

    def on_total():
        return sum((sizes[i] for i, (_, s, _) in enumerate(cell_rows) if s == 'ON'))
    guard = 0
    while on_total() != target_on:
        guard += 1
        assert guard < 5000
        need = target_on - on_total()
        src_state, dst_state = ('OFF', 'ON') if need > 0 else ('ON', 'OFF')
        moved = False
        for k in range(3):
            src = [i for i, (_, s, gg) in enumerate(cell_rows) if s == src_state and gg == k and (sizes[i] > 3)]
            dst = [i for i, (_, s, gg) in enumerate(cell_rows) if s == dst_state and gg == k]
            if src and dst:
                sizes[src[0]] -= 1
                sizes[dst[0]] += 1
                moved = True
                break
        assert moved
    for i, (p, s, gg) in enumerate(cell_rows):
        for _ in range(sizes[i]):
            person.append(p)
            state.append(s)
            grade.append(gg)
    order = rng.permutation(len(person))
    return (np.asarray(person)[order], [state[i] for i in order], np.asarray(grade)[order])

def synth_objs(mode, seed=0):
    rng = np.random.default_rng(seed)
    objs = {}
    for name in C.COHORTS:
        n_subj, fps = (C.EXPECTED_SUBJECTS[name], FPS[name])
        if name == 'BMCLab':
            assign, states, labels = bmclab_cells(rng)
        else:
            labels = np.repeat(np.arange(4), C.EXPECTED_CLASS_WALKS[name])
            rng.shuffle(labels)
            assign = np.concatenate([np.arange(n_subj), rng.integers(0, n_subj, labels.size - n_subj)])
            rng.shuffle(assign)
            states = [None] * labels.size
        cohort = {f'{MARK}_{name}_{i}': {} for i in range(n_subj)}
        T = 2 * fps
        tt = np.arange(T) / fps
        for w, (lab, s, st) in enumerate(zip(labels, assign, states)):
            v = 1.4 - 0.4 * lab + rng.normal(0, 0.12) if mode == 'speed' else 1.0 + rng.normal(0, 0.12)
            trans = np.column_stack([rng.normal(0, 0.001, T), rng.normal(0, 0.001, T), max(0.05, v) * tt]).astype(np.float32)
            pose = rng.normal(0, 0.01, (T, 72))
            knee = 0.6 + rng.normal(0, 0.06)
            pose[:, 3 * 4] += knee * np.sin(2 * np.pi * tt)
            pose[:, 3 * 5] += knee * np.sin(2 * np.pi * tt + np.pi)
            med = None if st is None else st.lower() if w % 2 else ' ' + st + ' '
            cohort[f'{MARK}_{name}_{s}'][f'{MARK}_w{w}'] = {'pose': pose.astype(np.float32), 'trans': trans, 'beta': np.zeros(10, dtype=np.float32), 'fps': fps, 'UPDRS_GAIT': int(lab), 'medication': med, 'other': None}
        objs[name] = cohort
    return objs

def test_cells_and_weights():
    subj = np.array([0, 0, 0, 0, 1, 1, 2])
    state = ['ON', 'ON', 'OFF', 'OFF', 'ON', 'ON', 'OFF']
    y = np.array([0, 0, 1, 1, 2, 2, 1])
    cells = R.build_cells(subj, state, y)
    check(cells['n_cells'] == 4 and list(cells['size']) == [2, 2, 2, 1], 'cells are (person,state) with correct sizes')
    wB = R.estimand_weights(cells, 'B_cell_reweighted')
    per_cell = np.zeros(4)
    np.add.at(per_cell, cells['cell'], wB)
    check(np.allclose(per_cell, 1), 'B: every cell totals exactly one despite unequal sizes')
    wD = R.estimand_weights(cells, 'D_person_cell_balanced')
    per_person = np.zeros(3)
    np.add.at(per_person, cells['person_idx'][cells['cell']], wD)
    check(np.allclose(per_person, 1), 'D: every person totals exactly one')
    check(np.allclose(wD[subj == 0], 0.25) and np.allclose(wD[subj == 2], 1.0), 'D: two-cell person splits mass equally across states')
    check(np.allclose(R.estimand_weights(cells, 'A_walk'), 1), 'A: unit weights')
    mixed = R.build_cells.__wrapped__ if hasattr(R.build_cells, '__wrapped__') else None
    check(raises_source(lambda: R.build_cells(subj, state, np.array([0, 1, 1, 1, 2, 2, 1]))), 'a mixed-grade cell is refused')
    check(raises_source(lambda: R.build_cells(subj, ['ON'] * 6 + ['unknown'], y)), 'an unrecognised state is refused')
    check(raises_source(lambda: R.build_cells(subj, state[:-1], y)), 'length mismatch is refused')
    s = R.cell_structure(cells, y)
    check(s['n_two_state_people'] == 1 and s['n_single_state_people'] == 2 and (s['n_discordant_two_state_people'] == 1), 'structure counts two-state, single-state and discordant persons')
    check(R.normalized_medication(' on ') == 'ON' and R.normalized_medication('Off') == 'OFF' and (R.normalized_medication('on-med') == 'unknown') and (R.normalized_medication(None) == 'unknown'), 'medication parsing is strict (R4W rule)')

def test_estimands_differ():
    """Two grade-0 cells (30 walks all right; 2 walks all wrong) and one grade-1 cell (10 walks, 5 right)."""
    subj = np.array([0] * 30 + [1] * 2 + [2] * 10)
    state = ['ON'] * 42
    y = np.array([0] * 32 + [1] * 10)
    pred = np.array([0] * 30 + [1] * 2 + [1] * 5 + [0] * 5)
    cells = R.build_cells(subj, state, y)
    confA = R.weighted_confusion(y, pred, R.estimand_weights(cells, 'A_walk'))
    confB = R.weighted_confusion(y, pred, R.estimand_weights(cells, 'B_cell_reweighted'))
    fA, fB = (P.macro_f1_conf(confA), P.macro_f1_conf(confB))
    check(abs(fA - fB) > 0.05, f'walk-weighted and cell-balanced macro-F1 differ on unequal clusters ({fA:.3f} vs {fB:.3f})')
    ident = R.recall_identity(y, pred, cells)
    check(abs(ident['0']['recall_walk_weighted'] - 30 / 32) < 1e-12 and abs(ident['0']['recall_cell_balanced'] - 0.5) < 1e-12, 'identity example: R^W=30/32, R^O=0.5')
    check(abs(ident['0']['cov_term'] - 0.4375) < 1e-12 and abs(ident['0']['identity_residual']) < 1e-12, 'identity: covariance term 7/16 and zero residual')
    proba = np.zeros((42, 3))
    proba[np.arange(42), pred] = 0.6
    proba[:, :] += 0.2
    proba /= proba.sum(axis=1, keepdims=True)
    cp = R.cell_predictions(proba, cells)
    confC = R.cell_person_confusions(cells, cp['pred_mean']).sum(axis=0)
    check(confC.sum() == 3 and P.macro_f1_conf(confC) != fA, 'C: one count per cell, differs from walk grain')
    subj2 = np.array([0, 0, 0])
    y2 = np.array([2, 2, 2])
    proba2 = np.array([[0.0, 0.55, 0.45], [0.0, 0.55, 0.45], [0.0, 0.0, 1.0]])
    cells2 = R.build_cells(subj2, ['ON'] * 3, y2)
    cp2 = R.cell_predictions(proba2, cells2)
    check(cp2['pred_vote'][0] == 1 and cp2['pred_mean'][0] == 2, 'mean probability (grade 2) and majority vote (grade 1) can disagree')
    check(raises_source(lambda: R.cell_predictions(proba2[:, :2], cells2)), 'wrong probability shape is refused')

def test_bootstrap_carries_cells():
    rng = np.random.default_rng(3)
    subj = np.repeat(np.arange(6), [4, 4, 2, 2, 6, 6])
    state = []
    for p in range(6):
        n = int((subj == p).sum())
        state += ['ON'] * (n // 2) + ['OFF'] * (n - n // 2)
    y = rng.integers(0, 3, subj.size)
    cells0 = None
    cell_key = list(zip(subj.tolist(), state))
    uniq = sorted(set(cell_key))
    gmap = {k: int(rng.integers(0, 3)) for k in uniq}
    y = np.array([gmap[k] for k in cell_key])
    cells = R.build_cells(subj, state, y)
    pred = rng.integers(0, 3, subj.size)
    counts = R.draw_person_counts(6, 50, 11)
    check(counts.shape == (50, 6) and (counts.sum(axis=1) == 6).all(), 'person draws resample 6 persons with replacement')
    pc = R.person_confusions(y, pred, R.estimand_weights(cells, 'B_cell_reweighted'), subj, cells['persons'])
    check(np.allclose(pc.sum(axis=(1, 2)), cells['cells_per_person']), "per-person B-confusion mass equals that person's cell count (both states carried)")
    pooled = np.einsum('bs,sij->bij', counts, pc)
    check(np.allclose(pooled.sum(axis=(1, 2)), counts @ cells['cells_per_person']), 'a drawn person brings all its cells, k times if drawn k times')
    cc = R.cell_person_confusions(cells, rng.integers(0, 3, cells['n_cells']))
    check(np.allclose(cc.sum(axis=(1, 2)), cells['cells_per_person']), 'cell-grain confusion attributes one count per cell to its person')
    boot = R.boot_from_person_conf(pc, counts)
    check(boot['macro_f1'].shape == (50,) and np.isfinite(boot['macro_f1']).all(), 'bootstrap macro-F1 defined for every draw')
    q = rng.random(cells['n_cells'])
    rb = R.boot_cell_rank(q, cells['grade'], cells, counts)
    check(rb['spearman'].shape == (50,), 'cell rank bootstrap returns one value per draw')
    counts_zero = np.zeros((1, 6), int)
    counts_zero[0, 0] = 6
    one_person = R.boot_cell_rank(q, cells['grade'], cells, counts_zero)
    check(np.isnan(one_person['spearman'][0]) or cells['cells_per_person'][0] >= 2, 'a degenerate draw yields an undefined rank metric, not a fake number')
    s = R.summary(0.5, np.array([0.4, np.nan, 0.6, 0.5]))
    check(s['n_undefined_draws'] == 1 and s['ci95'] is not None, 'summary counts undefined draws and uses the rest')

def test_two_world_counterexample():
    """Same X in both worlds, labels differ on a set of mass p_delta: any fixed classifier errs >= p_delta/2 in one world."""
    rng = np.random.default_rng(5)
    g = rng.random(5000)
    x = g
    yA = (g > 0.4).astype(int) + (g > 0.8).astype(int)
    yB = (g > 0.5).astype(int) + (g > 0.8).astype(int)
    p_delta = float(((g > 0.4) & (g <= 0.5)).mean())
    worst = 1.0
    for cut in np.linspace(0.3, 0.6, 31):
        f = (x > cut).astype(int) + (x > 0.8).astype(int)
        worst = min(worst, max((f != yA).mean(), (f != yB).mean()))
    check(worst >= p_delta / 2 - 1e-09, f'two-world bound holds: best worst-case error {worst:.3f} >= p_delta/2 = {p_delta / 2:.3f}')
    check(np.array_equal(x, x) and (not np.array_equal(yA, yB)), 'the two worlds share X and differ only in labels (known obstruction, not a new theorem)')

def test_single_fold_matches_outer_run():
    objs = synth_objs('speed', seed=41)
    t = Q.build_table(objs)
    feats = Q.arm_features(t)
    ci = C.COHORTS.index('BMCLab')
    full = Q.outer_run(feats['S'], t['label'], t['cohort'], t['subject'], 'walk', FAST, [0])
    one = R.single_outer_fold(feats['S'], t['label'], t['cohort'], t['subject'], 'walk', FAST, [0], ci)
    check(one['selected'] == full[ci]['selected'], 'single fold selects the same config as outer_run')
    check(all((np.allclose(a, b) for a, b in zip(one['proba'], full[ci]['proba']))), 'single fold probabilities equal outer_run fold probabilities')
    check(np.array_equal(one['test_mask'], full[ci]['test_mask']), 'single fold test mask equals outer_run')
    check(not one['test_mask'][t['cohort'] != ci].any(), 'test mask holds only the held-out cohort')
    y = t['label'][one['test_mask']]
    pred = R.selected_proba(one).argmax(axis=1)
    pw = R.person_weighted_macro_f1(y, pred, t['subject'][one['test_mask']])
    scorer = Q.Scorer(t, 10)
    preds = [np.zeros(scorer.y[n].size, int) for n in scorer.names]
    preds[ci] = pred
    conf = scorer.conf(preds)
    want = scorer.stats(conf, 'macro_f1', person=True)[0][ci]
    check(abs(pw - want) < 1e-12, 'person-weighted macro-F1 replica equals P1 Scorer person=True')
    return (objs, t)

def test_end_to_end(objs, t):
    ci = C.COHORTS.index('BMCLab')
    med_site = A.collect_medication(objs['BMCLab'])
    check(len(med_site) == int((t['cohort'] == ci).sum()), 'medication rows align with the table rows of the site')
    check(set(med_site) == {'ON', 'OFF'}, 'fabricated medication strings normalise to ON/OFF')
    feats = Q.arm_features(t)
    folds = {arm: R.single_outer_fold(feats[arm], t['label'], t['cohort'], t['subject'], 'walk', FAST, [0], ci) for arm in A.ARMS}
    te = folds['S']['test_mask']
    site_rows = np.flatnonzero(t['cohort'] == ci)
    med = [med_site[i] for i in range(site_rows.size) if te[site_rows[i]]]
    y, subj = (t['label'][te], t['subject'][te])
    probas = {arm: R.selected_proba(folds[arm]) for arm in A.ARMS}
    body = A.analyze(y, subj, med, probas, n_boot=60, seed=7)
    s = body['bmclab_structure']
    check(s['n_people'] == 23 and s['n_walks'] == 781 and (s['n_cells'] == 42) and (s['n_two_state_people'] == 19) and (s['n_discordant_two_state_people'] == 8), 'fabricated BMCLab structure matches the frozen expectation')
    for arm in A.ARMS:
        e = body['estimands'][arm]
        check(abs(e['B_cell_reweighted']['total_weight'] - 42) < 1e-09 and abs(e['D_person_cell_balanced']['total_weight'] - 23) < 1e-09 and (e['A_walk']['total_weight'] == 781) and (e['C_cell_mean_proba']['total_weight'] == 42), f'{arm}: estimand masses are 781 / 42 / 42 / 23')
        check(e['A_walk']['macro_f1']['point'] > 0.5 if arm == 'S' else True, 'speed arm recovers the fabricated speed signal at walk grain')
    text = R.render({'x': body})
    check(MARK not in text and 'medication' not in text.lower().replace('medication session', ''), 'no fabricated identifier reaches the output')
    check(len(text.encode()) < 262144 // 3, f'output is far below the summary cap ({len(text)} bytes)')
    body2 = A.analyze(y, subj, med, probas, n_boot=60, seed=7)
    check(R.render(body) == R.render(body2), 'analysis is deterministic for a fixed seed')
    bad = list(med)
    bad[0] = 'unknown'
    check(raises_source(lambda: A.analyze(y, subj, bad, probas, n_boot=5, seed=1)), 'an unrecognised medication value stops the analysis')
    y_bad = y.copy()
    y_bad[0] = (y_bad[0] + 1) % 3
    check(raises_source(lambda: A.analyze(y_bad, subj, med, probas, n_boot=5, seed=1)), 'a mixed-grade cell stops the analysis')
    fake_p1 = {'selected_configs': {arm: {'BMCLab': {'all': Q.CONFIGS[folds[arm]['selected']['all']][0]}} for arm in A.ARMS}, 'arms_inductive_strong': {arm: {k: {'per_cohort': {'BMCLab': {'point': 0.123456}}} for k in ('macro_f1', 'ordinal_mae', 'macro_f1_person_weighted_eval')} for arm in A.ARMS}}
    check(raises_source(lambda: A.check_parity(t, folds, fake_p1, te)), 'parity check refuses mismatching archived values')
    good_p1 = {'selected_configs': fake_p1['selected_configs'], 'arms_inductive_strong': {}}
    for arm in A.ARMS:
        pred = probas[arm].argmax(axis=1)
        good_p1['arms_inductive_strong'][arm] = {'macro_f1': {'per_cohort': {'BMCLab': {'point': Q.walk_macro_f1(y, pred)}}}, 'ordinal_mae': {'per_cohort': {'BMCLab': {'point': float(Q.ordinal_mae_conf(R.weighted_confusion(y, pred, np.ones(y.size))))}}}, 'macro_f1_person_weighted_eval': {'per_cohort': {'BMCLab': {'point': R.person_weighted_macro_f1(y, pred, subj)}}}}
    par = A.check_parity(t, folds, good_p1, te)
    check(all((par[arm]['refit']['macro_f1'] == par[arm]['archived']['macro_f1'] for arm in A.ARMS)), 'parity check passes on matching archived values')
    wrong_cfg = json.loads(json.dumps(good_p1))
    wrong_cfg['selected_configs']['S']['BMCLab']['all'] = 'NOT_A_CONFIG'
    check(raises_source(lambda: A.check_parity(t, folds, wrong_cfg, te)), 'parity check refuses a different selected config')

def main():
    test_cells_and_weights()
    test_estimands_differ()
    test_bootstrap_carries_cells()
    test_two_world_counterexample()
    objs, t = test_single_fold_matches_outer_run()
    test_end_to_end(objs, t)
    if FAILS:
        print(f'R8_A_SYNTHETIC_FAIL {len(FAILS)}: ' + '; '.join(FAILS))
        return 1
    print('R8_A_SYNTHETIC_PASS all fabricated checks passed (no real data read)')
    return 0
if __name__ == '__main__':
    sys.exit(main())
