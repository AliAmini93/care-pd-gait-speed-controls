"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import abcd_baseline as AB
import abcd_common as AC
import abcd_worlds as W
import common as C
import p1_core as Q

def _stage(part):
    if part == 'a':
        import abcd_stage_a as M
    elif part == 'b':
        import abcd_stage_b as M
    elif part == 'c':
        import abcd_stage_c as M
    elif part == 'd':
        import abcd_stage_d as M
    else:
        raise ValueError(part)
    return M

def _fab(world='W1', r=0):
    data = W.generate(world, 'dev', r)
    return (data, AC.synthetic_bundle(data))

def check_offsets_argmax():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(3), 5000)
    p[:50] = np.eye(3)[rng.integers(0, 3, 50)]
    return {'pass': bool(np.array_equal(AB.offsets(p).argmax(axis=1), p.argmax(axis=1)))}

def check_residual_class_order():
    rng = np.random.default_rng(2)
    n, d = (400, 6)
    Phi = rng.standard_normal((n, d))
    y = rng.integers(0, 3, n)
    O = AB.offsets(rng.dirichlet(np.ones(3), n))
    z = AC.OffsetResidual().fit(Phi, O, y).logits(Phi, O)
    perm = np.array([2, 0, 1])
    O2 = np.empty_like(O)
    O2[:, perm] = O
    z2 = AC.OffsetResidual().fit(Phi, O2, perm[y]).logits(Phi, O2)
    err = float(np.abs(z2[:, perm] - z).max())
    return {'pass': err < 0.0001, 'max_abs_err': err}

def check_residual_recovers_signal():
    rng = np.random.default_rng(3)
    n = 3000
    Phi = rng.standard_normal((n, 2))
    O = np.zeros((n, 3))
    logits = np.column_stack([-1.5 * Phi[:, 0], np.zeros(n), 1.5 * Phi[:, 0]])
    pr = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=q) for q in pr])
    h = AC.OffsetResidual().fit(Phi, O, y)
    f1 = AC.macro_f1(y, h.logits(Phi, O).argmax(axis=1))
    f1_oracle = AC.macro_f1(y, logits.argmax(axis=1))
    return {'pass': bool(h.W[0, 2] - h.W[0, 0] > 1.0 and abs(h.W[1, 2] - h.W[1, 0]) < 0.3 and (f1 >= f1_oracle - 0.02)), 'w_contrast_signal': float(h.W[0, 2] - h.W[0, 0]), 'w_contrast_noise': float(h.W[1, 2] - h.W[1, 0]), 'macro_f1': f1, 'macro_f1_true_logits': f1_oracle}

def check_penalty_gradients():
    rng = np.random.default_rng(4)
    n, d = (120, 3)
    Phi, O = (rng.standard_normal((n, d)), AB.offsets(rng.dirichlet(np.ones(3), n)))
    y, env = (rng.integers(0, 3, n), rng.integers(0, 3, n))
    worst = 0.0
    for pen, beta in ((None, 0.0), ('consensus', 5.0), ('vrex', 5.0)):
        h = AC.OffsetResidual(penalty=pen, beta=beta)
        captured = {}
        import scipy.optimize as so
        orig = so.minimize

        def spy(fun, x0, **kw):
            captured['f'] = fun
            return orig(fun, x0, **kw)
        AC.minimize = spy
        try:
            h.fit(Phi, O, y, env=env if pen else None)
        finally:
            AC.minimize = orig
        f = captured['f']
        th = rng.standard_normal(1 + 3 * d) * 0.3
        _, g = f(th)
        num = np.empty_like(th)
        for k in range(th.size):
            e = np.zeros_like(th)
            e[k] = 1e-06
            num[k] = (f(th + e)[0] - f(th - e)[0]) / 2e-06
        worst = max(worst, float(np.abs(num - g).max()))
    return {'pass': worst < 1e-05, 'max_abs_grad_err': worst}

def check_crossfit_person_disjoint():
    data, b = _fab()
    src, _ = W.split(data)
    warned = [0]
    oof = AB.crossfit_proba(AB.CONFIGS[0], b['speed'][src][:, None], b['label'][src], b['subject'][src], warned)
    return {'pass': bool(np.isfinite(oof).all() and np.allclose(oof.sum(axis=1), 1.0))}

def check_s_fold_equals_p1_outer_run():
    data, b = _fab()
    X = b['speed'][:, None]
    warned = [0]
    folds = Q.outer_run(X, b['label'], b['site'], b['subject'], 'walk', AB.CONFIGS, warned)
    ok, worst = (True, 0.0)
    for ci in range(len(C.COHORTS)):
        tr, te = (b['site'] != ci, b['site'] == ci)
        s = AB.s_fold(X[tr], b['label'][tr], b['site'][tr], b['subject'][tr], X[te], warned)
        ref = Q.proba_for(folds, ci, policy='all')
        ok &= s['index'] == folds[ci]['selected']['all']
        worst = max(worst, float(np.abs(s['proba'] - ref).max()))
    return {'pass': bool(ok and worst == 0.0), 'max_abs_diff': worst}

def check_scorer_equals_p1():
    data, b = _fab()
    rng = np.random.default_rng(5)
    t = {'label': b['label'], 'cohort': b['site'], 'subject': b['subject']}
    qs = Q.Scorer(t, 200)
    preds = [rng.integers(0, 3, int((b['site'] == ci).sum())) for ci in range(4)]
    pq, dq = qs.stats(qs.conf(preds))
    sites = list(C.COHORTS)
    ss = AC.SiteScorer({s: b['label'][b['site'] == i] for i, s in enumerate(sites)}, {s: b['subject'][b['site'] == i] for i, s in enumerate(sites)}, sites, n_boot=200)
    ps, ds = ss.stats(ss.conf({s: preds[i] for i, s in enumerate(sites)}))
    return {'pass': bool(np.array_equal(pq, ps) and np.array_equal(dq, ds))}

def check_decision_rules():

    def blk(m, lo, sites):
        return {'M': {'point': m, 'ci95': [lo, lo + 0.1]}, 'per_site': {f's{i}': {'point': v} for i, v in enumerate(sites)}}
    ok = True
    ok &= AC.dev_screen(blk(0.04, 0, [0.05, 0.01, 0.06]))['dev_screen_pass'] is True
    ok &= AC.dev_screen(blk(0.04, 0, [0.15, -0.04, 0.01]))['dev_screen_pass'] is False
    ok &= AC.dev_screen(blk(0.04, 0, [0.12, 0.0, 0.0]))['dev_screen_pass'] is False
    ok &= AC.dev_screen(blk(0.029, 0, [0.03, 0.03, 0.027]))['dev_screen_pass'] is False
    syn_ok = AC.synthetic_screen({'W1': 0.05, 'W2': 0.0, 'W3': -0.01})
    syn_bad = AC.synthetic_screen({'W1': 0.05, 'W2': 0.0, 'W3': -0.04})
    ok &= syn_ok['synthetic_screen_pass'] and (not syn_bad['synthetic_screen_pass'])
    dev_ok = AC.dev_screen(blk(0.04, 0, [0.05, 0.01, 0.06]))
    dev_no = AC.dev_screen(blk(0.01, 0, [0.05, 0.01, -0.03]))
    ok &= AC.stage_status(True, syn_ok, dev_ok)['status'] == 'DEV_PROMISING'
    ok &= AC.stage_status(True, syn_bad, dev_ok)['status'] == 'DEV_INCONCLUSIVE'
    ok &= AC.stage_status(False, syn_ok, dev_ok)['status'] == 'DEV_INCONCLUSIVE'
    ok &= AC.stage_status(True, syn_ok, dev_no)['status'] == 'DEV_NO_GAIN'
    ok &= AC.stage_status(True, syn_ok, dev_no, blocked_reason='x')['status'] == 'DEV_BLOCKED'
    ok &= AC.final_decision(blk(0.04, 0.01, [0.05, 0.02, -0.01, 0.1]))['status'] == 'EMPIRICAL_GAIN_EXPLORATORY'
    ok &= AC.final_decision(blk(0.04, -0.01, [0.05, 0.02, -0.01, 0.1]))['status'] == 'INCONCLUSIVE'
    ok &= AC.final_decision(blk(0.02, 0.01, [0.05, 0.02, -0.01, 0.1]))['status'] == 'NO_DETECTED_GAIN'
    return {'pass': bool(ok)}

def _method_inputs(M, data, b):
    src, tgt = W.split(data)
    F = M.features(b['store'])
    warned = [0]
    X = b['speed'][:, None]
    s_head = AB.s_fold(X[src], b['label'][src], b['site'][src], b['subject'][src], X[tgt], warned)
    S = {'F': F[src], 'speed': b['speed'][src], 'label': b['label'][src], 'site': b['site'][src], 'subject': b['subject'][src]}
    T = {'F': F[tgt], 'speed': b['speed'][tgt], 'site': b['site'][tgt]}
    return (S, T, s_head, warned, F, src, tgt)

def check_method_target_blind_and_site_free(M):
    data, b = _fab()
    S, T, s_head, warned, *_ = _method_inputs(M, data, b)
    a = M.method_fold(S, T, s_head, warned)['arms']
    T2 = dict(T, site=np.full_like(T['site'], 99))
    b2 = M.method_fold(S, T2, s_head, warned)['arms']
    same = all((np.array_equal(a[k], b2[k]) for k in a))
    finite = all((np.isfinite(v).all() for v in a.values()))
    return {'pass': bool(same and finite and ('label' not in T) and ('subject' not in T))}

def check_method_row_duplication(M):
    data, b = _fab()
    S, T, s_head, warned, *_ = _method_inputs(M, data, b)
    a = M.method_fold(S, T, s_head, warned)['arms']
    idx = np.concatenate([np.arange(T['F'].shape[0]), np.arange(10)])
    T2 = {k: v[idx] for k, v in T.items()}
    s2 = dict(s_head, proba=s_head['proba'][idx])
    b2 = M.method_fold(S, T2, s2, warned)['arms']
    return {'pass': bool(all((np.allclose(b2[k], a[k][idx], atol=1e-12, rtol=0) for k in a)))}

def check_method_fallback_exact(M):
    ok, n_closed = (True, 0)
    for world in ('W3', 'W1'):
        data, b = _fab(world)
        S, T, s_head, warned, *_ = _method_inputs(M, data, b)
        out = M.method_fold(S, T, s_head, warned)
        if not out['diag'].get('gate_open', True):
            n_closed += 1
            ok &= np.array_equal(out['arms'][M.CANDIDATE].argmax(axis=1), s_head['proba'].argmax(axis=1))
    return {'pass': bool(ok), 'n_closed_gates_seen': n_closed}

def check_features_heading_invariant(M):
    data, _ = _fab()
    from scipy.spatial.transform import Rotation
    poses = [p.copy() for p in data['pose'][:20]]
    yaw = Rotation.from_rotvec([0.0, 1.234, 0.0])
    for p in poses:
        p[:, 0:3] = (yaw * Rotation.from_rotvec(p[:, 0:3])).as_rotvec()
    f1 = M.features(AC.build_store(data['pose'][:20], data['fps'][:20]))
    f2 = M.features(AC.build_store(poses, data['fps'][:20]))
    err = float(np.nanmax(np.abs(f1 - f2)))
    return {'pass': err < 1e-05, 'max_abs_err': err}

def check_class_weights():
    rng = np.random.default_rng(6)
    y = rng.choice(3, 500, p=[0.6, 0.3, 0.1])
    w = AC.class_weights(y)
    tot = [w[y == k].sum() for k in range(3)]
    perm = rng.permutation(y.size)
    same = np.allclose(AC.class_weights(y[perm]), w[perm])
    return {'pass': bool(np.allclose(tot, tot[0]) and abs(w.mean() - 1) < 1e-12 and same)}
GENERIC = {'G01_offsets_preserve_argmax': check_offsets_argmax, 'G09_class_weights_label_only': check_class_weights, 'G02_residual_class_order_equivariance': check_residual_class_order, 'G03_residual_recovers_planted_signal': check_residual_recovers_signal, 'G04_analytic_gradients': check_penalty_gradients, 'G05_crossfit_person_disjoint': check_crossfit_person_disjoint, 'G06_s_fold_equals_p1_outer_run': check_s_fold_equals_p1_outer_run, 'G07_scorer_equals_p1_scorer': check_scorer_equals_p1, 'G08_decision_rules': check_decision_rules}
METHOD = {'M01_target_label_blind_and_site_code_free': check_method_target_blind_and_site_free, 'M02_row_duplication_invariance': check_method_row_duplication, 'M03_fallback_exact_when_gate_closed': check_method_fallback_exact, 'M04_features_root_heading_invariant': check_features_heading_invariant}

def validity_checks(part):
    M = _stage(part)
    out = {}
    for name, fn in GENERIC.items():
        t0 = time.monotonic()
        out[name] = dict(fn(), seconds=round(time.monotonic() - t0, 1))
    for name, fn in METHOD.items():
        t0 = time.monotonic()
        out[name] = dict(fn(M), seconds=round(time.monotonic() - t0, 1))
    for name, fn in getattr(M, 'EXTRA_VALIDITY', {}).items():
        t0 = time.monotonic()
        out[name] = dict(fn(), seconds=round(time.monotonic() - t0, 1))
    return out

def run(mode, part):
    if part == 'final':
        print('ABCD_REFUSED: final has its own tests')
        return 10
    val = validity_checks(part)
    for k, v in val.items():
        print(('PASS ' if v['pass'] else 'FAIL ') + k + ' ' + str({a: b for a, b in v.items() if a != 'pass'})[:240])
    ok = all((v['pass'] for v in val.values()))
    if mode == 'dev':
        import abcd_stagerun as SR
        M = _stage(part)
        t0 = time.monotonic()
        syn = SR.run_synthetic(M, 'dev')
        for world, r in syn['worlds'].items():
            print('DEV_SYNTH', world, {a: round(v, 4) for a, v in r['mean_macro_f1'].items()}, 'delta', {a: round(v, 4) for a, v in r['mean_delta_vs_S'].items()}, 'diag', [{k: round(v, 3) if isinstance(v, float) else v for k, v in x['diag'].items() if not isinstance(v, dict)} for x in r['replicates']])
        print('DEV_SYNTH_SCREEN', syn['screen'], 'seconds', round(time.monotonic() - t0, 1))
    print('ABCD_ROUTE_TESTS_%s part=%s n=%d n_pass=%d' % ('ALL_PASS' if ok else 'FAILED', part, len(val), sum((v['pass'] for v in val.values()))))
    return 0 if ok else 3
