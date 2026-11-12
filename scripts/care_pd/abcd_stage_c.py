"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
import abcd_baseline as AB
import abcd_common as AC
import abcd_stage_a as SA
import abcd_stage_b as SB
STAGE = 'C'
CANDIDATE = 'S_plus_C'
CONTROLS = ('ERM_RES_C', 'VREX_RES_C')
SYNTH_FALSIFIERS = ()
BETA = 100.0
FALLBACK_FOLDS = 5
MIN_PERSONS_PER_GRADE = 3
MIN_WALKS_PER_GRADE = 20
CFG = {'features': 'stage-B phase/symmetry features (16), imported unchanged from abcd_stage_b', 'objective': 'pooled class-weighted CE + beta * Var_e(R_e(offset only, tau0) - R_e(theta)) + lambda/2 ||W||^2', 'beta': BETA, 'ridge_lambda': AC.RIDGE_LAMBDA, 'conditioning_speed_floor': SA.COND_SPEED_FLOOR, 'env_risk': 'class-weighted mean cross-entropy within each source environment (weights from all training labels), equal environment weight in the penalty', 'fallback': "person-grouped 5-fold refits of the consensus head on the pooled source; open iff every source environment's held-out-person macro-F1 change (argmax z vs argmax o) is > 0", 'comparator': 'VREX_RES_C: beta * Var_e R_e(theta), same beta, features and head, no fallback', 'control': 'ERM_RES_C: beta = 0, no fallback (identical to ERM_RES_B by construction)', 'support_rule': {'min_persons_per_grade': MIN_PERSONS_PER_GRADE, 'min_walks_per_grade': MIN_WALKS_PER_GRADE, 'checked': 'per development cohort before any score; DEV_BLOCKED if violated'}, 'observability_rule': 'stage-B rule re-applied (share < 0.5 in any cohort -> DEV_BLOCKED)', 'n_parameters_head': 1 + SB.N_FEATURES * 3}
features = SB.features

def prepare(src, tgt, s_head, warned):
    config = AB.CONFIGS[s_head['index']]
    c_src = np.maximum(src['speed'], SA.COND_SPEED_FLOOR)
    c_tgt = np.maximum(tgt['speed'], SA.COND_SPEED_FLOOR)
    scaler = AC.SpeedConditionalScaler().fit(src['F'], c_src)
    phi_s, phi_t = (scaler.transform(src['F'], c_src), scaler.transform(tgt['F'], c_tgt))
    oof = AB.crossfit_proba(config, src['speed'][:, None], src['label'], src['subject'], warned)
    return (config, phi_s, phi_t, AB.offsets(oof), AB.offsets(s_head['proba']), oof)

def env_gains(head, phi, O, y, env):
    w = AC.class_weights(y)
    Y = AC.one_hot(y)

    def risks(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        lp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
        ce = -(lp * Y).sum(axis=1)
        return {int(e): float((w[env == e] * ce[env == e]).sum() / w[env == e].sum()) for e in np.unique(env)}
    r0, r1 = (risks(O), risks(head.logits(phi, O)))
    return {e: r0[e] - r1[e] for e in r0}

def consensus_fallback(phi, O, y, env, subj):
    zhat = np.full((y.size, 3), np.nan)
    for tr, va in GroupKFold(n_splits=FALLBACK_FOLDS).split(phi, y, subj):
        h = AC.OffsetResidual(penalty='consensus', beta=BETA).fit(phi[tr], O[tr], y[tr], env=env[tr])
        zhat[va] = h.logits(phi[va], O[va])
    gains = {}
    for e in np.unique(env):
        m = env == e
        gains[int(e)] = AC.macro_f1(y[m], zhat[m].argmax(axis=1)) - AC.macro_f1(y[m], O[m].argmax(axis=1))
    return (bool(min(gains.values()) > 0), gains)

def method_fold(src, tgt, s_head, warned):
    config, phi_s, phi_t, O_s, o_tgt, oof = prepare(src, tgt, s_head, warned)
    y, env = (src['label'], src['site'])
    erm = AC.OffsetResidual().fit(phi_s, O_s, y)
    vrex = AC.OffsetResidual(penalty='vrex', beta=BETA).fit(phi_s, O_s, y, env=env)
    cons = AC.OffsetResidual(penalty='consensus', beta=BETA).fit(phi_s, O_s, y, env=env)
    is_open, held_out_gains = consensus_fallback(phi_s, O_s, y, env, src['subject'])
    z_cons = cons.logits(phi_t, o_tgt)
    diag = {'gate_open': is_open, 'fallback_held_out_macro_f1_gain_by_env': list(held_out_gains.values()), 'in_sample_ce_gain_by_env': {k: list(env_gains(h, phi_s, O_s, y, env).values()) for k, h in (('erm', erm), ('vrex', vrex), ('consensus', cons))}, 'w_frobenius': {'erm': erm.diag['w_frobenius'], 'vrex': vrex.diag['w_frobenius'], 'consensus': cons.diag['w_frobenius']}, 'tau': {'erm': erm.tau, 'vrex': vrex.tau, 'consensus': cons.tau}, 'converged': {'erm': erm.diag['converged'], 'vrex': vrex.diag['converged'], 'consensus': cons.diag['converged']}, 'n_source_envs': int(np.unique(env).size), 'offset_entropy_ratio_target_over_source': AB.entropy(s_head['proba']) / max(AB.entropy(oof), 1e-12), 'S_config': config[0]}
    return {'arms': {'S_plus_C': z_cons if is_open else o_tgt, 'ERM_RES_C': erm.logits(phi_t, o_tgt), 'VREX_RES_C': vrex.logits(phi_t, o_tgt)}, 'diag': diag}

def _hetero(seed):
    rng = np.random.default_rng(seed)
    n = 1200
    env = np.repeat([0, 1], n // 2)
    phi = rng.standard_normal((n, 2))
    coef1 = np.where(env == 0, 1.2, -0.4)
    lat = 1.2 * phi[:, 0] + coef1 * phi[:, 1] + rng.standard_normal(n)
    y = np.digitize(lat, [-0.8, 0.8])
    return (phi, np.zeros((n, 3)), y, env)

def check_consensus_downweights_heterogeneous_direction():
    phi, O, y, env = _hetero(11)
    erm = AC.OffsetResidual().fit(phi, O, y)
    cons = AC.OffsetResidual(penalty='consensus', beta=BETA).fit(phi, O, y, env=env)

    def contrast(h, j):
        return float(abs(h.W[j, 2] - h.W[j, 0]))
    g_e, g_c = (env_gains(erm, phi, O, y, env), env_gains(cons, phi, O, y, env))
    var_e, var_c = (float(np.var(list(g_e.values()))), float(np.var(list(g_c.values()))))
    ok = contrast(cons, 1) < contrast(erm, 1) and contrast(cons, 0) > 0.5 and (var_c < var_e)
    return {'pass': bool(ok), 'erm_hetero_contrast': contrast(erm, 1), 'cons_hetero_contrast': contrast(cons, 1), 'cons_stable_contrast': contrast(cons, 0), 'gain_var_erm': var_e, 'gain_var_cons': var_c}

def _objective(head, phi, O, y, env):
    import scipy.optimize as so
    captured = {}

    def spy(fun, x0, **kw):
        captured['f'] = fun
        return so.minimize(fun, x0, **kw)
    AC.minimize = spy
    try:
        head.fit(phi, O, y, env=env)
    finally:
        AC.minimize = so.minimize
    return captured['f']

def check_consensus_equals_vrex_iff_equal_baseline_risk():
    rng = np.random.default_rng(12)
    n = 900
    env = np.repeat([0, 1, 2], n // 3)
    y = np.tile(np.repeat([0, 1, 2], n // 9), 3)
    phi = rng.standard_normal((n, 2)) + 0.5 * y[:, None]
    th = rng.standard_normal(1 + 6) * 0.3
    fc = _objective(AC.OffsetResidual(penalty='consensus', beta=BETA), phi, np.zeros((n, 3)), y, env)
    fv = _objective(AC.OffsetResidual(penalty='vrex', beta=BETA), phi, np.zeros((n, 3)), y, env)
    same = abs(fc(th)[0] - fv(th)[0])
    O = np.zeros((n, 3))
    O[env == 0] = AB.offsets(0.15 + 0.55 * np.eye(3)[y[env == 0]])
    fc2 = _objective(AC.OffsetResidual(penalty='consensus', beta=BETA), phi, O, y, env)
    fv2 = _objective(AC.OffsetResidual(penalty='vrex', beta=BETA), phi, O, y, env)
    diff = abs(fc2(th)[0] - fv2(th)[0])
    return {'pass': bool(same < 1e-10 and diff > 0.001), 'equal_baseline_abs_diff': same, 'unequal_baseline_abs_diff': diff}
EXTRA_VALIDITY = {'C01_consensus_downweights_env_heterogeneous_direction': check_consensus_downweights_heterogeneous_direction, 'C02_consensus_equals_vrex_iff_equal_baseline_risk': check_consensus_equals_vrex_iff_equal_baseline_risk, 'B01_phase_observable_on_W1_truth': SB.check_phase_observable_w1}

def support(b):
    out, ok = ({}, True)
    for s in AC.DEV_SITES:
        m = b['site'] == AC.C.COHORTS.index(s)
        persons = [int(np.unique(b['subject'][m & (b['label'] == k)]).size) for k in range(3)]
        walks = [int((m & (b['label'] == k)).sum()) for k in range(3)]
        good = min(persons) >= MIN_PERSONS_PER_GRADE and min(walks) >= MIN_WALKS_PER_GRADE
        out[s] = {'support_pass': bool(good), 'min_persons_any_grade_ge_3': bool(min(persons) >= 3), 'min_walks_any_grade_ge_20': bool(min(walks) >= 20)}
        ok &= good
    return (ok, out)
