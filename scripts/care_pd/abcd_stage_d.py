"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
import abcd_baseline as AB
import abcd_common as AC
import abcd_stage_b as SB
import abcd_stage_c as SC
import p1_core as Q
STAGE = 'D'
CANDIDATE = 'S_plus_D'
CONTROLS = ('ERM_RES_D',)
SYNTH_FALSIFIERS = ()
SHRINK_FOLDS = 5
CFG = {'features': 'stage-B phase/symmetry features (16) via abcd_stage_c.prepare (speed-conditional scaler, 0.05 floor)', 'weights': "P1 'person' scheme: each person equal total weight, then class balance (training labels only)", 'direction': 'person-weighted offset residual, ridge lambda 0.03, no intercept, tau learned', 'shrinkage': 'c* in [0,1] minimising pooled held-out-person person-weighted CE over 5 person-grouped folds; bounded scalar search (xatol 1e-4) plus explicit endpoint comparison', 'candidate': 'argmax(tau*o + c* phi W); c* = 0 is exactly S', 'control': 'ERM_RES_D: c = 1', 'ridge_lambda': AC.RIDGE_LAMBDA, 'shrink_folds': SHRINK_FOLDS, 'n_parameters_head': 1 + SB.N_FEATURES * 3, 'n_selected_scalars': 1}
features = SB.features

def person_weights(y, subj):
    w = Q.train_weights(np.asarray(y).astype(int), np.asarray(subj), 'person')
    return w / w.mean()

def fit_direction(phi, O, y, subj):
    return AC.OffsetResidual().fit(phi, O, y, w=person_weights(y, subj))

def _wce(Z, y, w):
    Z = Z - Z.max(axis=1, keepdims=True)
    lp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    return float(-(w * lp[np.arange(y.size), y]).sum())

def shrinkage(phi, O, y, subj):
    w_all = person_weights(y, subj)
    parts = []
    for tr, va in GroupKFold(n_splits=SHRINK_FOLDS).split(phi, y, subj):
        if np.intersect1d(subj[tr], subj[va]).size:
            raise AC.AbcdError('ABCD_D_PERSON_OVERLAP')
        h = fit_direction(phi[tr], O[tr], y[tr], subj[tr])
        parts.append((va, h.tau * O[va], phi[va] @ h.W))
    total = w_all.sum()

    def ce(c):
        return sum((_wce(base + c * delta, y[va], w_all[va]) for va, base, delta in parts)) / total
    res = minimize_scalar(ce, bounds=(0.0, 1.0), method='bounded', options={'xatol': 0.0001})
    cands = [(ce(0.0), 0.0), (ce(float(res.x)), float(res.x)), (ce(1.0), 1.0)]
    best = min(cands, key=lambda t: (t[0], t[1]))
    return (best[1], {'ce_c0': cands[0][0], 'ce_c_star': best[0], 'ce_c1': cands[2][0]})

def method_fold(src, tgt, s_head, warned):
    config, phi_s, phi_t, O_s, o_tgt, oof = SC.prepare(src, tgt, s_head, warned)
    y, subj = (src['label'], src['subject'])
    head = fit_direction(phi_s, O_s, y, subj)
    c, curve = shrinkage(phi_s, O_s, y, subj)
    delta_t = phi_t @ head.W
    z_d = head.tau * o_tgt + c * delta_t
    diag = {'gate_open': bool(c > 0.0), 'c_star': c, 'held_out_person_ce': curve, 'tau': head.tau, 'w_frobenius': head.diag['w_frobenius'], 'converged': head.diag['converged'], 'in_sample_ce_gain_by_env_person_weighted_direction': list(SC.env_gains(head, phi_s, O_s, y, src['site']).values()), 'offset_entropy_ratio_target_over_source': AB.entropy(s_head['proba']) / max(AB.entropy(oof), 1e-12), 'S_config': config[0]}
    return {'arms': {'S_plus_D': z_d, 'ERM_RES_D': head.tau * o_tgt + delta_t}, 'diag': diag}

def check_person_memorisation_counterexample():
    rng = np.random.default_rng(21)
    n_p, n_w = (40, 20)
    subj = np.repeat(np.arange(n_p), n_w)
    y = np.repeat(rng.integers(0, 3, n_p), n_w)
    code = rng.standard_normal((n_p, 8))
    phi = code[subj] + 0.05 * rng.standard_normal((subj.size, 8))
    O = np.zeros((subj.size, 3))
    walk = AC.OffsetResidual().fit(phi, O, y)
    in_sample = AC.macro_f1(y, walk.logits(phi, O).argmax(axis=1)) - AC.macro_f1(y, O.argmax(axis=1))
    c, curve = shrinkage(phi, O, y, subj)
    return {'pass': bool(in_sample > 0.3 and c <= 0.1), 'walk_level_in_sample_macro_f1_gain': in_sample, 'c_star': c, **curve}

def check_planted_generalisable_signal():
    rng = np.random.default_rng(22)
    n_p, n_w = (60, 10)
    subj = np.repeat(np.arange(n_p), n_w)
    sev = np.repeat(rng.standard_normal(n_p), n_w)
    y = np.digitize(sev + 0.3 * rng.standard_normal(subj.size), [-0.6, 0.6])
    phi = np.column_stack([sev + 0.5 * rng.standard_normal(subj.size), rng.standard_normal(subj.size)])
    O = np.zeros((subj.size, 3))
    c, curve = shrinkage(phi, O, y, subj)
    return {'pass': bool(c >= 0.5), 'c_star': c, **curve}

def check_person_weights():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
    subj = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4])
    w = person_weights(y, subj)
    per_class = [w[y == k].sum() for k in range(3)]
    ok = np.allclose(per_class, per_class[0]) and np.isclose(w[0] * 3, w[3] * 1) and np.isclose(w[6] * 4, w[10] * 2)
    return {'pass': bool(ok)}
EXTRA_VALIDITY = {'D01_person_memorisation_counterexample_shrinks_to_S': check_person_memorisation_counterexample, 'D02_planted_generalisable_signal_kept': check_planted_generalisable_signal, 'D03_person_weights_equal_person_then_class': check_person_weights, 'B01_phase_observable_on_W1_truth': SB.check_phase_observable_w1}
