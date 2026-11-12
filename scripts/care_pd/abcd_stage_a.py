"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import abcd_baseline as AB
import abcd_common as AC
STAGE = 'A'
CANDIDATE = 'S_plus_A'
CONTROLS = ('ERM_RES_A',)
SYNTH_FALSIFIERS = ('FX_UNIT',)
TARGET_FPS = 30.0
FEATURE_EPS = 0.001
GATE_MEAN_GT = 0.0
GATE_MIN_GT = -0.03
COND_SPEED_FLOOR = 0.05
CFG = {'features': "per local SMPL joint 1..23 of the 30-fps store: log(RMS geodesic deviation from the walk's chordal-mean rotation + 1e-3) and log(mean geodesic angular rate in rad/s + 1e-3)", 'n_features': 46, 'joints': '23 local joints; the root (pelvis global orientation) is excluded because it carries heading and path curvature', 'speed_conditioning': 'source-only OLS on [1, log max(speed, 0.05)] per column, residual/SD; the 0.05 floor applies to the conditioning covariate only (S uses raw speed)', 'conditioning_speed_floor': COND_SPEED_FLOOR, 'head': 'z = tau*o + phi W, no intercept, ridge lambda on W', 'ridge_lambda': AC.RIDGE_LAMBDA, 'log_tau_bounds': list(AC.LOG_TAU_BOUNDS), 'offset_eps': AB.OFFSET_EPS, 'crossfit': 'person-grouped GroupKFold', 'crossfit_folds': AB.CROSSFIT_FOLDS, 'class_weights': "walk class-balanced from training labels (P1 'walk')", 'gate': {'rule': 'open iff mean inner delta > 0 and min inner delta > -0.03', 'mean_gt': GATE_MEAN_GT, 'min_gt': GATE_MIN_GT, 'inner': 'leave-one-source-site-out, everything refit on inner sources, same S* config'}, 'n_parameters_head': 1 + 46 * 3, 'optimizer': 'scipy L-BFGS-B, maxiter 1000, gtol 1e-7, zero init'}

def features(store):
    F = np.empty((len(store), 46))
    for i in range(len(store)):
        R = AC.local_matrices(store, i)
        dev = AC.geodesic_angle(AC.chordal_mean(R)[None], R)
        amp = np.sqrt((dev ** 2).mean(axis=0))
        rate = AC.geodesic_angle(R[:-1], R[1:]).mean(axis=0) * TARGET_FPS if R.shape[0] > 1 else np.zeros(23)
        F[i, :23] = np.log(amp + FEATURE_EPS)
        F[i, 23:] = np.log(rate + FEATURE_EPS)
    if not np.isfinite(F).all():
        raise AC.AbcdError('ABCD_A_FEATURE_NONFINITE')
    return F

def residual_logits(F_src, s_src, y_src, subj_src, config, F_tgt, s_tgt, o_tgt, warned, lam=AC.RIDGE_LAMBDA):
    c_src, c_tgt = (np.maximum(s_src, COND_SPEED_FLOOR), np.maximum(s_tgt, COND_SPEED_FLOOR))
    scaler = AC.SpeedConditionalScaler().fit(F_src, c_src)
    phi_s, phi_t = (scaler.transform(F_src, c_src), scaler.transform(F_tgt, c_tgt))
    oof = AB.crossfit_proba(config, s_src[:, None], y_src, subj_src, warned)
    head = AC.OffsetResidual(lam=lam).fit(phi_s, AB.offsets(oof), y_src)
    return (head.logits(phi_t, o_tgt), head, oof)

def gate(src, config, warned):
    F, s, y, site, subj = (src['F'], src['speed'], src['label'], src['site'], src['subject'])
    deltas = {}
    for v in np.unique(site):
        u, w = (site != v, site == v)
        p_v = AB.fit_proba(config, s[u][:, None], y[u], subj[u], s[w][:, None], warned)
        z_v, _, _ = residual_logits(F[u], s[u], y[u], subj[u], config, F[w], s[w], AB.offsets(p_v), warned)
        deltas[int(v)] = AC.macro_f1(y[w], z_v.argmax(axis=1)) - AC.macro_f1(y[w], p_v.argmax(axis=1))
    vals = np.array(list(deltas.values()))
    is_open = bool(vals.mean() > GATE_MEAN_GT and vals.min() > GATE_MIN_GT)
    return (is_open, vals)

def method_fold(src, tgt, s_head, warned):
    config = AB.CONFIGS[s_head['index']]
    o_tgt = AB.offsets(s_head['proba'])
    z_tgt, head, oof = residual_logits(src['F'], src['speed'], src['label'], src['subject'], config, tgt['F'], tgt['speed'], o_tgt, warned)
    is_open, inner = gate(src, config, warned)
    z_a = z_tgt if is_open else o_tgt
    diag = {'gate_open': is_open, 'inner_deltas': inner.tolist(), 'inner_mean_delta': float(inner.mean()), 'tau': head.tau, 'w_frobenius': head.diag['w_frobenius'], 'converged': head.diag['converged'], 'offset_entropy_source_oof': AB.entropy(oof), 'offset_entropy_target': AB.entropy(s_head['proba']), 'S_config': config[0]}
    diag['offset_entropy_ratio_target_over_source'] = diag['offset_entropy_target'] / max(diag['offset_entropy_source_oof'], 1e-12)
    return {'arms': {'S_plus_A': z_a, 'ERM_RES_A': z_tgt}, 'diag': diag}
