"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
from scipy.signal import hilbert
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parent))
import abcd_baseline as AB
import abcd_common as AC
import abcd_stage_a as SA
STAGE = 'B'
CANDIDATE = 'S_plus_B'
CONTROLS = ('ERM_RES_B',)
SYNTH_FALSIFIERS = ('FX_APERIODIC',)
FS = 30.0
PAIRS = (('hip', 1, 2), ('knee', 4, 5), ('ankle', 7, 8), ('shoulder', 16, 17), ('elbow', 18, 19))
BAND = (0.3, 3.0)
GRID_HZ = 0.025
BP = (0.5, 1.5)
N_BINS = 20
MIN_CYCLES = 2
MIN_AUTOCORR = 0.5
MIN_MONOTONE = 0.9
PAD_MAX = 60
LOG_EPS = 0.01
OBSERVABILITY_MIN_SHARE = 0.5
N_FEATURES = 16
FEATURE_NAMES = tuple([f'V_{p[0]}' for p in PAIRS] + [f'SYM_{p[0]}' for p in PAIRS] + [f'RR_{p[0]}' for p in PAIRS] + ['CV_stride'])
CFG = {'proxy': 'Hilbert phase of the per-walk principal angle of a hip rotation (both hips as reference, averaged); an observable cyclic proxy, not a gait event', 'frame_rate': FS, 'band_hz': list(BAND), 'grid_hz': GRID_HZ, 'bandpass_factor': list(BP), 'observable_rule': {'min_autocorr_at_stride_lag': MIN_AUTOCORR, 'min_monotone_share': MIN_MONOTONE, 'min_complete_cycles': MIN_CYCLES}, 'phase_bins': N_BINS, 'joints': [p[0] for p in PAIRS], 'features': list(FEATURE_NAMES), 'n_features': N_FEATURES, 'not_observable_handling': 'NaN -> 0 after speed-conditional scaling (source mean)', 'observability_block_rule': 'DEV_BLOCKED before any score if any development cohort has an observable share < 0.5 (label-free)', 'observability_min_share': OBSERVABILITY_MIN_SHARE, 'head_and_gate': 'identical to stage A (abcd_stage_a.residual_logits and gate, imported unchanged)', 'ridge_lambda': AC.RIDGE_LAMBDA, 'conditioning_speed_floor': SA.COND_SPEED_FLOOR, 'n_parameters_head': 1 + N_FEATURES * 3}

def principal_angle(R):
    Rm = AC.chordal_mean(R[:, None])[0]
    w = Rotation.from_matrix(np.einsum('ji,kjl->kil', Rm, R)).as_rotvec()
    w = w - w.mean(axis=0)
    _, _, vt = np.linalg.svd(w, full_matrices=False)
    return w @ vt[0]

def stride_frequency(x, fs=FS):
    x = x - x.mean()
    n = max(x.size, int(round(fs / GRID_HZ)))
    power = np.abs(np.fft.rfft(x * np.hanning(x.size), n=n)) ** 2
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    band = (freq >= BAND[0]) & (freq <= BAND[1])
    if power[band].sum() <= 0:
        return None
    return float(freq[band][int(np.argmax(power[band]))])

def autocorr_at(x, lag):
    if lag < 1 or lag > x.size - 8:
        return -1.0
    a, b = (x[:-lag] - x[:-lag].mean(), x[lag:] - x[lag:].mean())
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else -1.0

def hilbert_phase(x, f, fs=FS):
    k = x.size
    t = np.arange(k)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    pad = min(k - 1, PAD_MAX)
    xp = np.pad(x, pad, mode='reflect')
    spec = np.fft.rfft(xp)
    freq = np.fft.rfftfreq(xp.size, 1.0 / fs)
    spec[(freq < BP[0] * f) | (freq > BP[1] * f)] = 0.0
    bp = np.fft.irfft(spec, n=xp.size)
    return np.unwrap(np.angle(hilbert(bp)))[pad:pad + k]

def reference_phase(x):
    info = {'observable': False}
    f = stride_frequency(x)
    if f is None:
        return (None, None, info)
    info['f'] = f
    info['autocorr'] = autocorr_at(x - x.mean(), int(round(FS / f)))
    if info['autocorr'] < MIN_AUTOCORR:
        return (None, None, info)
    ph = hilbert_phase(x, f)
    info['monotone'] = float((np.diff(ph) > 0).mean())
    if info['monotone'] < MIN_MONOTONE:
        return (None, None, info)
    ph = np.maximum.accumulate(ph)
    idx = np.floor(ph / (2 * np.pi)).astype(int)
    bounds = np.flatnonzero(np.diff(idx) > 0) + 1
    if bounds.size - 1 < MIN_CYCLES:
        return (None, None, info)
    info['observable'] = True
    info['n_cycles'] = int(bounds.size - 1)
    return (ph, bounds, info)

def cycle_matrix(x, ph, bounds):
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        p0 = 2 * np.pi * np.floor(ph[a] / (2 * np.pi))
        target = p0 + 2 * np.pi * (np.arange(N_BINS) + 0.5) / N_BINS
        seg = slice(max(a - 1, 0), min(b + 1, x.size))
        out.append(np.interp(target, ph[seg], x[seg]))
    return np.asarray(out)

def _features_one_reference(sig, ph, bounds):
    v, sym, rr = ([], [], [])
    for name, left, right in PAIRS:
        m = {}
        vv = []
        for j in (left, right):
            C = cycle_matrix(sig[j], ph, bounds)
            mean = C.mean(axis=0)
            rng_ = float(mean.max() - mean.min())
            vv.append(float(C.std(axis=0).mean() / (rng_ + 1e-06)))
            m[j] = (mean, rng_)
        v.append(np.log(np.mean(vv) + LOG_EPS))
        a, b = (m[left][0], np.roll(m[right][0], N_BINS // 2))
        den = a.std() * b.std()
        sym.append(abs(float(((a - a.mean()) * (b - b.mean())).mean() / den)) if den > 0 else 0.0)
        rr.append(abs(np.log((m[left][1] + 1e-06) / (m[right][1] + 1e-06))))
    dur = np.diff(bounds).astype(float)
    cv = np.log(dur.std() / dur.mean() + LOG_EPS)
    return np.asarray(v + sym + rr + [cv])

def walk_features(R_local):
    sig = {j: principal_angle(R_local[:, j - 1]) for p in PAIRS for j in p[1:]}
    feats, infos = ([], [])
    for ref in (1, 2):
        ph, bounds, info = reference_phase(sig[ref])
        infos.append(info)
        if ph is None:
            return (np.full(N_FEATURES, np.nan), infos)
        feats.append(_features_one_reference(sig, ph, bounds))
    return ((feats[0] + feats[1]) / 2.0, infos)

def features(store):
    F = np.empty((len(store), N_FEATURES))
    for i in range(len(store)):
        F[i], _ = walk_features(AC.local_matrices(store, i))
    if np.isinf(F).any():
        raise AC.AbcdError('ABCD_B_FEATURE_INF')
    return F

def method_fold(src, tgt, s_head, warned):
    config = AB.CONFIGS[s_head['index']]
    o_tgt = AB.offsets(s_head['proba'])
    z_tgt, head, oof = SA.residual_logits(src['F'], src['speed'], src['label'], src['subject'], config, tgt['F'], tgt['speed'], o_tgt, warned)
    is_open, inner = SA.gate(src, config, warned)
    z_b = z_tgt if is_open else o_tgt
    diag = {'gate_open': is_open, 'inner_deltas': inner.tolist(), 'inner_mean_delta': float(inner.mean()), 'tau': head.tau, 'w_frobenius': head.diag['w_frobenius'], 'converged': head.diag['converged'], 'observable_share_source': float(np.isfinite(src['F'][:, 0]).mean()), 'observable_share_target': float(np.isfinite(tgt['F'][:, 0]).mean()), 'offset_entropy_ratio_target_over_source': AB.entropy(s_head['proba']) / max(AB.entropy(oof), 1e-12), 'S_config': config[0]}
    return {'arms': {'S_plus_B': z_b, 'ERM_RES_B': z_tgt}, 'diag': diag}

def _phase_check(world):
    import abcd_worlds as W
    data = W.generate(world, 'dev', 0)
    n_obs, plv_ok, f_ok, n = (0, 0, 0, 0)
    for i in range(len(data['pose'])):
        if data['env'][i] != W.TARGET_ENV and world == 'W1' and i % 2:
            continue
        n += 1
        store = AC.build_store([data['pose'][i]], [data['fps'][i]])
        R = AC.local_matrices(store, 0)
        x = principal_angle(R[:, 0])
        ph, bounds, info = reference_phase(x)
        if ph is None:
            continue
        n_obs += 1
        tr = data['truth'][i]
        fps = float(data['fps'][i])
        pos = np.arange(ph.size) * fps / FS
        true = np.interp(pos, np.arange(tr['phase'].size), tr['phase'].astype(np.float64))
        keep = slice(int(0.1 * ph.size), int(0.9 * ph.size) + 1)
        d = ph[keep] - true[keep]
        plv = max(abs(np.exp(1j * d).mean()), abs(np.exp(1j * (ph[keep] + true[keep])).mean()))
        plv_ok += plv >= 0.8
        f_ok += abs(info['f'] - tr['f_apparent']) <= 0.1 * tr['f_apparent']
    return (n, n_obs, plv_ok, f_ok)

def check_phase_observable_w1():
    n, n_obs, plv_ok, f_ok = _phase_check('W1')
    share = n_obs / n
    return {'pass': bool(share >= 0.9 and plv_ok >= 0.8 * n_obs and (f_ok >= 0.9 * n_obs)), 'n': n, 'observable_share': share, 'plv_ge_0_8_share': plv_ok / max(n_obs, 1), 'freq_within_10pct_share': f_ok / max(n_obs, 1)}

def check_phase_not_observable_aperiodic():
    n, n_obs, _, _ = _phase_check('FX_APERIODIC')
    return {'pass': bool(n_obs / n <= 0.2), 'n': n, 'observable_share': n_obs / n}

def check_left_right_label_swap():
    import abcd_worlds as W
    data = W.generate('W1', 'dev', 0)
    poses = data['pose'][:30]
    swapped = []
    for p in poses:
        q = p.copy()
        for left, right in ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)):
            q[:, 3 * left:3 * left + 3], q[:, 3 * right:3 * right + 3] = (p[:, 3 * right:3 * right + 3], p[:, 3 * left:3 * left + 3])
        swapped.append(q)
    f1 = features(AC.build_store(poses, data['fps'][:30]))
    f2 = features(AC.build_store(swapped, data['fps'][:30]))
    same_nan = np.array_equal(np.isnan(f1), np.isnan(f2))
    err = float(np.nanmax(np.abs(f1 - f2))) if np.isfinite(f1).any() else 0.0
    return {'pass': bool(same_nan and err < 1e-09), 'max_abs_err': err}

def check_global_amplitude_scale():
    import abcd_worlds as W
    data = W.generate('W1', 'dev', 0)
    poses = data['pose'][:30]
    scaled = []
    for p in poses:
        q = p.copy()
        q[:, 3:] = 1.3 * q[:, 3:]
        scaled.append(q)
    f1 = features(AC.build_store(poses, data['fps'][:30]))
    f2 = features(AC.build_store(scaled, data['fps'][:30]))
    ok = np.isfinite(f1) & np.isfinite(f2)
    err = float(np.abs(f1[ok] - f2[ok]).max()) if ok.any() else 0.0
    return {'pass': bool(err < 0.25 and np.array_equal(np.isnan(f1), np.isnan(f2))), 'max_abs_err': err}
EXTRA_VALIDITY = {'B01_phase_observable_on_W1_truth': check_phase_observable_w1, 'B02_phase_not_observable_on_FX_APERIODIC': check_phase_not_observable_aperiodic, 'B03_left_right_label_swap_invariance': check_left_right_label_swap, 'B04_global_amplitude_scale_robustness': check_global_amplitude_scale}

class _Cached:

    def __init__(self, F):
        self.F = F
        self.STAGE, self.CANDIDATE, self.CONTROLS = (STAGE, CANDIDATE, CONTROLS)
        self.method_fold = method_fold

    def features(self, store):
        if len(store) != self.F.shape[0]:
            raise AC.AbcdError('ABCD_B_CACHE_ROWS')
        return self.F

def observability(F, b):
    out = {}
    for s in AC.DEV_SITES:
        m = b['site'] == AC.C.COHORTS.index(s)
        out[s] = {'walks': int(m.sum()), 'observable_share': float(np.isfinite(F[m, 0]).mean())}
    return out
