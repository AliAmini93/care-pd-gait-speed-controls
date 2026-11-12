"""Fabricated-only tests for R8 card d_sequence_loco (never reads real data; values are not evidence)."""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'care_pd'))
sys.path.insert(1, str(Path(__file__).resolve().parent))
import common as C
import p1_core as Q
import r8_core as R
import r8_d_sequence_loco as D
import r8_a_occasion_grain_synthetic_tests as AT
import r8_c_insite_ceiling_synthetic_tests as CT
FAILS = []
MARK = AT.MARK
FAST_TRAIN = dict(D.TRAIN, epochs=3, batch=64)
FAST_CONFIGS = (('GRU8', 'gru', 8), ('CNN8', 'cnn', 8))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def test_resample():
    fps, T = (150, 600)
    tt = np.arange(T) / fps
    pose = np.zeros((T, 72))
    pose[:, 5] = 0.7
    pose[:, 9] = np.sin(2 * np.pi * tt)
    trans = np.column_stack([np.zeros(T), np.zeros(T), 1.3 * tt])
    s = D.resample_walk(pose, trans, fps)
    check(s.shape == (100, 75) and s.dtype == np.float32, f'4 s at 150 fps -> 100 frames at 25 Hz with 75 columns ({s.shape})')
    check(np.allclose(s[:, 5], 0.7) and abs(s[:, 9].max() - 1) < 0.05, 'pose columns interpolate without distortion')
    check(np.allclose(s[:, 74], 1.3, atol=0.0001) and np.allclose(s[:, 72:74], 0), 'root velocity equals the constant forward speed')
    s2 = D.resample_walk(pose[:3], trans[:3], fps)
    check(s2.shape[0] >= 2, 'very short walks keep at least two frames')
    long = D.resample_walk(np.zeros((30 * 60, 72)), np.zeros((30 * 60, 3)), 30)
    check(long.shape[0] == D.MAX_FRAMES, 'long walks are truncated at 800 frames')
    s25 = D.resample_walk(pose[:100], trans[:100], 25)
    check(s25.shape[0] == 100, '25 fps input keeps its frame count')

def test_batching_and_training():
    rng = np.random.default_rng(0)
    seqs = [rng.normal(size=(int(n), 75)).astype(np.float32) for n in (30, 250, 120)]
    x, L = D.pad_batch(seqs, [0, 1, 2], 100, rng, 'cpu')
    check(x.shape == (3, 100, 75) and L.tolist() == [30, 100, 100], 'random crop caps at 100 frames and pads shorter walks')
    x2, L2 = D.pad_batch(seqs, [0, 1], None, rng, 'cpu')
    check(x2.shape == (2, 250, 75) and L2.tolist() == [30, 250], 'inference uses full sequences')
    n = 240
    y = rng.integers(0, 3, n)
    seqs = []
    for i in range(n):
        s = rng.normal(0, 0.1, size=(80, 75)).astype(np.float32)
        s[:, 74] += 1.5 - 0.5 * y[i]
        seqs.append(s)
    tr, te = (np.arange(0, 180), np.arange(180, 240))
    w = Q.train_weights(y[tr], np.arange(180), 'walk')
    p, emb = D.fit_predict_seq(seqs, 75, y[tr], w, tr, te, ('GRU8', 'gru', 8), 1, 'cpu', dict(D.TRAIN, epochs=25))
    check(p.shape == (60, 3) and np.allclose(p.sum(axis=1), 1) and (emb.shape == (60, 16)), 'probabilities and embeddings have the expected shapes')
    check(Q.walk_macro_f1(y[te], p.argmax(axis=1)) > 0.6, f'a planted velocity signal is learned (macro-F1 {Q.walk_macro_f1(y[te], p.argmax(axis=1)):.2f}, chance 0.33)')
    p2, _ = D.fit_predict_seq(seqs, 75, y[tr], w, tr, te, ('GRU8', 'gru', 8), 1, 'cpu', dict(D.TRAIN, epochs=25))
    check(np.allclose(p, p2), 'training is deterministic for a fixed seed on CPU')
    y_scr = y.copy()
    y_scr[te] = (y_scr[te] + 1) % 3
    p3, _ = D.fit_predict_seq(seqs, 75, y_scr[tr], w, tr, te, ('GRU8', 'gru', 8), 1, 'cpu', dict(D.TRAIN, epochs=25))
    check(np.allclose(p, p3), 'held-out labels do not influence predictions (only source labels are passed)')
    seqs_shift = [s.copy() for s in seqs]
    for i in te:
        seqs_shift[i][:, 74] += 100.0
    p4, _ = D.fit_predict_seq(seqs_shift, 75, y[tr], w, tr, te, ('GRU8', 'gru', 8), 1, 'cpu', dict(D.TRAIN, epochs=25))
    check(not np.allclose(p, p4), 'held-out inputs are standardised with source statistics (a held-out shift is not absorbed)')
    pc, embc = D.fit_predict_seq(seqs, 72, y[tr], w, tr, te, ('CNN8', 'cnn', 8), 1, 'cpu', dict(D.TRAIN, epochs=3))
    check(pc.shape == (60, 3) and embc.shape == (60, 8), 'CNN arm with pose-only input works')

def test_loco_fold_and_end_to_end():
    objs = AT.synth_objs('speed', seed=71)
    t = Q.build_table(objs)
    seqs = D.build_sequences(objs)
    check(len(seqs) == t['label'].size and all((s.shape[1] == 75 for s in seqs)), 'sequences align with the table rows')
    ci = C.COHORTS.index('BMCLab')
    f = D.seq_outer_fold(seqs, 75, t['label'], t['cohort'], t['subject'], ci, FAST_CONFIGS, 3, DEVICE, FAST_TRAIN)
    check(np.array_equal(f['test_idx'], np.flatnonzero(np.isin(t['label'], (0, 1, 2)) & (t['cohort'] == ci))), "LOCO fold tests exactly the held-out cohort's eligible walks")
    check(f['selected'] in ('GRU8', 'CNN8') and set(f['inner_mean_by_config']) == {'GRU8', 'CNN8'}, 'inner selection records every config')
    body = D.run(t, seqs, p1=None, configs=FAST_CONFIGS, seeds=(3,), train=FAST_TRAIN, n_boot=40, device=DEVICE, sk_configs=AT.FAST)
    m_full = body['arms']['SEQ_full_seed3']['macro_f1']['M']['point']
    m_pose = body['arms']['SEQ_pose_seed3']['macro_f1']['M']['point']
    check(m_full > 0.45, f'speed world: the full-input sequence model recovers the velocity signal ({m_full:.3f})')
    check(m_pose < m_full - 0.1, f'speed world: the pose-only sequence model cannot see speed ({m_pose:.3f} vs {m_full:.3f})')
    check(set(body['reading']['per_arm']) == set(D.ARMS), 'a reading is produced per arm')
    sid = body['site_identifiability_of_out_of_site_embeddings']['SEQ_full_seed3']
    check(sid is None or 0 <= sid <= 1, f'site identifiability of embeddings is reported or undefined when encoder widths differ ({sid})')
    text = R.render(body)
    check(MARK not in text and 'emb' not in body['arms'], 'no identifier or embedding reaches the output')
    check(len(text.encode()) < 262144 * 0.5, f'output well under the summary cap ({len(text)} bytes)')
    objs_p = CT.pose_world(72)
    t_p = Q.build_table(objs_p)
    seqs_p = D.build_sequences(objs_p)
    f_p = D.seq_outer_fold(seqs_p, 72, t_p['label'], t_p['cohort'], t_p['subject'], ci, (('CNN16', 'cnn', 16),), 4, DEVICE, dict(D.TRAIN, epochs=12))
    y_te = t_p['label'][f_p['test_idx']]
    check(Q.walk_macro_f1(y_te, f_p['proba'].argmax(axis=1)) > 0.5, f"pose world: the pose-only sequence model recovers a planted knee-amplitude signal across sites ({Q.walk_macro_f1(y_te, f_p['proba'].argmax(axis=1)):.2f})")
    check(raises_source(lambda: D.run(t, seqs[:-1], p1=None, configs=FAST_CONFIGS, seeds=(3,), train=FAST_TRAIN, n_boot=10, device=DEVICE, sk_configs=AT.FAST)), 'row misalignment is refused')

def main():
    print(f'device={DEVICE}')
    test_resample()
    test_batching_and_training()
    test_loco_fold_and_end_to_end()
    if FAILS:
        print(f'R8_D_SYNTHETIC_FAIL {len(FAILS)}: ' + '; '.join(FAILS))
        return 1
    print('R8_D_SYNTHETIC_PASS all fabricated checks passed (no real data read)')
    return 0
if __name__ == '__main__':
    sys.exit(main())
