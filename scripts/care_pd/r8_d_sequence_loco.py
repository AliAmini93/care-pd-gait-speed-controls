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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import p0_core as P
import p1_core as Q
import r8_core as R
CARD_ID = 'd_sequence_loco'
STATUS = 'R8_D_SEQUENCE_LOCO_COMPLETE'
RATE_HZ = 25.0
MAX_FRAMES = 800
CROP_FRAMES = 100
D_POSE = 72
D_FULL = 75
ARMS = {'SEQ_full': D_FULL, 'SEQ_pose': D_POSE}
SEEDS = (20261106, 20261107)
TRAIN = {'epochs': 25, 'batch': 64, 'lr': 0.001, 'weight_decay': 0.0001, 'dropout': 0.2}
CONFIGS = (('GRU64', 'gru', 64), ('GRU128', 'gru', 128), ('CNN64', 'cnn', 64))
N_BOOT = 2000
TOL = 1.1e-06
READ_MARGIN = 0.03
LIMITS = ['Controlled baseline, not an innovation; GRU/CNN sequence classifiers are standard.', 'Four previously inspected cohorts; LOCO with inner source-only selection as P1; exploratory, not independent-site evidence.', 'Fixed epochs, no early stopping; selection only among three fixed configs by inner source-site macro-F1.', 'CUDA kernels may be nondeterministic across runs; two seeds are reported with their range.', 'Sequences resampled to 25 Hz, truncated at 800 frames; root velocity arm carries speed information by construction.', 'Bootstrap intervals resample persons within site with predictions fixed; refit variance is shown by the seed range only.']

def resample_walk(pose, trans, fps, rate=RATE_HZ, max_frames=MAX_FRAMES):
    T = pose.shape[0]
    t_old = np.arange(T) / fps
    n_new = max(2, int(np.floor(t_old[-1] * rate)) + 1)
    t_new = np.arange(n_new) / rate
    t_new = t_new[t_new <= t_old[-1] + 1e-09]
    if t_new.size < 2:
        t_new = np.array([0.0, t_old[-1]])
    P_new = np.column_stack([np.interp(t_new, t_old, pose[:, j]) for j in range(pose.shape[1])])
    X_new = np.column_stack([np.interp(t_new, t_old, trans[:, j]) for j in range(trans.shape[1])])
    vel = np.diff(X_new, axis=0) * rate
    vel = np.vstack([vel[:1], vel])
    out = np.column_stack([P_new, vel]).astype(np.float32)
    return out[:max_frames]

def build_sequences(cohort_objs):
    seqs = []
    for name in C.COHORTS:
        for _, rec in C.walks_of(cohort_objs[name]):
            pose, trans = C.read_arrays(rec)
            if not (np.isfinite(pose).all() and np.isfinite(trans).all()):
                continue
            seqs.append(resample_walk(pose, trans, C.read_fps(rec['fps'])))
    return seqs

class SeqNet(torch.nn.Module):

    def __init__(self, d_in, kind, hidden, dropout):
        super().__init__()
        self.kind = kind
        if kind == 'gru':
            self.enc = torch.nn.GRU(d_in, hidden, batch_first=True, bidirectional=True)
            d_emb = 2 * hidden
        elif kind == 'cnn':
            self.enc = torch.nn.Sequential(torch.nn.Conv1d(d_in, hidden, 5, padding=2), torch.nn.ReLU(), torch.nn.Conv1d(hidden, hidden, 5, padding=2), torch.nn.ReLU(), torch.nn.Conv1d(hidden, hidden, 5, padding=2), torch.nn.ReLU())
            d_emb = hidden
        else:
            raise ValueError(kind)
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(d_emb, 3)

    def forward(self, x, lengths):
        mask = (torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]).float()
        if self.kind == 'gru':
            packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            o, _ = self.enc(packed)
            o, _ = torch.nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.shape[1])
        else:
            o = self.enc(x.transpose(1, 2)).transpose(1, 2)
        emb = (o * mask[:, :, None]).sum(1) / lengths[:, None].float()
        return (self.head(self.drop(emb)), emb)

def pad_batch(seqs, idx, crop, rng, device):
    arrs, lengths = ([], [])
    for i in idx:
        s = seqs[i]
        if crop is not None and s.shape[0] > crop:
            start = int(rng.integers(0, s.shape[0] - crop + 1))
            s = s[start:start + crop]
        arrs.append(s)
        lengths.append(s.shape[0])
    L = max(lengths)
    x = np.zeros((len(idx), L, arrs[0].shape[1]), dtype=np.float32)
    for k, s in enumerate(arrs):
        x[k, :s.shape[0]] = s
    return (torch.from_numpy(x).to(device), torch.tensor(lengths, device=device))

def fit_predict_seq(seqs, d_in, y_tr, w_tr, tr_idx, te_idx, config, seed, device, train=TRAIN):
    name, kind, hidden = config
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    frames = np.concatenate([seqs[i][:, :d_in] for i in tr_idx])
    mu, sd = (frames.mean(axis=0), frames.std(axis=0))
    sd = np.where(sd > 1e-08, sd, 1.0)
    std = [((seqs[i][:, :d_in] - mu) / sd).astype(np.float32) for i in range(len(seqs))]
    net = SeqNet(d_in, kind, hidden, train['dropout']).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=train['lr'], weight_decay=train['weight_decay'])
    y_t = torch.as_tensor(np.asarray(y_tr), dtype=torch.long, device=device)
    w_t = torch.as_tensor(np.asarray(w_tr), dtype=torch.float32, device=device)
    n = len(tr_idx)
    net.train()
    for _ in range(train['epochs']):
        perm = rng.permutation(n)
        for i in range(0, n, train['batch']):
            b = perm[i:i + train['batch']]
            x, L = pad_batch(std, [tr_idx[k] for k in b], CROP_FRAMES, rng, device)
            opt.zero_grad()
            logits, _ = net(x, L)
            loss = (torch.nn.functional.cross_entropy(logits, y_t[b], reduction='none') * w_t[b]).sum() / w_t[b].sum()
            loss.backward()
            opt.step()
    net.eval()
    probs, embs = ([], [])
    with torch.no_grad():
        for i in range(0, len(te_idx), 64):
            x, L = pad_batch(std, te_idx[i:i + 64], None, rng, device)
            logits, emb = net(x, L)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            embs.append(emb.cpu().numpy())
    del net, opt, std
    return (np.concatenate(probs).astype(np.float64), np.concatenate(embs))

def seq_outer_fold(seqs, d_in, label, cohort, subject, ci, configs, seed, device, train=TRAIN):
    eligible = np.isin(label, (0, 1, 2))
    tr = np.flatnonzero(eligible & (cohort != ci))
    te = np.flatnonzero(eligible & (cohort == ci))
    sources = np.unique(cohort[tr])
    inner = np.zeros((len(configs), sources.size))
    for c, cfg in enumerate(configs):
        for s, v in enumerate(sources):
            a = tr[cohort[tr] != v]
            b = tr[cohort[tr] == v]
            w = Q.train_weights(label[a], subject[a], 'walk')
            p, _ = fit_predict_seq(seqs, d_in, label[a], w, a, b, cfg, seed + 1000 * c + 10 * s, device, train)
            inner[c, s] = Q.walk_macro_f1(label[b], p.argmax(axis=1))
    best = Q.select_config(inner.mean(axis=1), list(range(len(configs))))
    w = Q.train_weights(label[tr], subject[tr], 'walk')
    proba, emb = fit_predict_seq(seqs, d_in, label[tr], w, tr, te, configs[best], seed + 7, device, train)
    return {'test_idx': te, 'proba': proba, 'emb': emb, 'selected': configs[best][0], 'inner_mean_by_config': {cfg[0]: float(v) for cfg, v in zip(configs, inner.mean(axis=1))}}

def embedding_site_identifiability(embs, cohort, subject):
    out = []
    scaler = StandardScaler()
    for tr, va in GroupKFold(n_splits=5).split(embs, cohort, subject):
        model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, class_weight='balanced')
        model.fit(scaler.fit_transform(embs[tr]), cohort[tr])
        pred = model.predict(scaler.transform(embs[va]))
        out.append(np.mean([(pred[cohort[va] == c] == c).mean() for c in np.unique(cohort[va])]))
    return float(np.mean(out))

def run(t, seqs, p1=None, configs=CONFIGS, seeds=SEEDS, train=TRAIN, n_boot=N_BOOT, device=None, sk_configs=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if len(seqs) != t['label'].size:
        raise C.SourceError('R8_D_SEQUENCE_ROW_ALIGNMENT')
    feats = Q.arm_features(t)
    warned = [0]
    scorer = Q.Scorer(t, n_boot)
    names = scorer.names
    sk_configs = Q.CONFIGS if sk_configs is None else sk_configs
    folds_s = [R.single_outer_fold(feats['S'], t['label'], t['cohort'], t['subject'], 'walk', sk_configs, warned, ci) for ci in range(len(names))]
    preds_s = [R.selected_proba(f).argmax(axis=1) for f in folds_s]
    if p1 is not None:
        pt, _ = scorer.stats(scorer.conf(preds_s))
        for ci, site in enumerate(names):
            if sk_configs[folds_s[ci]['selected']['all']][0] != p1['selected_configs']['S'][site]['all'] or abs(pt[ci] - p1['arms_inductive_strong']['S']['macro_f1']['per_cohort'][site]['point']) > TOL:
                raise C.SourceError('R8_D_S_PARITY:' + site)
    conf_s = scorer.conf(preds_s)
    results = {'arms': {}, 'contrasts_vs_S': {}, 'selected': {}, 'inner': {}, 'site_identifiability_of_out_of_site_embeddings': {}, 'timing_s': {}, 'S_loco_macro_f1': scorer.block(*scorer.stats(conf_s))}
    eligible = np.isin(t['label'], (0, 1, 2))
    for arm, d_in in ARMS.items():
        for seed in seeds:
            key = f'{arm}_seed{seed}'
            t0 = time.monotonic()
            preds, embs, cohorts, subjects, sel, inner = ([], [], [], [], {}, {})
            for ci, site in enumerate(names):
                f = seq_outer_fold(seqs, d_in, t['label'], t['cohort'], t['subject'], ci, configs, seed + 100 * ci, device, train)
                if not np.array_equal(f['test_idx'], np.flatnonzero(scorer.masks[site])):
                    raise C.SourceError('R8_D_TEST_ROWS')
                preds.append(f['proba'].argmax(axis=1))
                embs.append(f['emb'])
                cohorts.append(np.full(f['emb'].shape[0], ci))
                subjects.append(t['subject'][f['test_idx']])
                sel[site], inner[site] = (f['selected'], f['inner_mean_by_config'])
            conf = scorer.conf(preds)
            results['arms'][key] = scorer.all_metrics(conf)
            results['contrasts_vs_S'][key] = {'macro_f1': scorer.contrast(conf, conf_s), 'macro_f1_person_weighted': scorer.contrast(conf, conf_s, person=True)}
            results['selected'][key], results['inner'][key] = (sel, inner)
            dims = {e.shape[1] for e in embs}
            results['site_identifiability_of_out_of_site_embeddings'][key] = embedding_site_identifiability(np.concatenate(embs), np.concatenate(cohorts), np.concatenate(subjects)) if len(dims) == 1 else None
            results['timing_s'][key] = round(time.monotonic() - t0, 1)
            del embs
    verdicts = {}
    for arm in ARMS:
        ms = [results['contrasts_vs_S'][f'{arm}_seed{s}']['macro_f1']['M'] for s in seeds]
        beats = all((m['point'] >= READ_MARGIN and m['ci95'][0] > 0 for m in ms))
        fails = all((m['point'] < READ_MARGIN or m['ci95'][0] <= 0 for m in ms))
        verdicts[arm] = 'SEQUENCE_BEATS_SPEED_FLOOR' if beats else 'SEQUENCE_DOES_NOT_BEAT_SPEED_FLOOR' if fails else 'MIXED'
    results['reading'] = {'per_arm': verdicts, 'margin': READ_MARGIN, 'seeds': list(seeds)}
    results['protocol'] = {'rate_hz': RATE_HZ, 'max_frames': MAX_FRAMES, 'crop_frames': CROP_FRAMES, 'arms': ARMS, 'configs': [c[0] for c in configs], 'train': train, 'device': device, 'n_boot': n_boot, 'selection': 'inner leave-one-source-site-out walk macro-F1 (P1 rule)'}
    results['n_convergence_warnings_S'] = warned[0]
    return results
