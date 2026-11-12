"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import p1_core as Q
import representation_core as RC
CONFIG = {'width': 64, 'blocks': 3, 'kernel': 5, 'dilations': (1, 2, 4), 'embed_dim': 64, 'dropout': 0.2, 'epochs_max': 40, 'snapshot_epochs': (4, 8, 12, 16, 20, 24, 28, 32, 36, 40), 'batch': 64, 'lr': 0.001, 'weight_decay': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-08, 'warmup_epochs': 1, 'schedule': 'cosine_to_zero', 'grad_clip': 1.0, 'seeds': (20261105, 20261106, 20261107), 'eval_batch': 256, 'precision': 'float32', 'augmentation': 'one random 60-frame crop per walk per epoch', 'class_weights': 'P1 walk scheme, training only'}
PARAM_BUDGET = 350000

def seed_for(outer, slot, seed):
    return int(seed + 100000 * outer + 1000 * slot)

def graph_partitions(parents=RC.SMPL_PARENTS):
    V = len(parents)
    A = np.zeros((3, V, V), dtype=np.float32)
    for u in range(V):
        A[0, u, u] = 1.0
        if parents[u] >= 0:
            A[1, u, parents[u]] = 1.0
            A[2, parents[u], u] = 1.0
    deg = A[2].sum(axis=1, keepdims=True)
    A[2] = np.divide(A[2], deg, out=np.zeros_like(A[2]), where=deg > 0)
    return A

class Block(nn.Module):

    def __init__(self, width, kernel, dilations, dropout):
        super().__init__()
        self.width = width
        self.graph = nn.Linear(width, 3 * width, bias=False)
        self.graph_bias = nn.Parameter(torch.zeros(width))
        self.norm1 = nn.LayerNorm(width)
        self.dw = nn.ModuleList([nn.Conv1d(width, width, kernel, padding=d * (kernel - 1) // 2, dilation=d, groups=width) for d in dilations])
        self.pw = nn.Linear(len(dilations) * width, width)
        self.norm2 = nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, m, A):
        B, T, V, Cw = x.shape
        y = self.graph(x).reshape(B, T, V, 3, Cw)
        h = torch.einsum('puv,btvpc->btuc', A, y) + self.graph_bias
        h = F.gelu(self.norm1(h)) * m
        z = h.permute(0, 2, 3, 1).reshape(B * V, Cw, T)
        z = torch.cat([conv(z) for conv in self.dw], dim=1)
        z = z.reshape(B, V, -1, T).permute(0, 3, 1, 2)
        z = self.drop(F.gelu(self.norm2(self.pw(z)))) * m
        return x + z

class SMPLGraphTCN(nn.Module):

    def __init__(self, cfg=CONFIG):
        super().__init__()
        w = cfg['width']
        self.register_buffer('A', torch.from_numpy(graph_partitions()))
        self.inp = nn.Linear(6, w)
        self.joint = nn.Parameter(torch.zeros(RC.N_JOINTS, w))
        self.blocks = nn.ModuleList([Block(w, cfg['kernel'], cfg['dilations'], cfg['dropout']) for _ in range(cfg['blocks'])])
        self.drop = nn.Dropout(cfg['dropout'])
        self.emb = nn.Linear(RC.N_JOINTS * w, cfg['embed_dim'])
        self.head = nn.Linear(cfg['embed_dim'], 3)

    def forward(self, x, mask, return_embedding=False):
        m = mask.to(x.dtype)[:, :, None, None]
        h = (self.inp(x) + self.joint[None, None]) * m
        for blk in self.blocks:
            h = blk(h, m, self.A)
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = (h * m).sum(dim=1) / denom
        e = F.gelu(self.emb(self.drop(pooled.reshape(pooled.shape[0], -1))))
        logits = self.head(self.drop(e))
        return (logits, e) if return_embedding else logits

def n_parameters(model):
    return int(sum((p.numel() for p in model.parameters() if p.requires_grad)))

def train_batch(store, idx, rng, device):
    X = np.zeros((len(idx), RC.WINDOW, RC.N_JOINTS, 6), dtype=np.float32)
    M = np.zeros((len(idx), RC.WINDOW), dtype=bool)
    for k, i in enumerate(idx):
        X[k], M[k] = RC.window_array(store, i, RC.crop_start(int(store.lengths[i]), rng))
    return (torch.from_numpy(X).to(device), torch.from_numpy(M).to(device))

def prepare_eval(store, idx):
    owners, X, M = ([], [], [])
    for pos, i in enumerate(idx):
        for s in RC.eval_starts(int(store.lengths[i])):
            x, m = RC.window_array(store, i, s)
            owners.append(pos)
            X.append(x)
            M.append(m)
    if not X:
        raise C.SourceError('REP_EMPTY_EVAL_SET')
    M = np.stack(M)
    return {'n': len(idx), 'owners': np.asarray(owners), 'X': np.stack(X), 'M': M, 'frames': M.sum(axis=1).astype(np.float64)}

def walk_logits(model, prepared, device, eval_batch=None):
    eval_batch = eval_batch or CONFIG['eval_batch']
    model.eval()
    out = np.zeros((prepared['n'], 3))
    frames = np.zeros(prepared['n'])
    X, M, own, nf = (prepared['X'], prepared['M'], prepared['owners'], prepared['frames'])
    with torch.no_grad():
        for a in range(0, X.shape[0], eval_batch):
            xb = torch.from_numpy(X[a:a + eval_batch]).to(device)
            mb = torch.from_numpy(M[a:a + eval_batch]).to(device)
            lg = model(xb, mb).double().cpu().numpy()
            np.add.at(out, own[a:a + eval_batch], lg * nf[a:a + eval_batch, None])
            np.add.at(frames, own[a:a + eval_batch], nf[a:a + eval_batch])
    if np.any(frames <= 0):
        raise C.SourceError('REP_EMPTY_WALK')
    return out / frames[:, None]

def _lr_factor(step, steps_per_epoch, cfg):
    warm = cfg['warmup_epochs'] * steps_per_epoch
    total = cfg['epochs_max'] * steps_per_epoch
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

def train_snapshots(store, train_idx, y_train, w_train, eval_idx, prepared, seed, device, snapshots, cfg=CONFIG, last=None):
    train_idx = np.asarray(train_idx)
    y_train, w_train = (np.asarray(y_train), np.asarray(w_train, dtype=np.float64))
    if train_idx.size != y_train.size or y_train.size != w_train.size or train_idx.size == 0:
        raise C.SourceError('REP_TRAIN_SHAPE')
    if np.intersect1d(train_idx, np.asarray(eval_idx)).size or prepared['n'] != len(eval_idx):
        raise C.SourceError('REP_TRAIN_EVAL_OVERLAP')
    last = cfg['epochs_max'] if last is None else int(last)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SMPLGraphTCN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=cfg['betas'], eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    n = train_idx.size
    spe = math.ceil(n / cfg['batch'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, spe, cfg))
    y_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    w_t = torch.as_tensor(w_train, dtype=torch.float32, device=device)
    out, losses = ({}, [])
    for epoch in range(1, last + 1):
        model.train()
        perm = rng.permutation(n)
        ep_loss = 0.0
        for a in range(0, n, cfg['batch']):
            b = perm[a:a + cfg['batch']]
            xb, mb = train_batch(store, train_idx[b], rng, device)
            logits = model(xb, mb)
            bt = torch.as_tensor(b, device=device)
            loss = (F.cross_entropy(logits, y_t[bt], reduction='none') * w_t[bt]).sum() / w_t[bt].sum()
            if not torch.isfinite(loss):
                raise C.SourceError('REP_NONFINITE_LOSS')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step()
            sched.step()
            ep_loss += float(loss.detach()) * b.size
        losses.append(ep_loss / n)
        if epoch in snapshots:
            lg = walk_logits(model, prepared, device, cfg['eval_batch'])
            if not np.isfinite(lg).all():
                raise C.SourceError('REP_NONFINITE_LOGITS')
            out[epoch] = lg
    n_par = n_parameters(model)
    del model, opt
    return (out, {'loss_first': losses[0], 'loss_last': losses[-1], 'epochs_run': last, 'n_parameters': n_par})

def select_epoch(inner_f1_by_epoch):
    best = None
    for e in sorted(inner_f1_by_epoch):
        if best is None or inner_f1_by_epoch[e] > inner_f1_by_epoch[best] + Q.TIE_EPS:
            best = e
    return best

def native_outer_fold(store, label, cohort, subject, ci, device, cfg=CONFIG, budget=None, timing=None):
    eligible = np.isin(label, RC.ELIGIBLE)
    tr = np.flatnonzero(eligible & (cohort != ci))
    te = np.flatnonzero(eligible & (cohort == ci))
    if np.intersect1d(subject[tr], subject[te]).size:
        raise C.SourceError('REP_SUBJECT_OVERLAP')
    sources = [int(v) for v in np.unique(cohort[tr])]
    snaps = tuple(cfg['snapshot_epochs'])
    inner_logits = {e: np.zeros((tr.size, 3)) for e in snaps}
    inner_f1 = {e: [] for e in snaps}
    pos = {int(i): k for k, i in enumerate(tr)}
    meta = []
    for slot, v in enumerate(sources):
        a = tr[cohort[tr] != v]
        b = tr[cohort[tr] == v]
        if np.intersect1d(subject[a], subject[b]).size:
            raise C.SourceError('REP_SUBJECT_OVERLAP')
        w = Q.train_weights(label[a], subject[a], 'walk')
        acc = {e: np.zeros((b.size, 3)) for e in snaps}
        prep = prepare_eval(store, b)
        for s in cfg['seeds']:
            t0 = time.monotonic()
            lg, info = train_snapshots(store, a, label[a], w, b, prep, seed_for(ci, slot, s), device, snaps, cfg)
            meta.append(dict(info, role='inner', seconds=round(time.monotonic() - t0, 2)))
            for e in snaps:
                acc[e] += lg[e] / len(cfg['seeds'])
            if budget is not None:
                budget()
        rows = np.array([pos[int(i)] for i in b])
        for e in snaps:
            inner_logits[e][rows] = acc[e]
            inner_f1[e].append(Q.walk_macro_f1(label[b], acc[e].argmax(axis=1)))
    inner_mean = {e: float(np.mean(inner_f1[e])) for e in snaps}
    e_star = select_epoch(inner_mean)
    w_all = Q.train_weights(label[tr], subject[tr], 'walk')
    target = np.zeros((te.size, 3))
    per_seed = []
    del prep
    prep_t = prepare_eval(store, te)
    for s in cfg['seeds']:
        t0 = time.monotonic()
        lg, info = train_snapshots(store, tr, label[tr], w_all, te, prep_t, seed_for(ci, 3, s), device, (e_star,), cfg, last=e_star)
        meta.append(dict(info, role='final', seconds=round(time.monotonic() - t0, 2)))
        target += lg[e_star] / len(cfg['seeds'])
        per_seed.append(lg[e_star])
        if budget is not None:
            budget()
    return {'train_idx': tr, 'test_idx': te, 'e_star': e_star, 'inner_mean_macro_f1': inner_mean, 'inner_site_macro_f1': {e: [float(x) for x in inner_f1[e]] for e in snaps}, 'oof_logits': inner_logits[e_star], 'target_logits': target, 'per_seed_target_logits': per_seed, 'runs': meta}
S_PARITY_TOL = 1.1e-06
ARMS = ('S', 'N', 'S_plus_N', 'S_plus_NOISE3')

def _expected_grade(proba):
    return (np.asarray(proba, dtype=np.float64) * np.arange(3)).sum(axis=1)

def _softmax(z):
    return np.exp(RC.log_softmax(z))

def s_parity(folds_s, point_s, p1, configs, names):
    ref = p1['arms_inductive_strong']['S']['macro_f1']
    rows = {}
    for ci, site in enumerate(names):
        sel = configs[folds_s[ci]['selected']['all']][0]
        want_sel = p1['selected_configs']['S'][site]['all']
        want = ref['per_cohort'][site]['point']
        if sel != want_sel or abs(point_s[ci] - want) > S_PARITY_TOL:
            raise C.SourceError('REP_S_PARITY:' + site)
        rows[site] = {'selected': sel, 'macro_f1': float(point_s[ci]), 'archived': want}
    if abs(float(point_s.mean()) - ref['M']['point']) > S_PARITY_TOL:
        raise C.SourceError('REP_S_PARITY:M')
    return {'status': 'PASS', 'tolerance': S_PARITY_TOL, 'per_site': rows, 'M': float(point_s.mean()), 'archived_M': ref['M']['point']}

def run_campaign(t, store, device, cfg=CONFIG, n_boot=Q.N_BOOT, p1=None, budget=None, configs=Q.CONFIGS, with_partial=True):
    label, cohort, subject = (t['label'], t['cohort'], t['subject'])
    if len(store) != label.size:
        raise C.SourceError('REP_STORE_ROW_ALIGNMENT')
    names = list(C.COHORTS)
    feats = Q.arm_features(t)
    warned = [0]
    folds_s = Q.outer_run(feats['S'], label, cohort, subject, 'walk', configs, warned)
    scorer = Q.Scorer(t, n_boot)
    preds = {a: [] for a in ARMS}
    probas = {a: [] for a in ARMS}
    for ci in range(len(names)):
        p = Q.proba_for(folds_s, ci, policy='all')
        preds['S'].append(p.argmax(axis=1))
        probas['S'].append(p)
    point_s, _ = scorer.stats(scorer.conf(preds['S']))
    parity = s_parity(folds_s, point_s, p1, configs, names) if p1 is not None else {'status': 'NOT_CHECKED'}
    noise = RC.noise_columns(label.size)
    folds = {}
    seed_preds = [[] for _ in cfg['seeds']]
    for ci, site in enumerate(names):
        f = native_outer_fold(store, label, cohort, subject, ci, device, cfg, budget)
        if not np.array_equal(f['test_idx'], np.flatnonzero(scorer.masks[site])):
            raise C.SourceError('REP_TEST_ROWS')
        tr, te = (f['train_idx'], f['test_idx'])
        preds['N'].append(f['target_logits'].argmax(axis=1))
        probas['N'].append(_softmax(f['target_logits']))
        for k, lg in enumerate(f['per_seed_target_logits']):
            seed_preds[k].append(lg.argmax(axis=1))
        Xs_tr, Xs_te = (feats['S'][tr], feats['S'][te])
        p_sn, sel_sn, inner_sn = RC.p1_single_fold(np.column_stack([Xs_tr, RC.log_softmax(f['oof_logits'])]), label[tr], cohort[tr], subject[tr], np.column_stack([Xs_te, RC.log_softmax(f['target_logits'])]), warned, configs)
        p_sz, sel_sz, inner_sz = RC.p1_single_fold(np.column_stack([Xs_tr, noise[tr]]), label[tr], cohort[tr], subject[tr], np.column_stack([Xs_te, noise[te]]), warned, configs)
        for arm, p in (('S_plus_N', p_sn), ('S_plus_NOISE3', p_sz)):
            preds[arm].append(p.argmax(axis=1))
            probas[arm].append(p)
        folds[site] = {'E_star': f['e_star'], 'inner_mean_macro_f1_by_snapshot': f['inner_mean_macro_f1'], 'selected_downstream': {'S': configs[folds_s[ci]['selected']['all']][0], 'S_plus_N': sel_sn, 'S_plus_NOISE3': sel_sz}, 'inner_downstream_selected_mean': {'S_plus_N': float(inner_sn.mean(axis=1).max()), 'S_plus_NOISE3': float(inner_sz.mean(axis=1).max())}, 'n_trainings': len(f['runs']), 'seconds': round(sum((r['seconds'] for r in f['runs'])), 1), 'n_parameters': f['runs'][0]['n_parameters'], 'train_loss_last_mean': float(np.mean([r['loss_last'] for r in f['runs']]))}
        del f
    conf = {a: scorer.conf(preds[a]) for a in ARMS}
    arms = {a: scorer.all_metrics(conf[a]) for a in ARMS}

    def pair(a, b):
        return {'walk_eval': scorer.contrast(conf[a], conf[b]), 'person_weighted_eval': scorer.contrast(conf[a], conf[b], person=True)}
    contrasts = {'S_plus_N_minus_S': pair('S_plus_N', 'S'), 'N_minus_S': pair('N', 'S'), 'S_plus_NOISE3_minus_S': pair('S_plus_NOISE3', 'S'), 'S_plus_N_minus_S_plus_NOISE3': pair('S_plus_N', 'S_plus_NOISE3')}
    decision_sn = RC.decide_comparison(contrasts['S_plus_N_minus_S']['walk_eval'])
    seed_m = []
    for sp in seed_preds:
        pt, _ = scorer.stats(scorer.conf(sp))
        seed_m.append(float(pt.mean()))
    speed_rho = {}
    from scipy import stats as _st
    for ci, site in enumerate(names):
        s_te = t['speed'][scorer.masks[site]]
        speed_rho[site] = {a: float(_st.spearmanr(_expected_grade(probas[a][ci]), s_te).statistic) for a in ('N', 'S_plus_N')}
    body = {'S_parity': parity, 'arms': arms, 'contrasts': contrasts, 'decision': {'S_plus_N_vs_S': decision_sn, 'S_plus_P_vs_S': {'status': 'NOT_RUN_P_BLOCKED'}, 'program': RC.program_decision(decision_sn['status']), 'program_status': RC.PROGRAM_STATUS}, 'pooled_per_grade_recall_suppressed': {a: RC.pooled_recall_rows(conf[a]) for a in ARMS}, 'native_folds': folds, 'native_seed_sensitivity_four_site_macro_f1': {'seeds': list(cfg['seeds']), 'M_by_seed': seed_m, 'range': [min(seed_m), max(seed_m)]}, 'label_free_speed_rank_correlation_of_expected_grade': speed_rho, 'denominators': {site: {'walks_eligible': int(scorer.masks[site].sum()), 'persons_eligible': int(len(scorer.subjects[site]))} for site in names}, 'n_convergence_warnings_downstream': warned[0], 'n_boot': n_boot}
    if with_partial:
        body['speed_conditional_partial_rank_correlation'] = Q.speed_conditional_partial(t, scorer, {a: [_expected_grade(p) for p in probas[a]] for a in ('S', 'N', 'S_plus_N')})
    return body
