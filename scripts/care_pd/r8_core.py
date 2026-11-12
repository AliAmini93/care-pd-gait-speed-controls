"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import common as C
import p0_core as P
import p1_core as Q
SUMMARY_CAP = 262144
WEIGHT_ESTIMANDS = ('A_walk', 'B_cell_reweighted', 'D_person_cell_balanced')
CELL_ESTIMANDS = ('C_cell_mean_proba', 'C_cell_vote')
ESTIMANDS = WEIGHT_ESTIMANDS + CELL_ESTIMANDS

def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return round(v, 6) if np.isfinite(v) else None
    if isinstance(obj, str) or obj is None:
        return obj
    raise TypeError('R8_RESULT_NOT_AGGREGATE_JSON')

def render(obj):
    return json.dumps(clean(obj), indent=1, allow_nan=False) + '\n'

def save_aggregate(path, obj):
    path = Path(path)
    if path.exists():
        raise C.SourceError('R8_RESULT_EXISTS')
    text = render(obj)
    if len(text.encode('utf-8')) > SUMMARY_CAP:
        raise C.SourceError('R8_RESULT_TOO_LARGE')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

def single_outer_fold(X, label, cohort, subject, scheme, configs, warned, ci):
    eligible = np.isin(label, (0, 1, 2))
    members = Q.policy_members(configs)
    tr = eligible & (cohort != ci)
    te = eligible & (cohort == ci)
    scores = Q.inner_scores(X[tr], label[tr], cohort[tr], subject[tr], scheme, configs, warned)
    means = scores.mean(axis=1)
    selected = {name: Q.select_config(means, idx) for name, idx in members.items()}
    wtr = Q.train_weights(label[tr], subject[tr], scheme)
    proba = [Q.fit_predict_proba(c, X[tr], label[tr], wtr, X[te], warned) for c in configs]
    return {'inner': scores, 'selected': selected, 'proba': proba, 'test_mask': te}

def selected_proba(fold, policy='all'):
    return fold['proba'][fold['selected'][policy]]

def person_weighted_macro_f1(y, pred, subject):
    subjects = np.unique(subject)
    cm = P.confusion_by_subject(y, pred, subject, subjects)
    totals = cm.sum(axis=(1, 2), keepdims=True)
    cm = np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)
    return float(P.macro_f1_conf(cm.sum(axis=0)))

def normalized_medication(value):
    if not isinstance(value, str):
        return 'unknown'
    x = value.strip().upper()
    return x if x in ('ON', 'OFF') else 'unknown'

def build_cells(subject, state, label):
    subject = np.asarray(subject)
    label = np.asarray(label)
    state = list(state)
    if subject.ndim != 1 or subject.size == 0 or len(state) != subject.size or (label.shape != subject.shape):
        raise C.SourceError('R8_CELL_INPUT')
    if any((s not in ('ON', 'OFF') for s in state)):
        raise C.SourceError('R8_CELL_STATE_UNRECOGNIZED')
    keys = sorted(set(zip(subject.tolist(), state)))
    index = {k: i for i, k in enumerate(keys)}
    cell = np.array([index[s, st] for s, st in zip(subject.tolist(), state)], dtype=int)
    n_cells = len(keys)
    grade = np.full(n_cells, -1, dtype=int)
    size = np.zeros(n_cells, dtype=int)
    for c in range(n_cells):
        m = cell == c
        g = np.unique(label[m])
        if g.size != 1:
            raise C.SourceError('R8_CELL_MIXED_GRADE')
        grade[c] = int(g[0])
        size[c] = int(m.sum())
    cell_person = np.array([k[0] for k in keys])
    cell_state = np.array([k[1] for k in keys], dtype=object)
    persons = np.unique(subject)
    pos = {p: i for i, p in enumerate(persons.tolist())}
    cell_person_idx = np.array([pos[p] for p in cell_person.tolist()], dtype=int)
    cells_per_person = np.bincount(cell_person_idx, minlength=len(persons))
    return {'cell': cell, 'n_cells': n_cells, 'grade': grade, 'size': size, 'person_idx': cell_person_idx, 'state': cell_state, 'persons': persons, 'cells_per_person': cells_per_person}

def cell_structure(cells, label):
    persons, cpp = (cells['persons'], cells['cells_per_person'])
    two = np.flatnonzero(cpp == 2)
    discordant = 0
    for i in two:
        g = cells['grade'][cells['person_idx'] == i]
        discordant += int(g[0] != g[1])
    size = cells['size']
    return {'n_people': int(len(persons)), 'n_walks': int(cells['cell'].size), 'n_cells': int(cells['n_cells']), 'n_two_state_people': int(two.size), 'n_single_state_people': int((cpp == 1).sum()), 'n_people_with_more_than_two_cells': int((cpp > 2).sum()), 'n_discordant_two_state_people': int(discordant), 'state_walks': {s: int((cells['state'][cells['cell']] == s).sum()) for s in ('ON', 'OFF')}, 'cells_per_state': {s: int((cells['state'] == s).sum()) for s in ('ON', 'OFF')}, 'cells_per_grade': [int((cells['grade'] == k).sum()) for k in range(3)], 'walks_per_grade': [int((np.asarray(label) == k).sum()) for k in range(3)], 'cell_size': {'min': int(size.min()), 'q25': float(np.percentile(size, 25)), 'median': float(np.median(size)), 'q75': float(np.percentile(size, 75)), 'max': int(size.max()), 'mean': float(size.mean())}}

def estimand_weights(cells, kind):
    cell = cells['cell']
    n = cells['size'][cell].astype(float)
    if kind == 'A_walk':
        return np.ones(cell.size)
    if kind == 'B_cell_reweighted':
        return 1.0 / n
    if kind == 'D_person_cell_balanced':
        m = cells['cells_per_person'][cells['person_idx'][cell]].astype(float)
        return 1.0 / (n * m)
    raise ValueError(kind)

def cell_predictions(proba, cells):
    if proba.ndim != 2 or proba.shape != (cells['cell'].size, 3):
        raise C.SourceError('R8_PROBA_SHAPE')
    n = cells['n_cells']
    mean = np.zeros((n, 3))
    np.add.at(mean, cells['cell'], proba)
    mean /= cells['size'][:, None]
    vote = np.zeros((n, 3))
    np.add.at(vote, (cells['cell'], proba.argmax(axis=1)), 1.0)
    return {'mean_proba': mean, 'pred_mean': mean.argmax(axis=1), 'pred_vote': vote.argmax(axis=1), 'q_cell': mean @ np.arange(3)}

def weighted_confusion(y, pred, w):
    conf = np.zeros((3, 3))
    np.add.at(conf, (np.asarray(y, dtype=int), np.asarray(pred, dtype=int)), np.asarray(w, dtype=float))
    return conf

def per_class_f1(conf):
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    denom = conf.sum(axis=-1) + conf.sum(axis=-2)
    return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)

def ordinal_mae(conf):
    total = conf.sum(axis=(-2, -1))
    dist = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
    return np.divide((conf * dist).sum(axis=(-2, -1)), total, out=np.full_like(total, np.nan, dtype=float), where=total > 0)
METRICS = {'macro_f1': P.macro_f1_conf, 'balanced_accuracy': Q.balanced_accuracy_conf, 'ordinal_mae': ordinal_mae}

def recall_identity(y, pred, cells):
    correct = (np.asarray(y) == np.asarray(pred)).astype(float)
    n_c = cells['size'].astype(float)
    c_c = np.zeros(cells['n_cells'])
    np.add.at(c_c, cells['cell'], correct)
    c_c /= n_c
    out = {}
    for k in range(3):
        m = cells['grade'] == k
        if not m.any():
            out[str(k)] = None
            continue
        n, c = (n_c[m], c_c[m])
        rw = float((n * c).sum() / n.sum())
        ro = float(c.mean())
        cov = float((n * c).mean() - n.mean() * c.mean())
        term = cov / float(n.mean())
        out[str(k)] = {'n_cells': int(m.sum()), 'recall_walk_weighted': rw, 'recall_cell_balanced': ro, 'cov_term': term, 'identity_residual': rw - ro - term, 'spearman_size_vs_cell_correctness': _spearman(n, c)}
    return out

def _spearman(a, b):
    a, b = (np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    rho = stats.spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else None

def rank_metrics(q, grade):
    out = {'spearman': _spearman(q, grade)}
    grade = np.asarray(grade)
    for thr in (1, 2):
        binary = grade >= thr
        out[f'auc_grade_ge_{thr}'] = float(roc_auc_score(binary, q)) if np.unique(binary).size == 2 else None
    return out

def person_confusions(y, pred, w, subject, persons):
    pos = {p: i for i, p in enumerate(persons.tolist())}
    idx = np.array([pos[s] for s in np.asarray(subject).tolist()], dtype=int)
    out = np.zeros((len(persons), 3, 3))
    np.add.at(out, (idx, np.asarray(y, dtype=int), np.asarray(pred, dtype=int)), np.asarray(w, dtype=float))
    return out

def cell_person_confusions(cells, pred_cell):
    out = np.zeros((len(cells['persons']), 3, 3))
    np.add.at(out, (cells['person_idx'], cells['grade'], np.asarray(pred_cell, dtype=int)), 1.0)
    return out

def draw_person_counts(n_persons, n_boot, seed):
    return P.draw_counts([n_persons], n_boot, seed)[0]

def boot_from_person_conf(person_conf, counts):
    pooled = np.einsum('bs,sij->bij', counts, person_conf)
    absent = (pooled.sum(axis=2) == 0).any(axis=1)
    return {m: fn(pooled) for m, fn in METRICS.items()} | {'absent_true_class_draws': int(absent.sum())}

def boot_cell_rank(q_cell, grade, cells, counts):
    rows = {'spearman': [], 'auc_grade_ge_1': [], 'auc_grade_ge_2': []}
    for b in range(counts.shape[0]):
        idx = np.repeat(np.arange(cells['n_cells']), counts[b][cells['person_idx']])
        r = rank_metrics(q_cell[idx], grade[idx])
        for k in rows:
            rows[k].append(np.nan if r[k] is None else r[k])
    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}

def summary(point, draws):
    draws = np.asarray(draws, dtype=float)
    ok = np.isfinite(draws)
    if point is None or not np.isfinite(point) or ok.sum() < 2:
        return {'point': None if point is None or not np.isfinite(point) else float(point), 'ci95': None, 'n_undefined_draws': int((~ok).sum())}
    return {'point': float(point), 'ci95': P.ci95(draws[ok]), 'n_undefined_draws': int((~ok).sum())}
