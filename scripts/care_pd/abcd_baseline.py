"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import p1_core as Q
CONFIGS = Q.CONFIGS
SCHEME = 'walk'
CROSSFIT_FOLDS = 5
OFFSET_EPS = 0.001
ELIGIBLE = (0, 1, 2)

class BaselineError(C.SourceError):
    pass

def _check(X_src, y_src, site_src, subj_src, X_tgt):
    if X_src.ndim != 2 or X_tgt.ndim != 2 or X_src.shape[1] != X_tgt.shape[1]:
        raise BaselineError('ABCD_S_SHAPE')
    if not X_src.shape[0] == y_src.size == site_src.size == subj_src.size:
        raise BaselineError('ABCD_S_ROWS')
    if not np.isin(y_src, ELIGIBLE).all():
        raise BaselineError('ABCD_S_LABELS')
    if np.unique(site_src).size < 2:
        raise BaselineError('ABCD_S_NEEDS_TWO_SOURCE_SITES')
    if not (np.isfinite(X_src).all() and np.isfinite(X_tgt).all()):
        raise BaselineError('ABCD_S_NONFINITE')

def s_fold(X_src, y_src, site_src, subj_src, X_tgt, warned, configs=CONFIGS):
    _check(X_src, y_src, site_src, subj_src, X_tgt)
    scores = Q.inner_scores(X_src, y_src, site_src, subj_src, SCHEME, configs, warned)
    best = Q.select_config(scores.mean(axis=1), list(range(len(configs))))
    w = Q.train_weights(y_src, subj_src, SCHEME)
    proba = Q.fit_predict_proba(configs[best], X_src, y_src, w, X_tgt, warned)
    return {'index': int(best), 'name': configs[best][0], 'inner': scores, 'proba': proba}

def fit_proba(config, X_tr, y_tr, subj_tr, X_te, warned):
    return Q.fit_predict_proba(config, X_tr, y_tr, Q.train_weights(y_tr, subj_tr, SCHEME), X_te, warned)

def crossfit_proba(config, X, y, subj, warned, k=CROSSFIT_FOLDS):
    n_groups = np.unique(subj).size
    if n_groups < 2:
        raise BaselineError('ABCD_CROSSFIT_GROUPS')
    oof = np.full((y.size, 3), np.nan)
    for tr, va in GroupKFold(n_splits=min(k, n_groups)).split(X, y, subj):
        if np.intersect1d(subj[tr], subj[va]).size:
            raise BaselineError('ABCD_CROSSFIT_PERSON_OVERLAP')
        oof[va] = fit_proba(config, X[tr], y[tr], subj[tr], X[va], warned)
    if not np.isfinite(oof).all():
        raise BaselineError('ABCD_CROSSFIT_INCOMPLETE')
    return oof

def offsets(proba):
    p = np.asarray(proba, dtype=np.float64)
    return np.log(p * (1.0 - 3.0 * OFFSET_EPS) + OFFSET_EPS)

def entropy(proba):
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum(axis=1).mean())
