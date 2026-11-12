"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import json
import os
import platform
import sys
from pathlib import Path
import numpy as np
import scipy
import sklearn
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import p0_core as P
import p1_core as Q
import r8_core as R
CARD_ID = 'a_occasion_grain'
STATUS = 'R8_A_OCCASION_GRAIN_COMPLETE'
SITE = 'BMCLab'
ARMS = ('S', 'PS', 'KS')
SEED_BOOT = 20261103
N_BOOT = 1000
TOL = 1.1e-06
EXPECTED = {'n_people': 23, 'n_walks': 781, 'n_cells': 42, 'n_two_state_people': 19, 'n_single_state_people': 4, 'n_people_with_more_than_two_cells': 0, 'n_discordant_two_state_people': 8, 'state_walks': {'ON': 410, 'OFF': 371}}
LIMITS = ['Same frozen P1 predictions; no model, feature, threshold or aggregation choice was made on BMCLab labels.', 'BMCLab was inspected in P0/P1/R1C/R3C/R3T/R4W: exploratory, not an independent site result.', 'Cells are person x medication session (live examination at session start, trials recorded afterwards); the ON/OFF state is a cell key only, not a treatment-effect estimand.', 'The BMCLab instrument (original UPDRS item 29 vs MDS-UPDRS 3.10) is unresolved (R6); no rubric is harmonised.', 'Estimands A/B/C/D are different quantities; none is an accuracy corrected for annotation bias.', 'Person-cluster bootstrap intervals (23 people) describe within-site resampling only.', 'Cell-balanced and cell-aggregated evaluation are known informative-cluster-size and bag-pooling techniques; no novelty is claimed.']

def collect_medication(obj):
    med = []
    for _, rec in C.walks_of(obj):
        pose, trans = C.read_arrays(rec)
        if not (np.isfinite(pose).all() and np.isfinite(trans).all()):
            continue
        med.append(R.normalized_medication(rec.get('medication')))
    return med

def check_structure(structure):
    for key, want in EXPECTED.items():
        if structure[key] != want:
            raise C.SourceError('R8_A_STRUCTURE:' + key)

def check_parity(t, folds, p1, te):
    y, subj = (t['label'][te], t['subject'][te])
    out = {}
    for arm in ARMS:
        fold = folds[arm]
        name = Q.CONFIGS[fold['selected']['all']][0]
        want_name = p1['selected_configs'][arm][SITE]['all']
        if name != want_name:
            raise C.SourceError('R8_A_PARITY_CONFIG:' + arm)
        pred = R.selected_proba(fold).argmax(axis=1)
        archived = p1['arms_inductive_strong'][arm]
        got = {'macro_f1': Q.walk_macro_f1(y, pred), 'ordinal_mae': float(Q.ordinal_mae_conf(R.weighted_confusion(y, pred, np.ones(y.size)))), 'macro_f1_person_weighted_eval': R.person_weighted_macro_f1(y, pred, subj)}
        want = {k: archived[k]['per_cohort'][SITE]['point'] for k in got}
        for k in got:
            if abs(got[k] - want[k]) > TOL:
                raise C.SourceError('R8_A_PARITY_VALUE:' + arm + ':' + k)
        out[arm] = {'selected_config': name, 'refit': got, 'archived': want, 'tolerance': TOL}
    return out

def analyze(y, subj, med, probas, n_boot=N_BOOT, seed=SEED_BOOT):
    cells = R.build_cells(subj, med, y)
    structure = R.cell_structure(cells, y)
    check_structure(structure)
    persons = cells['persons']
    counts = R.draw_person_counts(len(persons), n_boot, seed)
    draws = {}
    est_out = {}
    rank_out = {}
    identity = {}
    walk_rank = {}
    for arm in ARMS:
        proba = probas[arm]
        pred = proba.argmax(axis=1)
        cp = R.cell_predictions(proba, cells)
        draws[arm], est_out[arm] = ({}, {})
        for kind in R.WEIGHT_ESTIMANDS:
            w = R.estimand_weights(cells, kind)
            pc = R.person_confusions(y, pred, w, subj, persons)
            point_conf = pc.sum(axis=0)
            boot = R.boot_from_person_conf(pc, counts)
            draws[arm][kind] = boot
            est_out[arm][kind] = {m: R.summary(fn(point_conf), boot[m]) for m, fn in R.METRICS.items()} | {'per_class_f1': R.per_class_f1(point_conf), 'confusion_weighted': point_conf, 'total_weight': float(point_conf.sum()), 'absent_true_class_draws': boot['absent_true_class_draws']}
        for kind, key in (('C_cell_mean_proba', 'pred_mean'), ('C_cell_vote', 'pred_vote')):
            pc = R.cell_person_confusions(cells, cp[key])
            point_conf = pc.sum(axis=0)
            boot = R.boot_from_person_conf(pc, counts)
            draws[arm][kind] = boot
            est_out[arm][kind] = {m: R.summary(fn(point_conf), boot[m]) for m, fn in R.METRICS.items()} | {'per_class_f1': R.per_class_f1(point_conf), 'confusion_cells': point_conf, 'total_weight': float(point_conf.sum()), 'absent_true_class_draws': boot['absent_true_class_draws']}
        rank_point = R.rank_metrics(cp['q_cell'], cells['grade'])
        rank_boot = R.boot_cell_rank(cp['q_cell'], cells['grade'], cells, counts)
        rank_out[arm] = {k: R.summary(rank_point[k], rank_boot[k]) for k in rank_point}
        identity[arm] = R.recall_identity(y, pred, cells)
        walk_rank[arm] = R.rank_metrics(proba @ np.arange(3), y)
    grain_shift, contrasts = ({}, {})
    for arm in ARMS:
        grain_shift[arm] = {}
        for kind in R.ESTIMANDS:
            if kind == 'A_walk':
                continue
            grain_shift[arm][kind + '_minus_A_walk'] = {m: R.summary(est_out[arm][kind][m]['point'] - est_out[arm]['A_walk'][m]['point'], draws[arm][kind][m] - draws[arm]['A_walk'][m]) for m in R.METRICS}
    for a, b in (('PS', 'S'), ('KS', 'S')):
        contrasts[f'{a}_minus_{b}'] = {kind: {m: R.summary(est_out[a][kind][m]['point'] - est_out[b][kind][m]['point'], draws[a][kind][m] - draws[b][kind][m]) for m in R.METRICS} for kind in R.ESTIMANDS}
    return {'bmclab_structure': structure, 'estimands': est_out, 'grain_shift_within_arm': grain_shift, 'contrasts_between_arms': contrasts, 'cell_level_rank': rank_out, 'walk_level_rank_point': walk_rank, 'recall_identity_by_grade': identity, 'bootstrap': {'unit': 'person (all cells and walks carried)', 'n_people': int(len(persons)), 'seed': seed, 'n_draws': n_boot}}
