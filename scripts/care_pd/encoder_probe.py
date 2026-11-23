"""Portable analysis kernel. See docs/provenance_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import aggregate_core as IC
import encoder_inputs as F
import encoder_precheck as K
N_BOOT = 2000
MARGIN = 0.05
ARCHIVED_S_M = None
ARCHIVED_COUNTS = {'walks': 2908, 'participants': 107}
PARITY_TOL = 1.1e-06
ENCODER_ARMS = ('E', 'E_noroot', 'E_rootonly')
VARIANT_OF = {'E': 'full', 'E_noroot': 'no_root', 'E_rootonly': 'root_only'}
ARMS = ('S_in', 'S_v4', 'E', 'E_noroot', 'E_rootonly', 'S_in_plus_E')
PRIMARY = ('E', 'S_in')
ORIGINAL_PRIMARY = ('E', 'S_v4')
CONTRASTS = (PRIMARY, ('S_in_plus_E', 'S_in'), ORIGINAL_PRIMARY, ('E', 'E_noroot'), ('E_rootonly', 'E_noroot'), ('E_rootonly', 'S_in'), ('S_in', 'S_v4'))
C2_PASSING = ('BMCLab', 'PD-GaM', 'T-SDU-PD')
WITHIN_FOLDS = 5
M_MIN_FRAMES = 8

def outer_run_foldwise(P1, x_by_fold, label, cohort, subject, scheme, configs, warned, n_cohorts):
    eligible = np.isin(label, (0, 1, 2))
    members = P1.policy_members(configs)
    folds = []
    for ci in range(n_cohorts):
        X = x_by_fold[ci]
        tr = eligible & (cohort != ci)
        te = eligible & (cohort == ci)
        scores = P1.inner_scores(X[tr], label[tr], cohort[tr], subject[tr], scheme, configs, warned)
        means = scores.mean(axis=1)
        selected = {name: P1.select_config(means, idx) for name, idx in members.items()}
        wtr = P1.train_weights(label[tr], subject[tr], scheme)
        proba = [P1.fit_predict_proba(c, X[tr], label[tr], wtr, X[te], warned) for c in configs]
        folds.append({'inner': scores, 'selected': selected, 'proba': proba, 'test_mask': te})
    return folds

def pair_tables(y, score, subject, subjects):
    pos = {s: i for i, s in enumerate(subjects)}
    idx = np.array([pos[s] for s in subject])
    dy = np.sign(y[:, None].astype(float) - y[None, :].astype(float))
    ds = np.sign(score[:, None] - score[None, :])
    informative = (dy != 0) & (idx[:, None] != idx[None, :])
    credit = np.where(ds == dy, 1.0, np.where(ds == 0, 0.5, 0.0)) * informative
    n = len(subjects)
    flat = idx[:, None] * n + idx[None, :]
    conc = np.bincount(flat.ravel(), weights=credit.ravel(), minlength=n * n).reshape(n, n)
    tot = np.bincount(flat.ravel(), weights=informative.ravel().astype(float), minlength=n * n).reshape(n, n)
    return (conc, tot)

def concordance(conc, tot, weights=None):
    if weights is None:
        t = tot.sum()
        return float(conc.sum() / t) if t > 0 else float('nan')
    num = np.einsum('ba,ac,bc->b', weights, conc, weights)
    den = np.einsum('ba,ac,bc->b', weights, tot, weights)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)

def verdict(delta, lo, hi, per_cohort, margin=MARGIN):
    if delta >= margin and lo > 0 and (sum((d >= 0 for d in per_cohort)) >= 3):
        return 'BEYOND'
    if hi < 0:
        return 'BELOW'
    return 'NO_DETECTED_DIFFERENCE'

def dependence_verdict(delta, lo, hi, margin=MARGIN):
    if delta >= margin and lo > 0:
        return 'ROOT_CHANNEL_DEPENDENT'
    if lo > -margin and hi < margin:
        return 'NOT_ROOT_CHANNEL_DEPENDENT'
    return 'INCONCLUSIVE'

def summary(point, draws):
    draws = np.asarray(draws, float)
    good = draws[np.isfinite(draws)]
    lo, hi = np.percentile(good, [2.5, 97.5]) if good.size else (float('nan'), float('nan'))
    return {'point': round(float(point), 6), 'ci95': [round(float(lo), 6), round(float(hi), 6)], 'draws_undefined': int(draws.size - good.size)}

def block(names, point, draws):
    keep = [c for c, n in enumerate(names) if n in C2_PASSING]
    return {'per_cohort': {n: summary(point[c], draws[:, c]) for c, n in enumerate(names)}, 'M': summary(float(np.mean(point)), draws.mean(axis=1)), 'M_three_c2_passing_cohorts': summary(float(np.mean(point[keep])), draws[:, keep].mean(axis=1))}

def input_fps(fps_v4, frames_npz, frames_v4):
    return float(fps_v4) * (frames_npz + 1) / float(frames_v4)

def grouped_folds(subject, k):
    order = {s: i % k for i, s in enumerate(np.unique(subject))}
    return np.array([order[s] for s in subject])

def load_tables(C, P):
    label, cohort, subject, s_old, s_in, raws = ([], [], [], [], [], [])
    counts, next_subject = ({}, 0)
    for ci, name in enumerate(C.COHORTS):
        obj = C.load_cohort(name)
        index = K.canonical_index(obj)
        seen = {}
        n_grade3 = n_unlabelled = n_unusable = 0
        with np.load(F.dv_path(name), allow_pickle=False) as npz:
            chosen, _ = K.join(sorted(npz.files), index)
            for subj_key, walks in obj.items():
                subj = str(subj_key)
                for walk, rec in walks.items():
                    if (subj, walk) not in chosen:
                        continue
                    key = chosen[subj, walk][0]
                    y = C.read_label(rec['UPDRS_GAIT'])
                    if y is None:
                        n_unlabelled += 1
                        continue
                    if y == 3:
                        n_grade3 += 1
                        continue
                    raw = K.first_view(np.asarray(npz[key]))
                    if raw is None or not np.isfinite(raw).all() or raw.shape[0] < M_MIN_FRAMES:
                        n_unusable += 1
                        continue
                    frames, fps, trans = index[subj, walk]
                    if subj not in seen:
                        seen[subj] = next_subject
                        next_subject += 1
                    label.append(y)
                    cohort.append(ci)
                    subject.append(seen[subj])
                    s_old.append(P.speed_and_duration(trans, fps)[0])
                    s_in.append(K.speed_from_humanml(raw, input_fps(fps, raw.shape[0], frames)))
                    raws.append(np.ascontiguousarray(raw, dtype=np.float32))
        counts[name] = {'n_walks_v4': IC.count(len(index)), 'n_walks_joined': IC.count(len(chosen)), 'n_grade3_excluded': IC.count(n_grade3), 'n_unlabelled': IC.count(n_unlabelled), 'n_unusable_input': IC.count(n_unusable)}
    t = {'label': np.array(label), 'cohort': np.array(cohort), 'subject': np.array(subject), 'S_v4': np.array(s_old, float), 'S_in': np.array(s_in, float)}
    return (t, raws, counts)

def encode_all(M, model, raws, mean, std, mode):
    out = np.empty((len(raws), 512))
    for i, raw in enumerate(raws):
        emb = M.walk_embedding(model, M.variant(M.normalize(raw, mean, std), mode))
        if emb is None:
            IC.stop('a walk shorter than the encoder minimum reached the encoder')
        out[i] = emb
    return out

def analyse(meta, t, raws, counts, model, stats, P1, M, names, n_boot=N_BOOT, configs=None):
    configs = P1.CONFIGS if configs is None else configs
    y, coh, subj = (t['label'], t['cohort'], t['subject'])
    n_walks, n_part = (int(y.size), int(np.unique(subj).size))
    feats = {arm: [] for arm in ENCODER_ARMS}
    for ci, name in enumerate(names):
        mean, std = stats(f'LODO_{name}')
        for arm in ENCODER_ARMS:
            feats[arm].append(encode_all(M, model, raws, mean, std, VARIANT_OF[arm]))
        print(f'PROGRESS encoded fold {ci}')
    x_by_fold = {'S_in': [t['S_in'][:, None]] * len(names), 'S_v4': [t['S_v4'][:, None]] * len(names), 'E': feats['E'], 'E_noroot': feats['E_noroot'], 'E_rootonly': feats['E_rootonly'], 'S_in_plus_E': [np.column_stack([feats['E'][ci], t['S_in']]) for ci in range(len(names))]}
    warned = [0]
    folds = {arm: outer_run_foldwise(P1, x_by_fold[arm], y, coh, subj, 'walk', configs, warned, len(names)) for arm in ARMS}
    print('PROGRESS models fitted')
    scorer = P1.Scorer(t, n_boot)
    pred = lambda arm, **kw: [P1.proba_for(folds[arm], ci, **kw).argmax(axis=1) for ci in range(len(names))]
    conf = {arm: scorer.conf(pred(arm, policy='all')) for arm in ARMS}
    stat = {arm: scorer.stats(conf[arm]) for arm in ARMS}
    arms_out = {}
    for arm in ARMS:
        point, draws = stat[arm]
        per_config = []
        for k in range(len(configs)):
            pk, _dk = scorer.stats(scorer.conf(pred(arm, config_index=k)))
            per_config.append(round(float(pk.mean()), 6))
        arms_out[arm] = {'macro_f1': block(names, point, draws), 'selected_config': {n: configs[folds[arm][ci]['selected']['all']][0] for ci, n in enumerate(names)}, 'M_by_fixed_config': dict(zip([c[0] for c in configs], per_config)), 'M_fixed_config_min': min(per_config), 'M_fixed_config_max': max(per_config)}
    contrasts = {}
    for a, b in CONTRASTS:
        pa, da = stat[a]
        pb, db = stat[b]
        blk = block(names, pa - pb, da - db)
        blk['M_fixed_logistic_LR_C1'] = round(arms_out[a]['M_by_fixed_config']['LR_C1'] - arms_out[b]['M_by_fixed_config']['LR_C1'], 6)
        contrasts[f'{a}_minus_{b}'] = blk
    grades = np.arange(3.0)
    conc_out, conc_stat = ({}, {})
    for arm in ARMS:
        point, draws = (np.empty(len(names)), np.empty((n_boot, len(names))))
        for ci, n in enumerate(names):
            m = scorer.masks[n]
            score = P1.proba_for(folds[arm], ci, policy='all') @ grades
            cmat, tmat = pair_tables(scorer.y[n], score, scorer.subject[n], scorer.subjects[n])
            point[ci] = concordance(cmat, tmat)
            draws[:, ci] = concordance(cmat, tmat, scorer.counts[ci].astype(float))
        conc_stat[arm] = (point, draws)
        conc_out[arm] = block(names, point, draws)
    conc_contrasts = {f'{a}_minus_{b}': block(names, conc_stat[a][0] - conc_stat[b][0], conc_stat[a][1] - conc_stat[b][1]) for a, b in CONTRASTS}
    within = {}
    lr = configs[0]
    for ci, n in enumerate(names):
        m = coh == ci
        mean, std = stats(n)
        e_own = encode_all(M, model, [raws[i] for i in np.flatnonzero(m)], mean, std, 'full')
        fold_of = grouped_folds(subj[m], WITHIN_FOLDS)
        res = {}
        for arm, X in (('S_in', t['S_in'][m][:, None]), ('E', e_own)):
            p = np.zeros(int(m.sum()), int)
            for k in range(WITHIN_FOLDS):
                tr, te = (fold_of != k, fold_of == k)
                if te.any() and np.unique(y[m][tr]).size >= 2:
                    w = P1.train_weights(y[m][tr], subj[m][tr], 'walk')
                    p[te] = P1.fit_predict_proba(lr, X[tr], y[m][tr], w, X[te], warned).argmax(axis=1)
            res[arm] = round(P1.walk_macro_f1(y[m], p), 6)
        within[n] = res
    within['M'] = {arm: round(float(np.mean([within[n][arm] for n in names])), 6) for arm in ('S_in', 'E')}
    d = lambda key: contrasts[key]['M']
    per = lambda key: [contrasts[key]['per_cohort'][n]['point'] for n in names]
    key_of = lambda pair: f'{pair[0]}_minus_{pair[1]}'
    v_primary = verdict(d(key_of(PRIMARY))['point'], *d(key_of(PRIMARY))['ci95'], per(key_of(PRIMARY)))
    v_original = verdict(d(key_of(ORIGINAL_PRIMARY))['point'], *d(key_of(ORIGINAL_PRIMARY))['ci95'], per(key_of(ORIGINAL_PRIMARY)))
    v_fusion = verdict(d('S_in_plus_E_minus_S_in')['point'], *d('S_in_plus_E_minus_S_in')['ci95'], per('S_in_plus_E_minus_S_in'))
    v_dep = dependence_verdict(d('E_minus_E_noroot')['point'], *d('E_minus_E_noroot')['ci95'])
    s_m = arms_out['S_v4']['macro_f1']['M']['point']
    same_set = n_walks == ARCHIVED_COUNTS['walks'] and n_part == ARCHIVED_COUNTS['participants']
    floor_point, floor_draws = scorer.floor_stats()
    out = dict(meta)
    out.update({'status': f'C3X_ENCODER_{v_primary}_INPUT_SPEED', 'verdicts': {'primary_E_vs_S_in': v_primary, 'original_primary_E_vs_S_v4': v_original, 'primary_and_original_agree': bool(v_primary == v_original), 'fusion_S_in_plus_E_vs_S_in': v_fusion, 'root_channel_ablation': v_dep}, 'design': {'label': 'EXPLORATORY-ADAPTIVE; amended after the failed label-free gate c2; reused cohorts', 'protocol': 'source-only leave-one-cohort-out', 'selection': 'frozen GAIT-FLOOR P1 six candidates, inner leave-one-source-cohort-out, policy all', 'bootstrap': 'participant-cluster within held-out cohort, predictions fixed', 'n_boot': int(n_boot), 'margin': MARGIN, 'encoder': 'MoMask RVQ-VAE, official CARE-PD wiring, eval mode', 'encoder_input': 'Dataverse v1.3 HumanML3D files, official LODO normalization of the held-out fold', 'checkpoint_provenance': 'official repository file; identity with the original MoMask release UNVERIFIED', 'gate_c2': 'amended after a label-free pre-check failed in the original analysis; run encoder-precheck for the local result; no historical value is bundled'}, 'n_walks_eligible': IC.count(n_walks), 'n_participants_eligible': IC.count(n_part), 'cohort_counts': counts, 'class_walks': {n: {f'n_grade{k}': IC.count(int(((coh == ci) & (y == k)).sum())) for k in range(3)} for ci, n in enumerate(names)}, 'cohort_participants': {n: {'n_participants': IC.count(len(scorer.subjects[n]))} for n in names}, 'uniform_floor': block(names, floor_point, floor_draws), 'arms': arms_out, 'contrasts': contrasts, 'between_participant_concordance': conc_out, 'concordance_contrasts': conc_contrasts, 'within_cohort_reference_fixed_LR': within, 'speed_arm_parity': {'walk_set_equals_archived': bool(same_set), 'archived_S_M': ARCHIVED_S_M, 'S_v4_M': s_m, 'S_v4_M_minus_archived': round(s_m - ARCHIVED_S_M, 6), 'exact_parity': bool(same_set and abs(s_m - ARCHIVED_S_M) <= PARITY_TOL)}, 'convergence_warnings': int(warned[0])})
    return out
