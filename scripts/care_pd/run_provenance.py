#!/usr/bin/env python3
"""Portable command-line orchestration of the data-provenance and frozen-encoder probe kernels.

Commands, in the order of the original analyses (docs/provenance_protocol.md):

  fetch-encoder-inputs          pinned v1.3 HumanML3D inputs, MoMask checkpoint, 16 normalization files
  encoder-precheck              label-free comparability pre-check of those inputs; no model fitted
  encoder-probe                 source-only frozen-encoder probe against input speed; needs P0 and P1
  fetch-originals --first|--rest   pinned original annotated SMPL files, strict safe load
  root-parity --first|--rest       record-level label and trajectory parity; no model fitted
  root-speed                    original-trajectory speed sensitivity of the P1 contrast; needs P0 and P1

The two fetch commands do nothing without --accept-care-pd-terms and contact only the pinned hosts.
Every other command blocks the network before reading data. Each command writes one aggregate JSON
file to --out-dir and never overwrites it. Exit codes: 0 complete, 2 usage error or terms not
accepted, 10 existing result, 20 input, environment, gate or parity stop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import aggregate_core as IC  # noqa: E402
import common as C  # noqa: E402
import release_context as X  # noqa: E402

COMMANDS = ('fetch-encoder-inputs', 'encoder-precheck', 'encoder-probe',
            'fetch-originals', 'root-parity', 'root-speed')
STAGED = ('fetch-originals', 'root-parity')
STAGES = {'first': ('3DGait',), 'rest': ('BMCLab', 'T-SDU-PD', 'PD-GaM')}
MODULE = {'fetch-encoder-inputs': 'encoder_inputs', 'encoder-precheck': 'encoder_precheck',
          'encoder-probe': 'encoder_probe', 'fetch-originals': 'root_fetch', 'root-parity': 'root_parity',
          'root-speed': 'root_speed_sensitivity'}
# Seeds and protocol classes of the original runs. Only the probe seeds torch; inference is deterministic.
SEED = {'fetch-encoder-inputs': 20261112, 'encoder-precheck': 20261112, 'encoder-probe': 20261112,
        'fetch-originals': 20261113, 'root-parity': 20261113, 'root-speed': 20261030}
PROTOCOL_CLASS = {'fetch-encoder-inputs': 'DOWNLOAD', 'encoder-precheck': 'METADATA', 'encoder-probe': 'SCORED',
                  'fetch-originals': 'DOWNLOAD', 'root-parity': 'METADATA', 'root-speed': 'SCORED'}
TERMS_URL = 'https://neurips2025.care-pd.ca/terms-of-use'
DATAVERSE_URL = 'https://doi.org/10.5683/SP3/TWIKMK'
CODE_URL = 'https://github.com/TaatiTeam/CARE-PD'
SCOPE = 'EXPLORATORY_NONCLINICAL_PORTABLE_RERUN'
E2_UNAVAILABLE = {'status': 'NOT_AVAILABLE_IN_PORTABLE_RELEASE',
                  'reason': 'needs a separately archived same-definition speed aggregate that is not distributed'}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=COMMANDS)
    stage = p.add_mutually_exclusive_group()
    stage.add_argument('--first', action='store_true', help='fetch-originals, root-parity: 3DGait only')
    stage.add_argument('--rest', action='store_true',
                       help='fetch-originals, root-parity: BMCLab, T-SDU-PD and PD-GaM, after the first stage')
    p.add_argument('--data-dir', type=Path, required=True,
                   help='data folder outside this repository: the four pinned *_canonical.pkl files, '
                        'plus original/, humanml3d_v1.3/ and care_pd_9bc22b0/ written by the fetch commands')
    p.add_argument('--out-dir', type=Path, required=True, help='local aggregate folder outside this repository')
    p.add_argument('--p0-result', type=Path, help='encoder-probe, root-speed: default <out-dir>/p0.json')
    p.add_argument('--p1-result', type=Path, help='encoder-probe, root-speed: default <out-dir>/p1_strong_models.json')
    p.add_argument('--accept-care-pd-terms', action='store_true',
                   help=f'fetch commands only: you have read and accept {TERMS_URL}')
    a = p.parse_args(argv)
    if a.command in STAGED and not (a.first or a.rest):
        p.error(a.command + ' needs --first or --rest')
    if a.command not in STAGED and (a.first or a.rest):
        p.error('--first and --rest apply only to ' + ' and '.join(STAGED))
    return a


def result_name(command, stage=None):
    return command.replace('-', '_') + ('_' + stage if stage else '') + '.json'


def configure(data_dir):
    """Portable locations of the fetched inputs; the kernels read these module attributes."""
    import encoder_inputs as F
    import root_core as G
    G.ROOT_DIR = data_dir / 'original'
    F.DV_DIR = data_dir / 'humanml3d_v1.3'
    F.GH_DIR = data_dir / 'care_pd_9bc22b0'


def run_meta(command, stage):
    """The frozen renderer requires these top-level keys. "card" names the command; "card_sha256" is the
    SHA-256 of the kernel module that ran; "run_commit" is RUN_COMMIT_SHA when set."""
    return {'card': command + ('-' + stage if stage else ''), 'card_sha256': X.digest(HERE / (MODULE[command] + '.py')),
            'run_commit': os.environ.get('RUN_COMMIT_SHA', 'UNSET'), 'seed': SEED[command],
            'protocol_class': PROTOCOL_CLASS[command]}


def save_to(path, obj):
    """Write the validated aggregate once. An existing file is never replaced."""
    text = IC.render(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x', encoding='utf-8') as f:
        f.write(text)
    return path


# ---- encoder inputs, pre-check and probe ------------------------------------------------------------------------
def fetch_encoder_inputs(a, meta):
    import encoder_inputs as F
    receipt = IC.Receipt()
    IC.block_network(F.ALLOWED_HOSTS)
    files, all_ok, total = [], True, 0
    for row in F.plan():
        good, sha, attempts = F.fetch(row['url'], row['dest'], row['bytes'], row['md5'], row['blob'])
        print(f"PROGRESS {row['label']} ok={good}")
        all_ok &= good
        total += row['bytes'] if good else 0
        files.append({'label': row['label'], 'bytes_pinned': row['bytes'], 'integrity_ok': bool(good),
                      'pinned_digest_kind': 'md5' if row['md5'] else 'git_blob_sha1', 'sha256': sha or 'ABSENT',
                      'download_attempts': attempts})
    out = dict(meta)
    out.update({
        'status': 'C1_FETCH_COMPLETE' if all_ok else 'C1_FETCH_INCOMPLETE',
        'sources': {'dataverse_doi': '10.5683/SP3/TWIKMK', 'dataverse_version': F.DV_VERSION,
                    'dataverse_version_licence_as_listed': 'CC BY-NC-ND 4.0',
                    'github_repository': 'TaatiTeam/CARE-PD', 'github_commit': F.GH_COMMIT, 'github_code_licence': 'MIT'},
        'files_part1': files[:len(files) // 2], 'files_part2': files[len(files) // 2:],
        'n_files': IC.count(len(files)), 'n_files_verified': IC.count(sum(f['integrity_ok'] for f in files)),
        'bytes_verified': int(total), 'content_parsed': False,
        'compute': receipt.block(),
    })
    return out, all_ok


def verify_plan(cohorts=None, code=20):
    import encoder_inputs as F
    for row in F.plan():
        if cohorts is not None and not (row['md5'] is not None and row['label'].rsplit('/', 1)[-1] in cohorts):
            continue
        if not F.verified(row['dest'], row['bytes'], row['md5'], row['blob'])[0]:
            IC.stop('a fetched file is missing or does not match its pin' if cohorts is None
                    else 'a HumanML3D input does not match its pin', code)


def encoder_precheck(a, meta):
    import numpy as np
    receipt = IC.Receipt()
    IC.block_network()
    import torch
    import encoder_inputs as F
    import encoder_precheck as K
    import p0_core as P
    torch.set_num_threads(1)
    verify_plan()
    speed_fn = lambda trans, fps: P.speed_and_duration(trans, fps)[0]  # noqa: E731
    reports = {}
    for cohort in K.COHORTS:
        index = K.canonical_index(C.load_cohort(cohort, a.data_dir))
        with np.load(F.dv_path(cohort), allow_pickle=False) as npz:
            reports[cohort] = K.cohort_report(npz, index, speed_fn)
        print('PROGRESS cohort done')
    stats_ok, stats_flags = True, {}
    for name in list(K.COHORTS) + [f'LODO_{c}' for c in K.COHORTS]:
        mean = np.load(F.gh_path(f'{F.STATS}/{name}/Mean.npy'), allow_pickle=False)
        std = np.load(F.gh_path(f'{F.STATS}/{name}/Std_FEAT_BIAS_included.npy'), allow_pickle=False)
        stats_flags[name] = K.stats_report(mean, std)
        stats_ok &= stats_flags[name]
    try:
        obj = torch.load(F.gh_path(F.CKPT), map_location='cpu', weights_only=True)
        ckpt = K.checkpoint_report(obj)
    except Exception as exc:
        ckpt = {'strict_load_ok': False, 'reason': 'weights-only-load-' + type(exc).__name__}
    out = dict(meta)
    out.update({
        'status': K.decide(reports, stats_ok, ckpt),
        'labels_read': False,
        'thresholds': {'min_walk_coverage': K.MIN_WALK_COVERAGE, 'min_participant_coverage': K.MIN_PARTICIPANT_COVERAGE,
                       'min_clean_array_share': K.MIN_CLEAN_ARRAY_SHARE,
                       'min_speed_rank_agreement': K.MIN_SPEED_RANK_AGREEMENT},
        'cohorts': reports, 'cohort_passes': {c: K.cohort_passes(r) for c, r in reports.items()},
        'normalization_files_ok': stats_flags, 'checkpoint': ckpt,
        'compute': receipt.block(),
    })
    return out, out['status'] == 'C2_PRECHECK_PASS'


def encoder_probe(a, meta, p1):
    import numpy as np
    import encoder_probe as A
    # The speed-arm parity reference is the user's own validated P1 result, never a bundled value.
    A.ARCHIVED_S_M = float(p1['arms_inductive_strong']['S']['macro_f1']['M']['point'])
    receipt = IC.Receipt()
    IC.block_network()
    import torch
    import encoder_inputs as F
    import momask_encoder as M
    import p0_core as P
    import p1_core as P1
    torch.set_num_threads(1)
    torch.manual_seed(meta['seed'])
    verify_plan()
    names = list(C.COHORTS)
    data_dir = a.data_dir
    # load_tables receives the common module; this adapter binds the explicit --data-dir.
    common = types.SimpleNamespace(COHORTS=C.COHORTS, read_label=C.read_label,
                                   load_cohort=lambda name: C.load_cohort(name, data_dir))
    t, raws, counts = A.load_tables(common, P)
    print('PROGRESS tables built')
    model = M.load_strict(torch.load(F.gh_path(F.CKPT), map_location='cpu', weights_only=True)['net'])
    stats = lambda name: (np.load(F.gh_path(f'{F.STATS}/{name}/Mean.npy'), allow_pickle=False),  # noqa: E731
                          np.load(F.gh_path(f'{F.STATS}/{name}/Std_FEAT_BIAS_included.npy'), allow_pickle=False))
    out = A.analyse(meta, t, raws, counts, model, stats, P1, M, names)
    out['compute'] = receipt.block(threads=1)
    return out, True


# ---- original annotated files, record-level parity and speed sensitivity -------------------------------------------
def prerequisites_ok(out_dir):
    """The rest stage needs a passed first-stage download and a first-stage parity aggregate (any status)."""
    import root_fetch as D
    for name, status in ((result_name('fetch-originals', 'first'), D.PASS_STATUS),
                         (result_name('root-parity', 'first'), None)):
        path = out_dir / name
        if not path.is_file():
            return False
        try:
            got = json.loads(path.read_text(encoding='utf-8')).get('status')
        except Exception:
            return False
        if status is not None and got != status:
            return False
    return True


def fetch_originals(a, meta, stage):
    import root_core as G
    import root_fetch as D
    cohorts = STAGES[stage]
    D.decode_checks = D.hardened_decode_checks      # opcode counts banded for every cohort
    os.umask(0o077)
    receipt = IC.Receipt()
    if stage == 'rest' and not prerequisites_ok(a.out_dir):
        IC.stop('prerequisite result missing or not passed', 22)
    IC.block_network(D.ALLOWED_HOSTS)
    opener, handler = D.make_opener(D.ALLOWED_HOSTS, D.MAX_REDIRECTS)
    api_opener, _ = D.make_opener(D.API_HOSTS, 0)
    rows, licence, dataset_problems = [], None, []
    need_network = any(not G.root_path(c).exists() for c in cohorts)
    if need_network:
        try:
            dataset_problems, licence = D.check_dataset_meta(D.fetch_json(D.API_DATASET, api_opener))
        except D.Stop as exc:
            dataset_problems = [str(exc)]
    for cohort in cohorts:
        row, _census = D.process(cohort, opener, handler, api_opener, dataset_problems)
        rows.append(row)
        print(f"PROGRESS {cohort} stop={row['stop'] or 'none'}")
        if row['stop'] is not None:
            break
    done = {r['cohort'] for r in rows}
    for cohort in cohorts:
        if cohort not in done:
            rows.append({'cohort': cohort, 'stop': 'NOT_ATTEMPTED_AFTER_EARLIER_STOP'})
    status, code = D.overall_status(rows, dataset_problems)
    out = dict(meta)
    out.update({
        'status': status,
        'labels_read': False,
        'authority': 'terms accepted by the user with --accept-care-pd-terms; local copies only',
        'publisher': {'doi': G.DOI, 'dataset_problems': dataset_problems, 'network_used': bool(need_network),
                      'latest_version_licence_as_listed': licence, 'expected_latest_version': '4.0',
                      'access_endpoint': 'https://borealisdata.ca/api/access/datafile/<datafile>',
                      'allowed_host_classes': ['borealisdata.ca', '*.scholarsportal.info']},
        'files': rows,
        'local_storage': {'location': 'user data folder outside the code repository', 'dir_mode': '0700',
                          'file_mode': '0600'},
        'compute': receipt.block(),
    })
    return out, code == 0


def download_receipt(out_dir, stage):
    """Verified rows of the user's own passed download aggregate for this stage."""
    import root_parity as Q
    path = out_dir / result_name('fetch-originals', stage)
    if not path.is_file():
        IC.stop('download aggregate missing', 22)
    agg = json.loads(path.read_text(encoding='utf-8'))
    if agg.get('status') != Q.DOWNLOAD_PASS:
        IC.stop('download did not pass', 22)
    return {r['cohort']: r for r in agg.get('files', []) if isinstance(r, dict) and r.get('integrity_ok')}


def root_parity(a, meta, stage):
    import numpy as np
    import encoder_inputs as F
    import encoder_precheck as K
    import root_core as G
    import root_parity as Q
    cohorts = STAGES[stage]
    receipt = IC.Receipt()
    IC.block_network()
    rows = download_receipt(a.out_dir, stage)
    verify_plan(cohorts, 23)
    out_cohorts = {}
    for name in cohorts:
        root = Q.load_root(name, rows)
        canon = C.load_cohort(name, a.data_dir)
        with np.load(F.dv_path(name), allow_pickle=False) as npz:
            keys = list(npz.files)
            out_cohorts[name] = Q.cohort_report(name, root, canon, keys, lambda k: npz[k], lambda: dict(E2_UNAVAILABLE))
        out_cohorts[name]['root_file'] = {'publisher_datafile': G.ROOT_PINS[name][1], 'bytes': G.ROOT_PINS[name][2],
                                          'md5': G.ROOT_PINS[name][3], 'sha256': rows[name]['sha256'],
                                          'versions': G.ROOT_PINS[name][4], 'verified_before_use': True}
        out_cohorts[name]['canonical_walks_public'] = IC.count(len(K.canonical_index(canon)))
        del root, canon
        print(f'PROGRESS {name} done')
    out = dict(meta)
    out.update({
        'status': 'G_ROOT_PARITY_COMPLETE',
        'labels_read': True,
        'design': {'label': 'descriptive; reused cohorts; no model fitted; labels never used to choose a transformation',
                   'thresholds': {'profile_r': Q.PROFILE_R, 'ratio_dev': Q.RATIO_DEV, 'residual': Q.RESID,
                                  'match_share': Q.MATCH_SHARE, 'no_match_share': Q.NO_MATCH_SHARE,
                                  'min_steps': Q.MIN_STEPS, 'max_lag': Q.MAX_LAG, 'pose_tol': Q.POSE_TOL,
                                  'orient_tol_deg': Q.ORIENT_TOL_DEG, 'scale_unity_tol': Q.SCALE_UNITY_TOL,
                                  'e2_parity_tol': Q.E2_PARITY_TOL},
                   'official_rate': {c: Q.OFFICIAL_RATE[c] for c in cohorts}},
        'cohorts': out_cohorts,
        'summary': {c: {'label_verdict': r['clinical']['label_verdict'], 'c3x_subset': r['clinical']['c3x_subset_verdict'],
                        'trajectory_class': r['trajectory']['class'],
                        'derivation_step': r['trajectory']['derivation_step']} for c, r in out_cohorts.items()},
        'compute': receipt.block(),
    })
    problems = G.disclosure_problems(out)
    if problems:
        IC.stop('disclosure check failed: ' + ';'.join(problems[:4]))
    return out, True


def root_speed(a, meta, p1_path):
    import root_speed_sensitivity as H
    H.CANONICAL_DIR = a.data_dir
    H.P1_RESULT = p1_path
    H.P1_RESULT_SHA256 = X.digest(p1_path)
    H.RECEIPT_FILES = tuple(a.out_dir / result_name('fetch-originals', s) for s in ('first', 'rest'))
    receipt = IC.Receipt()
    IC.block_network()
    out = H.run(meta)
    out['compute'] = receipt.block()
    out = H.finalize(out)
    return out, out['status'] == H.STATUS_COMPLETE


def terms_refusal():
    print('TERMS_NOT_ACCEPTED: nothing was downloaded.')
    print(f'Read the terms at {TERMS_URL}, the dataset record at {DATAVERSE_URL} and the code repository at {CODE_URL}.')
    print('If you accept them and your use is permitted, repeat with --accept-care-pd-terms.')
    return 2


def main(argv=None):
    a = parse_args(argv)
    stage = 'first' if a.first else ('rest' if a.rest else None)
    try:
        a.data_dir = X.outside_repo(a.data_dir)
        a.out_dir = X.outside_repo(a.out_dir)
        result = a.out_dir / result_name(a.command, stage)
        if result.exists():
            print('RELEASE_REFUSED_RESULT_EXISTS')
            return 10
        if a.command.startswith('fetch-') and not a.accept_care_pd_terms:
            return terms_refusal()
        paths, p1 = {}, None
        if a.command in ('encoder-probe', 'root-speed'):
            paths['p0'] = a.p0_result or a.out_dir / 'p0.json'
            paths['p1'] = a.p1_result or a.out_dir / 'p1_strong_models.json'
            _p0, p1 = X.prior_results(paths['p0'], paths['p1'])
            paths['p1'] = X.outside_repo(paths['p1'])
        if a.command in ('root-parity', 'root-speed'):
            for s in ((stage,) if stage else ('first', 'rest')):
                paths['originals_' + s] = a.out_dir / result_name('fetch-originals', s)
        configure(a.data_dir)
        meta = run_meta(a.command, stage)
        if a.command == 'fetch-encoder-inputs':
            out, ok = fetch_encoder_inputs(a, meta)
        elif a.command == 'encoder-precheck':
            out, ok = encoder_precheck(a, meta)
        elif a.command == 'encoder-probe':
            out, ok = encoder_probe(a, meta, p1)
        elif a.command == 'fetch-originals':
            out, ok = fetch_originals(a, meta, stage)
        elif a.command == 'root-parity':
            out, ok = root_parity(a, meta, stage)
        else:
            out, ok = root_speed(a, meta, paths['p1'])
        out['release_scope'] = SCOPE
        out['release_provenance'] = X.provenance({k: v for k, v in paths.items() if Path(v).is_file()})
        save_to(result, out)
    except IC.RouteError:
        print('RELEASE_STOP:KERNEL_GATE')
        return 20
    except (C.SourceError, OSError, KeyError, ValueError, TypeError) as exc:
        # Never echo a data-derived value or identifier from an exception.
        reason = str(exc) if isinstance(exc, C.SourceError) else type(exc).__name__
        print('RELEASE_STOP:' + reason)
        return 20
    print('STATUS ' + str(out.get('status')))
    if not ok:
        print('RELEASE_GATE_NOT_PASSED:' + result.name)
        return 20
    print('RELEASE_AGGREGATES_ONLY_COMPLETE:' + result.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
