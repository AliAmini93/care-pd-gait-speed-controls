"""Explicit local input/output contract for the portable follow-up commands.

This module neither downloads data nor imports any private repository. Aggregate
inputs are validated before raw files are opened. Numerical parity is subsequently
checked by the frozen kernels before any follow-up metric is computed.
"""
from __future__ import annotations
import hashlib
import json
import platform
import math
from pathlib import Path
import numpy as np
import scipy
import sklearn
import common as C
import p0_core as P
import p1_core as Q
REPO = Path(__file__).resolve().parents[2]

def outside_repo(path):
    path = Path(path).resolve()
    if path == REPO or REPO in path.parents:
        raise C.SourceError('RELEASE_PATH_MUST_BE_OUTSIDE_CODE_REPOSITORY')
    return path

def read_aggregate(path, status):
    path = outside_repo(path)
    if not path.is_file():
        raise C.SourceError('RELEASE_UPSTREAM_MISSING')
    if path.stat().st_size > 262144:
        raise C.SourceError('RELEASE_UPSTREAM_TOO_LARGE')
    try:

        def refuse_constant(_):
            raise ValueError('nonfinite')
        obj = json.loads(path.read_text(encoding='utf-8'), parse_constant=refuse_constant)
        if not isinstance(obj, dict) or obj.get('status') != status:
            raise ValueError('status')
    except (ValueError, UnicodeError):
        raise C.SourceError('RELEASE_UPSTREAM_INVALID') from None
    return obj

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prior_results(p0_path, p1_path):
    Q.check_versions(np.__version__, scipy.__version__, sklearn.__version__)
    p0 = read_aggregate(p0_path, 'P0_COMPLETE')
    p1 = read_aggregate(p1_path, 'P1_COMPLETE') if p1_path is not None else None
    want_files = {c: {'name': f[0], 'bytes': f[1], 'sha256': f[2], 'md5': f[3]} for c, f in C.FILES.items()}
    try:
        for obj in (p0,) if p1 is None else (p0, p1):
            provenance = obj['provenance']
            if not provenance['hf_commit'] == C.HF_COMMIT:
                raise ValueError('contract')
            if not provenance['files'] == want_files:
                raise ValueError('contract')
            if not (obj['n_boot'] == P.N_BOOT and obj['seed'] == P.SEED):
                raise ValueError('contract')
        for a in Q.PARITY_ARMS:
            if not np.isfinite(p0['arms'][a]['M']['point']):
                raise ValueError('contract')
        if p1 is not None:
            if not p1['provenance']['p0_result_sha256'] == digest(p0_path):
                raise ValueError('contract')
            if not p1['candidate_configs'] == [{'name': n, 'family': f, 'params': p} for n, f, p in Q.CONFIGS]:
                raise ValueError('contract')
            names = {c[0] for c in Q.CONFIGS}
            for a in ('S', 'PS', 'KS'):
                ref = p1['arms_inductive_strong'][a]['macro_f1']
                if not np.isfinite(ref['M']['point']):
                    raise ValueError('contract')
                for site in C.COHORTS:
                    if not p1['selected_configs'][a][site]['all'] in names:
                        raise ValueError('contract')
                    if not np.isfinite(ref['per_cohort'][site]['point']):
                        raise ValueError('contract')
    except (AssertionError, KeyError, TypeError, ValueError):
        raise C.SourceError('RELEASE_UPSTREAM_CONTRACT') from None
    return (p0, p1)

def load_table(data_dir, p0):
    objs = {s: C.load_cohort(s, data_dir) for s in C.COHORTS}
    t = Q.build_table(objs)
    Q.check_stops(t)
    Q.parity_check(t, {a: p0['arms'][a]['M']['point'] for a in Q.PARITY_ARMS})
    return (objs, t)

def check_within_parity(t, p0):
    """Compare with the user's validated local P0, without historical outcomes.

    Every deterministic field and its exact JSON type must agree. The bootstrap
    interval has a different draw count here; validate its schema and bounds but
    do not claim equality of that interval or of the historical study run.
    """
    got = P.within_subject_control(t, n_boot=10)

    def equal(actual, expected):
        if type(actual) is not type(expected):
            return False
        if type(expected) is dict:
            return actual.keys() == expected.keys() and all(
                equal(actual[k], expected[k]) for k in expected)
        if type(expected) is float:
            return math.isfinite(actual) and math.isfinite(expected) and actual == expected
        return actual == expected

    def interval(value):
        return (type(value) is list and len(value) == 2
                and all(type(v) is float and math.isfinite(v) and -1 <= v <= 1 for v in value)
                and value[0] <= value[1])

    if type(p0) is not dict or type(p0.get('status')) is not str or p0['status'] != 'P0_COMPLETE':
        raise C.SourceError('RELEASE_R8B_P0_STATUS')
    want = p0.get('within_subject_control')
    if type(want) is not dict or want.keys() != got.keys():
        raise C.SourceError('RELEASE_R8B_WITHIN_SCHEMA')
    # The frozen P0 has a separate sparse schema when every association is undefined.
    for key in got:
        if key == 'mean_rho_ci95':
            if not interval(want[key]) or not interval(got[key]):
                raise C.SourceError('RELEASE_R8B_WITHIN_INTERVAL')
        elif not equal(want[key], got[key]):
            raise C.SourceError('RELEASE_R8B_WITHIN_PARITY')
    return {k: got[k] for k in got if k != 'mean_rho_ci95'}

def provenance(paths):
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'hf_pinned_dataset_commit': C.HF_COMMIT, 'input_hashes': {s: {'bytes': f[1], 'sha256': f[2]} for s, f in C.FILES.items()}, 'upstream_sha256': {k: digest(v) for k, v in paths.items() if v is not None}, 'portable_real_data_parity': 'NOT_VERIFIED'}
