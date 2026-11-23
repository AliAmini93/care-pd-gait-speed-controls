"""Fabricated fixtures shared by the provenance and encoder-probe suites. Not a test suite itself.

Pins are redirected to fabricated files for the calling process only, and restore() puts the real pins
back. Every number written here is fabricated and is never evidence.
"""
from __future__ import annotations

import contextlib
import io
import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'scripts' / 'care_pd'

import common as C  # noqa: E402  (callers insert SCRIPTS into sys.path first)
import encoder_inputs as F  # noqa: E402
import p0_core as P  # noqa: E402
import p1_core as Q  # noqa: E402
import release_context as X  # noqa: E402
import root_core as G  # noqa: E402


def snapshot():
    return {'C.FILES': dict(C.FILES), 'F.DV_FILES': dict(F.DV_FILES), 'F.GH_FILES': dict(F.GH_FILES),
            'G.ROOT_PINS': dict(G.ROOT_PINS), 'G.ROOT_DIR': G.ROOT_DIR, 'F.DV_DIR': F.DV_DIR, 'F.GH_DIR': F.GH_DIR}


def restore(saved):
    for key, target in (('C.FILES', C.FILES), ('F.DV_FILES', F.DV_FILES), ('F.GH_FILES', F.GH_FILES),
                        ('G.ROOT_PINS', G.ROOT_PINS)):
        target.clear()
        target.update(saved[key])
    G.ROOT_DIR, F.DV_DIR, F.GH_DIR = saved['G.ROOT_DIR'], saved['F.DV_DIR'], saved['F.GH_DIR']


def layout(data_dir):
    """The portable data layout the CLI configures (run_provenance.configure)."""
    G.ROOT_DIR = Path(data_dir) / 'original'
    F.DV_DIR = Path(data_dir) / 'humanml3d_v1.3'
    F.GH_DIR = Path(data_dir) / 'care_pd_9bc22b0'


def pin_canonical(data_dir, objs):
    """Write fabricated canonical pickles and point the canonical pins at them."""
    for cohort, obj in objs.items():
        name = C.FILES[cohort][0]
        path = Path(data_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(obj, protocol=4))
        n, sha, md5 = C.hash_file(path)
        C.FILES[cohort] = (name, n, sha, md5)


def pin_roots(objs):
    """Write fabricated original-root pickles under G.ROOT_DIR and point the root pins at them."""
    shas = {}
    for cohort, obj in objs.items():
        blob = pickle.dumps(obj, protocol=4)
        logical, fid, _size, _md5, versions = G.ROOT_PINS[cohort]
        path = G.root_path(cohort)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        n, md5, sha = G.file_digests(path)
        G.ROOT_PINS[cohort] = (logical, fid, n, md5, versions)
        shas[cohort] = sha
    return shas


def receipt(out_dir, name, cohorts, shas, status='G_ROOT_DOWNLOAD_VERIFIED_SAFE_LOAD_PASS'):
    rows = [{'cohort': c, 'publisher_datafile': G.ROOT_PINS[c][1], 'md5_pinned': G.ROOT_PINS[c][3],
             'sha256': shas[c], 'integrity_ok': True, 'stop': None} for c in cohorts]
    path = Path(out_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'status': status, 'files': rows}))
    return path


def pin_npz(cohort, arrays):
    """Write one fabricated HumanML3D collection and point its Dataverse pin at it."""
    path = F.dv_path(cohort)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        np.savez(f, **arrays)
    size, md5, _sha, _blob = F.digests(path)
    fid = F.DV_FILES[cohort][0]
    F.DV_FILES[cohort] = (fid, size, md5)


def pin_repository_files(ckpt_obj, mean=None, std=None):
    """Write the 16 normalization files and a checkpoint; point the repository pins (size, git blob) at them."""
    import torch
    mean = np.zeros(263, np.float32) if mean is None else mean
    std = np.ones(263, np.float32) if std is None else std
    for repo_path in list(F.GH_FILES):
        path = F.gh_path(repo_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if repo_path == F.CKPT:
            torch.save(ckpt_obj, path)
        else:
            with open(path, 'wb') as f:
                np.save(f, mean if repo_path.endswith('Mean.npy') else std)
        size, _md5, _sha, blob = F.digests(path)
        F.GH_FILES[repo_path] = (size, blob)


def upstream(out_dir, s_m=0.5):
    """Minimal P0/P1 aggregates that satisfy release_context.prior_results (all values fabricated)."""
    files = {s: {'name': f[0], 'bytes': f[1], 'sha256': f[2], 'md5': f[3]} for s, f in C.FILES.items()}
    p0 = {'status': 'P0_COMPLETE', 'n_boot': P.N_BOOT, 'seed': P.SEED,
          'arms': {a: {'M': {'point': 0.5}} for a in Q.PARITY_ARMS},
          'provenance': {'hf_commit': C.HF_COMMIT, 'files': files}}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'p0.json').write_text(json.dumps(p0))
    p1 = {'status': 'P1_COMPLETE', 'n_boot': P.N_BOOT, 'seed': P.SEED,
          'provenance': {'hf_commit': C.HF_COMMIT, 'files': files, 'p0_result_sha256': X.digest(out_dir / 'p0.json')},
          'candidate_configs': [{'name': n, 'family': f, 'params': p} for n, f, p in Q.CONFIGS],
          'selected_configs': {a: {s: {'all': Q.CONFIGS[0][0]} for s in C.COHORTS} for a in ('S', 'PS', 'KS')},
          'arms_inductive_strong': {a: {'macro_f1': {'M': {'point': s_m if a == 'S' else 0.5},
                                                     'per_cohort': {s: {'point': 0.5} for s in C.COHORTS}}}
                                    for a in ('S', 'PS', 'KS')}}
    (out_dir / 'p1_strong_models.json').write_text(json.dumps(p1))
    return p0, p1


def quiet_main(module, argv):
    """Run a CLI main() in this process; returns (exit code, captured stdout)."""
    with contextlib.redirect_stdout(io.StringIO()) as output:
        try:
            code = module.main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, output.getvalue()
