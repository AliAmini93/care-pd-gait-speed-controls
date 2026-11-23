"""Fabricated-only rotation, graph, stacking and privacy tests. No archived results or clinical records are read; function and audit tripwires reject real-data access."""
from __future__ import annotations
import inspect
import io
import json
import math
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
HERE = Path(__file__).resolve().parents[1] / 'scripts' / 'care_pd'
REPO_S = os.path.realpath(str(HERE.parents[1]))
PARENT_S = os.path.dirname(REPO_S)
DATA_S = os.path.join(REPO_S, 'data')
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_NET = {'socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyname_ex', 'socket.gethostbyaddr', 'socket.connect', 'socket.bind', 'socket.sendto', 'urllib.Request'}
_PROC = {'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.spawn', 'os.fork', 'os.forkpty', 'pty.spawn'}
_MUT = {'os.remove', 'os.rename', 'os.link', 'os.symlink', 'os.mkdir', 'os.rmdir', 'os.truncate', 'os.chmod', 'os.chown', 'shutil.rmtree', 'os.utime'}

class RealDataAccess(BaseException):
    pass

class TripwireError(BaseException):
    pass

def _real(p):
    if isinstance(p, int) or p is None:
        return None
    try:
        return os.path.realpath(os.path.abspath(os.fsdecode(p)))
    except (TypeError, ValueError, OSError):
        return '\x00unresolvable'

def _under(r, root):
    return r == root or r.startswith(root + os.sep)
_DIR_FD_INDEX = {'os.remove': (1, None), 'os.rmdir': (1, None), 'os.mkdir': (2, None), 'os.rename': (2, 3), 'os.link': (2, 3), 'os.symlink': (None, 2), 'os.chmod': (2, None), 'os.chown': (3, None), 'os.utime': (3, None), 'shutil.rmtree': (1, None)}

def _resolve_with_dir_fd(p, dir_fd):
    """Realpath of p; a relative p with an int dir_fd resolves against that descriptor (tempfile cleanup)."""
    try:
        s = os.fsdecode(p)
    except TypeError:
        return '\x00unresolvable'
    if isinstance(dir_fd, int) and (not isinstance(dir_fd, bool)) and (not os.path.isabs(s)):
        try:
            s = os.path.join(os.readlink('/proc/self/fd/%d' % dir_fd), s)
        except OSError:
            return os.path.join(REPO_S, '\x00unresolvable_dir_fd')
    return _real(s)

def _hook(event, args):
    if event in _NET:
        raise TripwireError('TRIPWIRE_NETWORK')
    if event in _PROC:
        raise TripwireError('TRIPWIRE_PROCESS')
    if event == 'pickle.find_class':
        raise TripwireError('TRIPWIRE_UNPICKLE_GLOBAL')
    if event == 'open':
        r = _real(args[0] if args else None)
        if r is None:
            return
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        base = os.path.basename(r).lower()
        if _under(r, DATA_S) or base.endswith(('.pkl', '.pickle')):
            raise RealDataAccess('REAL_DATA_ACCESS_ATTEMPTED')
        if '.local.' in base:
            raise TripwireError('TRIPWIRE_LOCAL_LEDGER')
        if _under(r, PARENT_S) and (not _under(r, REPO_S)):
            raise TripwireError('TRIPWIRE_SIBLING')
        if _under(r, REPO_S) and (bool(flags & _WRITE_FLAGS) or any((c in mode for c in 'wax+'))):
            raise TripwireError('TRIPWIRE_REPO_WRITE')
        return
    if event in ('os.listdir', 'os.scandir'):
        r = _real(args[0] if args else None)
        if r and (_under(r, DATA_S) or (_under(r, PARENT_S) and (not _under(r, REPO_S)))):
            raise RealDataAccess('REAL_DATA_LISTING_ATTEMPTED')
        return
    if event in _MUT:
        for i, a in enumerate(args[:2]):
            if isinstance(a, (str, bytes, os.PathLike)):
                fd_idx = _DIR_FD_INDEX.get(event, (None, None))[i]
                fd = args[fd_idx] if fd_idx is not None and len(args) > fd_idx else None
                r = _resolve_with_dir_fd(a, fd)
                if r and _under(r, REPO_S):
                    raise TripwireError('TRIPWIRE_REPO_MUTATION')
sys.addaudithook(_hook)
sys.path.insert(0, str(HERE))
import common as C

def _trip(*_a, **_k):
    raise RealDataAccess('REAL_DATA_ACCESS_ATTEMPTED')
C.load_cohort = _trip
C.verify_file = _trip
C.hash_file = _trip
C.safe_loads = _trip
C.RestrictedUnpickler.load = _trip
_real_loads = pickle.loads
_GLOBAL_OPS = {'GLOBAL', 'STACK_GLOBAL', 'INST', 'OBJ', 'NEWOBJ', 'NEWOBJ_EX', 'REDUCE', 'BUILD', 'PERSID', 'BINPERSID', 'EXT1', 'EXT2', 'EXT4'}

def _global_free_loads(data, *a, **k):
    """Only global-free payloads (e.g. the None sentinel of sklearn's threading pool) may be unpickled."""
    import pickletools
    try:
        ops = {op.name for op, _a, _p in pickletools.genops(bytes(data))}
    except Exception:
        raise RealDataAccess('UNPICKLE_REFUSED') from None
    if ops & _GLOBAL_OPS or len(data) > 4096:
        raise RealDataAccess('UNPICKLE_REFUSED')
    return _real_loads(data, *a, **k)
pickle.load = _trip
pickle.loads = _global_free_loads
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
import p0_core as P
import p1_core as Q
import representation_core as RC
import representation_native as RN
import representation_one_shot as OS
import representation_public_encoder as PE
torch.use_deterministic_algorithms(True)
torch.utils.deterministic.fill_uninitialized_memory = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(2)
TESTS = []
FROZEN_CONFIG = {'width': 64, 'blocks': 3, 'kernel': 5, 'dilations': (1, 2, 4), 'embed_dim': 64, 'dropout': 0.2, 'epochs_max': 40, 'snapshot_epochs': (4, 8, 12, 16, 20, 24, 28, 32, 36, 40), 'batch': 64, 'lr': 0.001, 'weight_decay': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-08, 'warmup_epochs': 1, 'schedule': 'cosine_to_zero', 'grad_clip': 1.0, 'seeds': (20261105, 20261106, 20261107), 'eval_batch': 256, 'precision': 'float32', 'augmentation': 'one random 60-frame crop per walk per epoch', 'class_weights': 'P1 walk scheme, training only'}
FROZEN_N_PARAMS = 178883
WHAM_SMPL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23))
TEST_CFG = dict(RN.CONFIG, width=16, embed_dim=16, epochs_max=40, snapshot_epochs=(10, 20, 30, 40), seeds=(11,), batch=32, lr=0.003, eval_batch=128)
FAST_CFG = dict(TEST_CFG, epochs_max=4, snapshot_epochs=(2, 4))

def test(fn):
    TESTS.append(fn)
    return fn

def raises(code, fn, *a, **k):
    try:
        fn(*a, **k)
    except C.SourceError as exc:
        assert str(exc).startswith(code), (code, str(exc))
        return
    raise AssertionError('expected SourceError ' + code)
FAB_FPS = {'3DGait': 30.0, 'BMCLab': 150.0, 'PD-GaM': 25.0, 'T-SDU-PD': 30.0}

def fab_walk(rng, fps, dur_s, v, trunk, heading, knee=0.6, hip=0.4, cadence=1.8):
    T = max(2, int(round(dur_s * fps)))
    tt = np.arange(T) / fps
    ph = 2 * np.pi * cadence * tt + rng.uniform(0, 2 * np.pi)
    pose = np.zeros((T, 72))
    root = Rotation.from_euler('y', heading + 0.05 * np.sin(ph)) * Rotation.from_euler('x', 0.05 + 0.02 * np.sin(2 * ph))
    pose[:, 0:3] = root.as_rotvec()
    pose[:, 3] = hip * np.sin(ph)
    pose[:, 6] = hip * np.sin(ph + np.pi)
    pose[:, 12] = knee * (1 + np.sin(ph)) / 2
    pose[:, 15] = knee * (1 + np.sin(ph + np.pi)) / 2
    for j in (3, 6, 9):
        pose[:, 3 * j] = trunk / 3 + 0.01 * np.sin(2 * ph)
    pose[:, 3 * 16 + 2] = -1.2 + 0.2 * np.sin(ph + np.pi)
    pose[:, 3 * 17 + 2] = 1.2 + 0.2 * np.sin(ph)
    pose += rng.normal(0, 0.01, pose.shape)
    trans = np.column_stack([v * tt * np.sin(heading), 0.9 + 0.02 * np.sin(2 * ph) + rng.normal(0, 0.002, T), v * tt * np.cos(heading)])
    return (pose, trans)

def fab_objs(seed, world, walks=3, dur=(2.2, 3.6), trunk_step=0.25):
    """Nine persons per cohort covering every (speed class, rotation class) pair once, plus one grade-3 person and
    one unlabelled walk per cohort. world 'null': grade = speed class; 'signal': grade from speed AND trunk class;
    'signal_untied': grade mostly from trunk class with speed incremental, and no exactly tied rotation stratum."""
    rng = np.random.default_rng(seed)
    objs = {}
    for ci, name in enumerate(C.COHORTS):
        cohort, combos = ({}, [(cv, r) for cv in range(3) for r in range(3)])
        for p, (cv, r) in enumerate(combos):
            if world == 'signal':
                u = cv + r
                grade = 0 if u <= 1 else 1 if u == 2 else 2
                trunk_class = r
            elif world == 'signal_untied':
                # Same rotation/speed construction as 'signal', but no rotation class has a tied grade distribution:
                # under class-balanced weights the rotation-only Bayes macro-F1 is 0.783 (no tie) instead of 0.444 or
                # 0.556 depending only on how an exactly uniform stratum is broken; speed stays incremental (S 0.300,
                # S+N 1.0). See docs/t40_fixture_analysis.md.
                u = 2 * r + cv
                grade = 0 if u <= 1 else 1 if u <= 4 else 2
                trunk_class = r
            else:
                grade = cv
                trunk_class = int(rng.integers(0, 3))
            v0 = 1.3 - 0.3 * cv + rng.normal(0, 0.03)
            t0 = trunk_step * trunk_class + rng.normal(0, 0.02)
            ws = {}
            for k in range(walks):
                pose, trans = fab_walk(rng, FAB_FPS[name], rng.uniform(*dur), v0 * rng.uniform(0.98, 1.02), t0 + rng.normal(0, 0.01), rng.uniform(-np.pi, np.pi))
                ws[f'w{k}'] = {'pose': pose, 'trans': trans, 'beta': np.zeros(10), 'fps': FAB_FPS[name], 'UPDRS_GAIT': grade, 'medication': 'ON', 'other': None}
            cohort[f'fab{ci}_{p}'] = ws
        pose, trans = fab_walk(rng, FAB_FPS[name], 2.5, 0.6, 0.5, 0.0)
        cohort[f'fab{ci}_g3'] = {'w0': {'pose': pose, 'trans': trans, 'beta': np.zeros(10), 'fps': FAB_FPS[name], 'UPDRS_GAIT': 3, 'medication': 'OFF', 'other': None}}
        pose, trans = fab_walk(rng, FAB_FPS[name], 2.5, 1.0, 0.1, 0.0)
        cohort[f'fab{ci}_none'] = {'w0': {'pose': pose, 'trans': trans, 'beta': np.zeros(10), 'fps': FAB_FPS[name], 'UPDRS_GAIT': None, 'medication': 'ON', 'other': None}}
        objs[name] = cohort
    return objs
_CACHE = {}

def campaign(world, seed=5, n_boot=200, with_partial=True):
    key = (world, seed, n_boot, with_partial)
    if key not in _CACHE:
        t, store = OS.build_inputs(fab_objs(seed, world, walks=6, trunk_step=0.8))
        _CACHE[key] = (t, store, RN.run_campaign(t, store, 'cpu', cfg=TEST_CFG, n_boot=n_boot, with_partial=with_partial))
    return _CACHE[key]

def rand_rot(rng, n):
    return Rotation.random(n, random_state=int(rng.integers(1 << 30))).as_matrix()

@test
def t01_function_tripwire():
    for fn in (C.load_cohort, C.verify_file, C.hash_file, C.safe_loads, pickle.load):
        try:
            fn('x')
        except RealDataAccess:
            continue
        raise AssertionError('function tripwire not armed')
    for payload in (pickle.dumps({'pose': np.zeros(3)}), pickle.dumps(Path('x')), b'\x80\x04' + b'x' * 5000):
        try:
            pickle.loads(payload)
        except RealDataAccess:
            continue
        raise AssertionError('pickle.loads accepted a payload with globals')
    assert pickle.loads(pickle.dumps(None)) is None and pickle.loads(pickle.dumps([1, 'a'])) == [1, 'a']

@test
def t02_audit_tripwire_paths():
    for path in (os.path.join(DATA_S, 'care_pd', 'raw', 'BMCLab_canonical.pkl'), '/tmp/anything.pkl', os.path.join(DATA_S, 'care_pd', 'anything.json')):
        try:
            open(path, 'rb')
        except RealDataAccess:
            continue
        raise AssertionError('open not refused: ' + path)
    try:
        os.listdir(DATA_S)
    except RealDataAccess:
        pass
    else:
        raise AssertionError('data listing not refused')
    for path, mode in ((os.path.join(REPO_S, 'results', 'summary', 'x.json'), 'w'), (os.path.join(REPO_S, 'logs', 'gait.local.tsv'), 'r')):
        try:
            open(path, mode)
        except TripwireError:
            continue
        raise AssertionError('repo write / local ledger not refused')

@test
def t03_audit_tripwire_network_process_unpickle():
    import socket
    import subprocess
    for fn in (lambda: socket.getaddrinfo('example.org', 443), lambda: subprocess.run(['true']), lambda: pickle._Unpickler(io.BytesIO(pickle.dumps(Path('x')))).load()):
        try:
            fn()
        except TripwireError:
            continue
        raise AssertionError('network/process/unpickle not refused')

@test
def t04_static_source_hygiene():
    names = ('representation_core.py', 'representation_native.py', 'representation_public_encoder.py', 'representation_one_shot.py')
    banned = ('torch.load', 'np.load(', 'pickle.load', 'pickle.dump', 'torch.save', 'np.save', 'import joblib', 'import requests', 'import urllib', 'import socket', 'import subprocess', 'extractall', 'state_dict(', 'weights_only=False')
    for n in names:
        src = (HERE / n).read_text(encoding='utf-8')
        for b in banned:
            assert b not in src, (n, b)
    one = (HERE / 'representation_one_shot.py').read_text(encoding='utf-8')
    assert one.count('open(') == 0
    import ast
    import textwrap
    for fn in (RC.walk_features, RC.window_array, RC.canonical_root, RN.native_outer_fold, RN.train_snapshots, RN.prepare_eval, RN.train_batch):
        assert 'trans' not in inspect.signature(fn).parameters
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and (len(n.value) < 12)}
        assert not {'trans', 'speed', 'speed_alt', 'duration'} & used, (fn.__name__, used & {'trans', 'speed'})

@test
def t05_frozen_constants():
    assert RN.CONFIG == FROZEN_CONFIG
    assert OS.VRAM_CAP_GIB == 4.0 and OS.TRAIN_BUDGET_S == 16200 and (OS.TORCH_VERSION == '2.9.1+cu128')
    assert RC.MARGIN == 0.03 and RC.SITE_FLOOR == -0.03 and (RC.MIN_POSITIVE_SITES == 2)
    assert RC.WINDOW == 60 and RC.TARGET_FPS == 30.0 and (RC.N_NOISE_COLUMNS == 3)
    assert RN.S_PARITY_TOL == 1.1e-06 and RN.ARMS == ('S', 'N', 'S_plus_N', 'S_plus_NOISE3')

@test
def t06_rodrigues_matches_reference():
    rng = np.random.default_rng(1)
    rv = np.concatenate([rng.normal(0, 1.5, (500, 3)), np.zeros((1, 3)), [[1e-09, 0, 0], [0, 1e-07, 2e-07], [np.pi, 0, 0], [0, -np.pi + 1e-06, 0], [2.0, -1.0, 0.5]]])
    err = np.abs(RC.rotvec_to_matrix(rv) - Rotation.from_rotvec(rv).as_matrix()).max()
    assert err < 1e-12, err
    raises('REP_ROTVEC_SHAPE', RC.rotvec_to_matrix, np.zeros((3, 2)))

@test
def t07_six_d_convention_and_inverse():
    rng = np.random.default_rng(2)
    R = rand_rot(rng, 300)
    x = RC.matrix_to_6d(R)
    assert np.array_equal(x[:, :3], R[:, :, 0]) and np.array_equal(x[:, 3:], R[:, :, 1])
    assert np.abs(RC.sixd_to_matrix(x) - R).max() < 1e-12
    raises('REP_6D_DEGENERATE', RC.sixd_to_matrix, np.zeros((1, 6)))
    raises('REP_6D_DEGENERATE', RC.sixd_to_matrix, np.array([[1.0, 0, 0, 2.0, 0, 0]]))

@test
def t08_axis_angle_wrapping_is_continuous_in_6d():
    rng = np.random.default_rng(3)
    n = rng.normal(size=(50, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    th = rng.uniform(0.1, np.pi, (50, 1))
    a = RC.matrix_to_6d(RC.rotvec_to_matrix(th * n))
    b = RC.matrix_to_6d(RC.rotvec_to_matrix((th - 2 * np.pi) * n))
    assert np.abs(a - b).max() < 1e-12
    c = RC.matrix_to_6d(RC.rotvec_to_matrix(np.pi * n))
    d = RC.matrix_to_6d(RC.rotvec_to_matrix(-np.pi * n))
    assert np.abs(c - d).max() < 1e-12

@test
def t09_smpl_tree_matches_wham_source_and_partitions():
    par = RC.SMPL_PARENTS
    assert len(par) == 24 and par[0] == -1 and all((0 <= par[j] < j for j in range(1, 24)))
    assert tuple(((par[j], j) for j in range(1, 24))) == WHAM_SMPL_EDGES
    A = RN.graph_partitions()
    assert A.shape == (3, 24, 24) and np.array_equal(A[0], np.eye(24, dtype=np.float32))
    assert A[1][0].sum() == 0 and np.allclose(A[1][1:].sum(axis=1), 1.0)
    leaves = {j for j in range(24) if j not in par}
    assert leaves == {10, 11, 15, 22, 23}
    for u in range(24):
        assert math.isclose(A[2][u].sum(), 0.0 if u in leaves else 1.0, abs_tol=1e-06)
    m = RN.SMPLGraphTCN(TEST_CFG)
    assert 'A' in dict(m.named_buffers()) and all((n != 'A' for n, _ in m.named_parameters()))

@test
def t10_heading_removal_invariance_and_tilt():
    rng = np.random.default_rng(4)
    R0 = (Rotation.from_euler('y', 0.3 + 0.05 * rng.normal(size=60)) * Rotation.from_euler('x', 0.2)).as_matrix()
    base = RC.canonical_root(R0)
    for psi in (-2.5, 0.7, np.pi):
        moved = RC.canonical_root(RC.yaw_matrix(psi) @ R0)
        assert np.abs(moved - base).max() < 1e-12
    f = base[:, :, 2]
    assert abs(math.atan2(f[:, 0].sum(), f[:, 2].sum())) < 1e-12
    up_before = (R0 @ np.array([0, 1.0, 0]))[:, 1]
    up_after = (base @ np.array([0, 1.0, 0]))[:, 1]
    assert np.abs(up_before - up_after).max() < 1e-12
    vertical = Rotation.from_euler('x', [np.pi / 2, -np.pi / 2]).as_matrix()
    assert RC.window_heading(vertical) == 0.0

@test
def t11_heading_is_per_window_and_leaves_local_joints():
    rng = np.random.default_rng(5)
    pose, _ = fab_walk(rng, 30.0, 5.0, 1.0, 0.2, 1.1)
    root, loc = RC.walk_features(pose, 30.0)
    store = RC.WalkStore([root], [loc])
    x0, m0 = RC.window_array(store, 0, 0)
    x1, m1 = RC.window_array(store, 0, 60)
    assert np.array_equal(x0[:, 1:], loc[:60]) and np.array_equal(x1[:, 1:], loc[60:120])
    R_a = RC.sixd_to_matrix(x0[:, 0].astype(np.float64))
    f = R_a[:, :, 2]
    assert abs(math.atan2(f[:, 0].sum(), f[:, 2].sum())) < 1e-05

@test
def t12_integer_multiple_rates_use_exact_stride():
    rng = np.random.default_rng(6)
    R = rand_rot(rng, 151 * 24).reshape(151, 24, 3, 3)
    assert np.array_equal(RC.resample_rotations(R, 30.0), R)
    assert np.array_equal(RC.resample_rotations(R, 150.0), R[::5])
    assert np.array_equal(RC.resample_rotations(R, 60.0), R[::2])
    assert RC.resample_rotations(R, 150.0).shape[0] == 31

@test
def t13_slerp_25_to_30():
    rng = np.random.default_rng(7)
    T = 101
    R = rand_rot(rng, T * 24).reshape(T, 24, 3, 3)
    out = RC.resample_rotations(R, 25.0)
    K = int(math.floor((T - 1) * 6 / 5)) + 1
    assert out.shape[0] == K
    for n in range(0, K, 6):
        assert np.array_equal(out[n], R[5 * n // 6])
    j = 7
    ref = Slerp(np.arange(T) / 25.0, Rotation.from_matrix(R[:, j]))(np.arange(K) / 30.0).as_matrix()
    assert np.abs(out[:, j] - ref).max() < 1e-09
    assert (K - 1) / 30.0 <= (T - 1) / 25.0 + 1e-12

@test
def t14_no_drift_over_ten_minutes_and_odd_rates():
    for fps in (25.0, 24.0, 50.0, 100.0, 29.97):
        T = int(600 * fps)
        R = np.broadcast_to(np.eye(3), (T, 1, 3, 3)).copy()
        K = RC.resample_rotations(R, fps).shape[0]
        assert (K - 1) / 30.0 <= (T - 1) / fps + 1e-09 < K / 30.0 + 1e-09, fps

@test
def t15_invalid_inputs_refused_never_imputed():
    good = np.zeros((10, 72))
    raises('REP_FPS_INVALID', RC.walk_features, good, 0.0)
    raises('REP_FPS_INVALID', RC.walk_features, good, float('nan'))
    raises('REP_FPS_INVALID', RC.walk_features, good, 5000.0)
    bad = good.copy()
    bad[3, 5] = np.nan
    raises('REP_POSE_NONFINITE', RC.walk_features, bad, 30.0)
    raises('REP_POSE_SHAPE', RC.walk_features, np.zeros((10, 69)), 30.0)
    raises('REP_POSE_SHAPE', RC.walk_features, np.zeros((0, 72)), 30.0)
    raises('REP_POSE_DTYPE', RC.walk_features, np.zeros((10, 72), dtype=int), 30.0)
    r, l6 = RC.walk_features(np.zeros((1, 72)), 25.0)
    assert r.shape == (1, 3, 3) and l6.shape == (1, 23, 6)
    r, _ = RC.walk_features(np.zeros((2, 72)), 150.0)
    assert r.shape[0] == 1

def small_store(seed=8, lengths=(17, 60, 61, 150)):
    rng = np.random.default_rng(seed)
    roots, locs = ([], [])
    for K in lengths:
        r, l6 = RC.walk_features(rng.normal(0, 0.3, (K, 72)), 30.0)
        roots.append(r)
        locs.append(l6)
    return RC.WalkStore(roots, locs)

@test
def t16_window_masks_padding_and_bounds():
    store = small_store()
    X, M = RC.window_array(store, 0, 0)
    assert X.shape == (60, 24, 6) and X.dtype == np.float32 and (M.sum() == 17)
    assert np.all(X[17:] == 0) and (not np.any(X[:17, 1:] == 0))
    raises('REP_WINDOW_START', RC.window_array, store, 0, 17)
    raises('REP_WINDOW_START', RC.window_array, store, 0, -1)
    rng = np.random.default_rng(0)
    for K in (17, 60, 61, 150):
        s = [RC.crop_start(K, rng) for _ in range(200)]
        assert min(s) >= 0 and max(s) <= max(0, K - 60)
        cover = np.zeros(K, int)
        for st in RC.eval_starts(K):
            cover[st:st + 60] += 1
        assert np.all(cover == 1)

@test
def t17_padding_garbage_changes_nothing():
    torch.manual_seed(0)
    m = RN.SMPLGraphTCN().eval()
    store = small_store()
    X, M = RC.window_array(store, 0, 0)
    xb, mb = (torch.from_numpy(X[None]), torch.from_numpy(M[None]))
    with torch.no_grad():
        a = m(xb, mb)
        xg = xb.clone()
        xg[0, 17:] = torch.randn(43, 24, 6) * 50
        b = m(xg, mb)
    assert torch.equal(a, b)

@test
def t18_padded_equals_unpadded_and_batch_invariance():
    torch.manual_seed(1)
    m = RN.SMPLGraphTCN().eval()
    store = small_store()
    X, M = RC.window_array(store, 0, 0)
    with torch.no_grad():
        a = m(torch.from_numpy(X[None]), torch.from_numpy(M[None]))
        b = m(torch.from_numpy(X[None, :17].copy()), torch.from_numpy(M[None, :17].copy()))
        X2, M2 = RC.window_array(store, 3, 60)
        batch = m(torch.from_numpy(np.stack([X2, X])), torch.from_numpy(np.stack([M2, M])))
    assert (a - b).abs().max() < 1e-05 and (batch[1] - a[0]).abs().max() < 1e-05

@test
def t19_walk_pooling_is_frame_weighted_mean():
    torch.manual_seed(2)
    m = RN.SMPLGraphTCN(TEST_CFG).eval()
    store = small_store()
    idx = [3, 0, 2]
    prep = RN.prepare_eval(store, idx)
    got = RN.walk_logits(m, prep, 'cpu', 7)
    for pos, i in enumerate(idx):
        num, den = (np.zeros(3), 0.0)
        for s in RC.eval_starts(int(store.lengths[i])):
            x, mk = RC.window_array(store, i, s)
            with torch.no_grad():
                lg = m(torch.from_numpy(x[None]), torch.from_numpy(mk[None])).double().numpy()[0]
            num += lg * mk.sum()
            den += mk.sum()
        assert np.abs(got[pos] - num / den).max() < 1e-05

@test
def t20_parameter_budget_and_frozen_size():
    m = RN.SMPLGraphTCN()
    assert RN.n_parameters(m) == FROZEN_N_PARAMS <= RN.PARAM_BUDGET
    logits, e = m(torch.zeros(2, 60, 24, 6), torch.ones(2, 60, dtype=torch.bool), return_embedding=True)
    assert logits.shape == (2, 3) and e.shape == (2, 64) and (len(m.blocks) == 3)

@test
def t21_fixed_topology_is_used():
    """Relabel joints 4 <-> 16 consistently everywhere EXCEPT the fixed graph. A graph-free per-joint model would be
    exactly equivariant; a difference shows the fixed SMPL topology carries information."""
    torch.manual_seed(3)
    m = RN.SMPLGraphTCN().eval()
    with torch.no_grad():
        m.joint.data.normal_(0, 0.5)
    store = small_store()
    X, M = RC.window_array(store, 3, 0)
    perm = np.arange(24)
    perm[[4, 16]] = perm[[16, 4]]
    with torch.no_grad():
        a = m(torch.from_numpy(X[None]), torch.from_numpy(M[None]))
        m.joint.data[:] = m.joint.data[perm].clone()
        W = m.emb.weight.data.reshape(m.emb.out_features, 24, -1)
        m.emb.weight.data[:] = W[:, perm].reshape(m.emb.out_features, -1).clone()
        b = m(torch.from_numpy(X[None][:, :, perm].copy()), torch.from_numpy(M[None]))
        m2 = RN.SMPLGraphTCN().eval()
        m2.load_state_dict(m.state_dict())
        m2.A.zero_()
        m2.A[0] = torch.eye(24)
        m3 = RN.SMPLGraphTCN().eval()
        m3.load_state_dict(m2.state_dict())
        m3.joint.data[:] = m3.joint.data[perm].clone()
        W3 = m3.emb.weight.data.reshape(m3.emb.out_features, 24, -1)
        m3.emb.weight.data[:] = W3[:, perm].reshape(m3.emb.out_features, -1).clone()
        c = m2(torch.from_numpy(X[None][:, :, perm].copy()), torch.from_numpy(M[None]))
        d = m3(torch.from_numpy(X[None]), torch.from_numpy(M[None]))
    assert (a - b).abs().max() > 0.0001
    assert (c - d).abs().max() < 1e-05

def tiny_training_set(seed=9, n=40):
    rng = np.random.default_rng(seed)
    roots, locs, y = ([], [], [])
    for k in range(n):
        g = k % 3
        pose, _ = fab_walk(rng, 30.0, 2.5, 1.0, 0.25 * g, rng.uniform(-3, 3))
        r, l6 = RC.walk_features(pose, 30.0)
        roots.append(r)
        locs.append(l6)
        y.append(g)
    return (RC.WalkStore(roots, locs), np.array(y))

@test
def t22_training_refuses_overlap_and_shape():
    store, y = tiny_training_set()
    prep = RN.prepare_eval(store, [30, 31])
    cfg = dict(TEST_CFG, epochs_max=2, snapshot_epochs=(2,))
    raises('REP_TRAIN_EVAL_OVERLAP', RN.train_snapshots, store, np.arange(31), y[:31], np.ones(31), [30, 31], prep, 1, 'cpu', (2,), cfg)
    raises('REP_TRAIN_SHAPE', RN.train_snapshots, store, np.arange(30), y[:29], np.ones(30), [30, 31], prep, 1, 'cpu', (2,), cfg)

@test
def t23_training_is_seed_deterministic_and_truncation_exact():
    store, y = tiny_training_set()
    ev = list(range(30, 40))
    prep = RN.prepare_eval(store, ev)
    cfg = dict(TEST_CFG, epochs_max=6, snapshot_epochs=(2, 4, 6))
    a, _ = RN.train_snapshots(store, np.arange(30), y[:30], np.ones(30), ev, prep, 7, 'cpu', (2, 4, 6), cfg)
    b, _ = RN.train_snapshots(store, np.arange(30), y[:30], np.ones(30), ev, prep, 7, 'cpu', (2, 4, 6), cfg)
    c, _ = RN.train_snapshots(store, np.arange(30), y[:30], np.ones(30), ev, prep, 8, 'cpu', (2, 4, 6), cfg)
    d, info = RN.train_snapshots(store, np.arange(30), y[:30], np.ones(30), ev, prep, 7, 'cpu', (4,), cfg, last=4)
    assert all((np.array_equal(a[e], b[e]) for e in (2, 4, 6)))
    assert not np.array_equal(a[6], c[6])
    assert np.array_equal(a[4], d[4]) and info['epochs_run'] == 4

@test
def t24_epoch_selection_rule():
    assert RN.select_epoch({4: 0.5, 8: 0.6, 12: 0.6}) == 8
    assert RN.select_epoch({4: 0.6, 8: 0.6 + 1e-13, 12: 0.59}) == 4
    assert RN.select_epoch({8: 0.2, 4: 0.3}) == 4

def fold_inputs(world='signal', seed=21):
    return OS.build_inputs(fab_objs(seed, world))

@test
def t25_target_labels_never_used_by_native_fold():
    t, store = fold_inputs()
    cfg = FAST_CFG
    a = RN.native_outer_fold(store, t['label'], t['cohort'], t['subject'], 1, 'cpu', cfg)
    lab = t['label'].copy()
    tgt = np.flatnonzero((t['cohort'] == 1) & np.isin(lab, (0, 1, 2)))
    lab[tgt] = (lab[tgt] + 1) % 3
    b = RN.native_outer_fold(store, lab, t['cohort'], t['subject'], 1, 'cpu', cfg)
    assert a['e_star'] == b['e_star'] and np.array_equal(a['target_logits'], b['target_logits'])
    assert np.array_equal(a['oof_logits'], b['oof_logits'])
    assert set(np.unique(t['label'][a['train_idx']])) <= {0, 1, 2}
    assert set(np.unique(t['cohort'][a['train_idx']])) == {0, 2, 3}

@test
def t26_target_inputs_never_train_the_model():
    t, store = fold_inputs()
    cfg = FAST_CFG
    a = RN.native_outer_fold(store, t['label'], t['cohort'], t['subject'], 2, 'cpu', cfg)
    rng = np.random.default_rng(0)
    roots, locs = (list(store.roots), list(store.locals))
    for i in np.flatnonzero(t['cohort'] == 2):
        roots[i] = RC.rotvec_to_matrix(rng.normal(0, 1, (roots[i].shape[0], 3))).astype(np.float32)
        locs[i] = rng.uniform(-1, 1, locs[i].shape).astype(np.float32)
    b = RN.native_outer_fold(RC.WalkStore(roots, locs), t['label'], t['cohort'], t['subject'], 2, 'cpu', cfg)
    assert np.array_equal(a['oof_logits'], b['oof_logits']) and a['e_star'] == b['e_star']
    assert [r['loss_last'] for r in a['runs']] == [r['loss_last'] for r in b['runs']]
    assert not np.array_equal(a['target_logits'], b['target_logits'])

@test
def t27_subject_overlap_refused():
    t, store = fold_inputs()
    subj = t['subject'].copy()
    i_tr = np.flatnonzero(t['cohort'] == 0)[0]
    i_te = np.flatnonzero((t['cohort'] == 1) & np.isin(t['label'], (0, 1, 2)))[0]
    subj[i_te] = subj[i_tr]
    raises('REP_SUBJECT_OVERLAP', RN.native_outer_fold, store, t['label'], t['cohort'], subj, 1, 'cpu', dict(TEST_CFG, epochs_max=2, snapshot_epochs=(2,)))

@test
def t28_window_features_are_per_record():
    store = small_store()
    X, M = RC.window_array(store, 3, 60)
    roots, locs = (list(store.roots), list(store.locals))
    roots[0] = np.broadcast_to(np.eye(3, dtype=np.float32), roots[0].shape).copy()
    locs[0] = np.zeros_like(locs[0])
    X2, M2 = RC.window_array(RC.WalkStore(roots, locs), 3, 60)
    assert np.array_equal(X, X2) and np.array_equal(M, M2)
    src = inspect.getsource(RN)
    for banned in ('StandardScaler', '.mean(axis=0)', '.std('):
        assert banned not in src, banned

@test
def t29_class_weights_are_training_only_and_balanced():
    y = np.array([0] * 10 + [1] * 5 + [2] * 2)
    w = Q.train_weights(y, np.arange(17), 'walk')
    assert np.allclose([w[y == k].sum() for k in range(3)], 17 / 3)
    src = inspect.getsource(RN.native_outer_fold)
    assert 'train_weights(label[a]' in src and 'train_weights(label[tr]' in src and ('label[te]' not in src)

@test
def t30_single_fold_equals_p1_outer_run():
    t, _ = fold_inputs('null', 22)
    feats = Q.arm_features(t)
    warned = [0]
    folds = Q.outer_run(feats['S'], t['label'], t['cohort'], t['subject'], 'walk', Q.CONFIGS, warned)
    el = np.isin(t['label'], (0, 1, 2))
    for ci in range(4):
        tr, te = (el & (t['cohort'] != ci), el & (t['cohort'] == ci))
        p, name, _ = RC.p1_single_fold(feats['S'][tr], t['label'][tr], t['cohort'][tr], t['subject'][tr], feats['S'][te], warned)
        assert name == Q.CONFIGS[folds[ci]['selected']['all']][0]
        assert np.array_equal(p, Q.proba_for(folds, ci, policy='all'))

@test
def t31_single_fold_cannot_see_target_labels():
    params = list(inspect.signature(RC.p1_single_fold).parameters)
    assert params[:5] == ['X_src', 'y_src', 'site_src', 'subj_src', 'X_tgt'] and 'y_tgt' not in params
    raises('REP_DOWNSTREAM_LABELS', RC.p1_single_fold, np.zeros((3, 1)), np.array([0, 1, 3]), np.zeros(3), np.arange(3), np.zeros((2, 1)), [0])

@test
def t32_noise_control_is_label_free_and_fixed():
    a, b = (RC.noise_columns(50), RC.noise_columns(50))
    assert a.shape == (50, 3) and np.array_equal(a, b)
    assert 'label' not in inspect.signature(RC.noise_columns).parameters

def block(m, lo, sites):
    return {'M': {'point': m, 'ci95': [lo, lo + 0.1]}, 'per_cohort': {n: {'point': d, 'ci95': [d - 0.1, d + 0.1]} for n, d in zip(C.COHORTS, sites)}}

@test
def t33_decision_rule_boundaries():
    d = RC.decide_comparison
    assert d(block(0.03, 0.001, [0.01, 0.01, -0.03, 0.2]))['status'] == 'SIGNAL_SUPPORTED_EXPLORATORY'
    assert d(block(0.0299999, 0.001, [0.1, 0.1, 0.1, 0.1]))['status'] == 'NO_DETECTED_GAIN'
    assert d(block(0.05, 0.0, [0.1, 0.1, 0.1, 0.1]))['status'] == 'INCONCLUSIVE'
    assert d(block(0.05, 0.01, [0.2, 0.0, 0.0, 0.0]))['status'] == 'INCONCLUSIVE'
    assert d(block(0.05, 0.01, [0.2, 0.1, -0.0300001, 0.0]))['status'] == 'INCONCLUSIVE'
    assert d(block(-0.2, -0.3, [0, 0, 0, 0]))['status'] == 'NO_DETECTED_GAIN'
    assert d(block(0.5, 0.4, [0.5] * 4), integrity_ok=False)['status'] == 'BLOCKED'
    assert RC.program_decision('SIGNAL_SUPPORTED_EXPLORATORY') == 'PARTIAL_EVIDENCE'
    assert RC.program_decision('NO_DETECTED_GAIN') == 'PARTIAL_EVIDENCE'
    assert RC.program_decision('BLOCKED') == 'BLOCKED'

@test
def t34_person_bootstrap_is_paired_and_person_clustered():
    t, _ = fold_inputs('null', 23)
    sc = Q.Scorer(t, 100)
    for c, name in enumerate(sc.names):
        assert sc.counts[c].shape == (100, len(sc.subjects[name]))
        assert np.all(sc.counts[c].sum(axis=1) == len(sc.subjects[name]))
    preds = [np.zeros(sc.y[n].size, int) for n in sc.names]
    conf = sc.conf(preds)
    same = sc.contrast(conf, conf)
    assert same['M']['point'] == 0 and same['M']['ci95'] == [0.0, 0.0]

@test
def t35_only_grades_0_to_2_are_eligible():
    t, store = fold_inputs('null', 24)
    assert np.sum(t['label'] == 3) == 4 and np.sum(t['label'] == -1) == 4
    sc = Q.Scorer(t, 10)
    for c, name in enumerate(sc.names):
        assert set(np.unique(sc.y[name])) == {0, 1, 2}
    assert len(store) == t['label'].size

@test
def t36_pooled_recall_suppression():
    good = {n: np.array([[[10, 5, 5], [6, 10, 6], [5, 5, 20]]]) for n in C.COHORTS[:1]}
    rows = RC.pooled_recall_rows(good)
    assert all((isinstance(r, dict) for r in rows.values()))
    assert math.isclose(sum(rows['grade_0'].values()), 1.0)
    small = {'a': np.array([[[10, 0, 3], [6, 10, 6], [5, 5, 20]]])}
    rows = RC.pooled_recall_rows(small)
    assert rows['grade_0'] == 'SUPPRESSED' and isinstance(rows['grade_1'], dict)
    assert RC.pooled_recall_rows({'a': np.zeros((1, 3, 3))})['grade_2'] == 'SUPPRESSED'

@test
def t37_s_parity_gate():
    folds = [{'selected': {'all': 2}}, {'selected': {'all': 2}}, {'selected': {'all': 4}}, {'selected': {'all': 2}}]
    names = list(C.COHORTS)
    pts = np.array([0.5, 0.6, 0.7, 0.4])
    p1 = {'selected_configs': {'S': {n: {'all': Q.CONFIGS[f['selected']['all']][0]} for n, f in zip(names, folds)}}, 'arms_inductive_strong': {'S': {'macro_f1': {'per_cohort': {n: {'point': float(v)} for n, v in zip(names, pts)}, 'M': {'point': float(pts.mean())}}}}}
    assert RN.s_parity(folds, pts, p1, Q.CONFIGS, names)['status'] == 'PASS'
    raises('REP_S_PARITY:BMCLab', RN.s_parity, folds, pts + np.array([0, 2e-06, 0, 0]), p1, Q.CONFIGS, names)
    folds[0]['selected']['all'] = 3
    raises('REP_S_PARITY:3DGait', RN.s_parity, folds, pts, p1, Q.CONFIGS, names)

@test
def t39_null_world_gives_no_false_signal():
    _, _, body = campaign('null')
    sn = body['contrasts']['S_plus_N_minus_S']['walk_eval']['M']
    assert body['decision']['S_plus_N_vs_S']['status'] != 'SIGNAL_SUPPORTED_EXPLORATORY', sn
    assert sn['point'] < 0.03, sn
    assert body['decision']['program'] == 'PARTIAL_EVIDENCE' and body['S_parity']['status'] == 'NOT_CHECKED'
    assert body['arms']['S']['macro_f1']['M']['point'] > 0.8

@test
def t40_known_incremental_rotation_signal_is_detected():
    # 'signal_untied', not 'signal': in 'signal' one rotation stratum is an exact three-way grade tie, so N's score
    # (4/9 or 5/9 per cohort) was set by floating-point tie-breaking, not by detection. Threshold and assertions unchanged.
    _, _, body = campaign('signal_untied')
    n = body['arms']['N']['macro_f1']['M']['point']
    s = body['arms']['S']['macro_f1']['M']['point']
    sn = body['contrasts']['S_plus_N_minus_S']['walk_eval']['M']['point']
    assert n > 0.5 and sn > 0.05, (n, s, sn)
    rho = body['label_free_speed_rank_correlation_of_expected_grade']
    assert all((set(v) == {'N', 'S_plus_N'} for v in rho.values()))

@test
def t41_campaign_is_deterministic():

    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k != 'seconds'}
        return obj
    t, store = OS.build_inputs(fab_objs(5, 'null'))
    a = RN.run_campaign(t, store, 'cpu', cfg=FAST_CFG, n_boot=50, with_partial=True)
    b = RN.run_campaign(t, store, 'cpu', cfg=FAST_CFG, n_boot=50, with_partial=True)
    assert RC.render(strip(RC.clean(a))) == RC.render(strip(RC.clean(b)))
    assert all(('seconds' in f for f in a['native_folds'].values()))

@test
def t42_eligibility_identical_across_arms():
    t, _, body = campaign('null')
    sc = Q.Scorer(t, 10)
    for site in C.COHORTS:
        assert body['denominators'][site]['walks_eligible'] == int(sc.masks[site].sum())
    assert set(body['arms']) == set(RN.ARMS)
    assert set(body['contrasts']) == {'S_plus_N_minus_S', 'N_minus_S', 'S_plus_NOISE3_minus_S', 'S_plus_N_minus_S_plus_NOISE3'}
    keys = {a: set(body['arms'][a]['macro_f1']['per_cohort']) for a in RN.ARMS}
    assert all((k == set(C.COHORTS) for k in keys.values()))

@test
def t44_run_guard_allows_only_pinned_reads_and_result_write():
    repo = REPO_S
    raw = os.path.join(repo, 'data', 'care_pd', 'raw', 'BMCLab_canonical.pkl')
    res = os.path.join(repo, 'aggregate.json')
    tmp = res + '.tmp'
    g = lambda ev, args: RC.guard_decision(ev, args, repo, frozenset([raw]), frozenset([tmp]), frozenset([(tmp, res)]))
    g('open', (raw, 'r', 0))
    g('open', (tmp, 'x', os.O_CREAT | os.O_WRONLY))
    g('os.rename', (tmp, res, None, None))
    g('open', (os.path.join(repo, 'results', 'summary', 'gait_floor_p1_strong_models.json'), 'r', 0))
    g('pickle.find_class', ('numpy', 'dtype'))
    for ev, args in (('open', (os.path.join(repo, 'data', 'care_pd', 'other.pkl'), 'r', 0)), ('open', (os.path.join(repo, 'logs', 'x.local.tsv'), 'r', 0)), ('open', (os.path.join(repo, 'w.pt'), 'r', 0)), ('open', (os.path.join(os.path.dirname(repo), 'sibling', 'f.txt'), 'r', 0)), ('open', (res, 'w', 0)), ('os.rename', (tmp, os.path.join(repo, 'other.json'), None, None)), ('pickle.find_class', ('os', 'system')), ('os.remove', (res,)), ('socket.getaddrinfo', ('h', 1))):
        try:
            g(ev, args)
        except RC.GuardError:
            continue
        raise AssertionError('guard allowed ' + ev + ' ' + str(args[0]))
    import tempfile
    tmpdir = tempfile.mkdtemp()
    fd_tmp, fd_repo = (os.open(tmpdir, os.O_RDONLY), os.open(repo, os.O_RDONLY))
    try:
        g('os.remove', ('relative_name', fd_tmp))
        g('os.rmdir', ('relative_dir', fd_tmp))
        for ev, args in (('os.remove', ('relative_name', fd_repo)), ('os.rmdir', ('scripts', fd_repo))):
            try:
                g(ev, args)
            except RC.GuardError:
                continue
            raise AssertionError('guard allowed a repo-relative dir_fd mutation')
    finally:
        os.close(fd_tmp)
        os.close(fd_repo)
        os.rmdir(tmpdir)

def _max_list_len(obj):
    if isinstance(obj, dict):
        return max([0] + [_max_list_len(v) for v in obj.values()])
    if isinstance(obj, list):
        return max([len(obj)] + [_max_list_len(v) for v in obj])
    return 0

def mock_output():
    _, _, body = campaign('null')
    return {'schema': RC.SCHEMA, 'status': RC.STATUS_COMPLETE, 'program_status': RC.PROGRAM_STATUS, **body, 'P_published_encoder': PE.p_status_record(), 'native_design': {'config': RN.CONFIG}, 'limits': OS.LIMITS, 'release_provenance': {'portable_real_data_parity': 'NOT_VERIFIED'}}

@test
def t45_output_is_aggregate_only_and_small():
    out = mock_output()
    text = RC.render(out)
    n = len(text.encode('utf-8'))
    assert n < 0.7 * RC.SUMMARY_CAP, n
    assert _max_list_len(RC.clean(out)) <= 32
    low = text.lower()
    for bad in ('fab0_', 'fab1_', '"w0"', 'state_dict', '/data/care_pd', '.pkl"', 'password'):
        assert bad not in low, bad
    data = json.loads(text)
    assert data['status'] == RC.STATUS_COMPLETE and data['program_status'] == 'F2_PARTIAL'
    assert data['decision']['S_plus_P_vs_S']['status'] == 'NOT_RUN_P_BLOCKED'
    print('    mock result bytes', n)

@test
def t46_render_refuses_identifiers_and_nonfinite():
    raises('REP_RESULT_FORBIDDEN_KEY:subject', RC.render, {'a': {'subject_index': 1}})
    raises('REP_RESULT_FORBIDDEN_KEY:embedding', RC.render, {'embedding_rows': [1]})
    raises('REP_RESULT_FORBIDDEN_KEY:token', RC.render, {'x': [{'api_token': 'y'}]})
    raises('REP_RESULT_TOO_LARGE', RC.render, {'x': 'a' * (RC.SUMMARY_CAP + 1)})
    assert json.loads(RC.render({'a': float('nan'), 'b': np.float32(0.5), 'c': np.int64(3), 'd': (1, 2)})) == {'a': None, 'b': 0.5, 'c': 3, 'd': [1, 2]}
    try:
        RC.render({'a': object()})
    except TypeError:
        pass
    else:
        raise AssertionError('non-aggregate object accepted')

@test
def t47_published_encoder_record_is_blocked_and_inert():
    r = PE.p_status_record()
    assert r['status'] == 'BLOCKED_PRETRAINED_BASELINE' and r['downloads'] == 0 and (r['pretrained_loads'] == 0)
    assert [c['name'] for c in r['candidates']] == ['ACTOR', 'TEMOS']
    assert all((c['status'] == 'BLOCKED' and c['pretrained_archive']['downloaded'] is False for c in r['candidates']))
    assert r['arms_not_run'] == ['P', 'S_plus_P']
    r['candidates'][0]['status'] = 'X'
    assert PE.p_status_record()['candidates'][0]['status'] == 'BLOCKED'
    src = (HERE / 'representation_public_encoder.py').read_text(encoding='utf-8')
    assert 'import torch' not in src and 'urlopen' not in src and ('requests' not in src)

@test
def t50_row_alignment_check_in_build_inputs():
    objs = fab_objs(41, 'null')
    t, store = OS.build_inputs(objs)
    assert len(store) == t['label'].size and np.all(store.lengths >= 1)
    bad = fab_objs(41, 'null')
    first = next(iter(bad['3DGait']))
    bad['3DGait'][first]['w0']['pose'][0, 0] = np.nan
    raises('NON_FINITE', P.check_stops, OS.build_inputs(bad)[0])

def main():
    t0 = time.monotonic()
    failed = 0
    for fn in TESTS:
        t1 = time.monotonic()
        try:
            fn()
            print(f'PASS {fn.__name__} ({time.monotonic() - t1:.1f}s)')
        except (RealDataAccess, TripwireError) as exc:
            print(f'FAIL {fn.__name__}: tripwire {exc}')
            failed += 1
            break
        except Exception:
            print(f'FAIL {fn.__name__}')
            traceback.print_exc(limit=4)
            failed += 1
    print(f'elapsed {time.monotonic() - t0:.0f}s')
    if failed:
        print(f'REP_SYNTHETIC_FAILED n_failed={failed}')
        return 1
    print(f'REP_SYNTHETIC_ALL_PASS n_tests={len(TESTS)}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
