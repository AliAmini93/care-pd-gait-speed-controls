"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import p1_core as Q
SCHEMA = 'gait_representation_final_controlled_v1'
STATUS_COMPLETE = 'GAIT_REPRESENTATION_F2_PARTIAL_COMPLETE'
PROGRAM_STATUS = 'F2_PARTIAL'
SUMMARY_CAP = 262144
N_JOINTS = 24
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
TARGET_FPS = 30.0
WINDOW = 60
FPS_TOL = 1e-09
SLERP_LINEAR_DOT = 0.9995
UP_AXIS = 1
FORWARD_AXIS = 2
ELIGIBLE = (0, 1, 2)
MARGIN = 0.03
SITE_FLOOR = -0.03
MIN_POSITIVE_SITES = 2
N_NOISE_COLUMNS = 3
NOISE_SEED = (20261105, 77)
SUPPRESS_LOW, SUPPRESS_HIGH = (1, 4)

def rotvec_to_matrix(rv):
    rv = np.asarray(rv, dtype=np.float64)
    if rv.shape[-1] != 3:
        raise C.SourceError('REP_ROTVEC_SHAPE')
    theta2 = np.sum(rv * rv, axis=-1)
    theta = np.sqrt(theta2)
    small = theta < 1e-06
    safe = np.where(small, 1.0, theta)
    a = np.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, np.sin(safe) / safe)
    b = np.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - np.cos(safe)) / (safe * safe))
    x, y, z = (rv[..., 0], rv[..., 1], rv[..., 2])
    zero = np.zeros_like(x)
    K = np.stack([np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1), np.stack([-y, x, zero], -1)], -2)
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + a[..., None, None] * K + b[..., None, None] * (K @ K)

def matrix_to_6d(R):
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)

def sixd_to_matrix(x):
    x = np.asarray(x, dtype=np.float64)
    a1, a2 = (x[..., 0:3], x[..., 3:6])
    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    if np.any(n1 < 1e-08):
        raise C.SourceError('REP_6D_DEGENERATE')
    b1 = a1 / n1
    u2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    n2 = np.linalg.norm(u2, axis=-1, keepdims=True)
    if np.any(n2 < 1e-08):
        raise C.SourceError('REP_6D_DEGENERATE')
    b2 = u2 / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)

def yaw_matrix(psi):
    psi = np.asarray(psi, dtype=np.float64)
    c, s = (np.cos(psi), np.sin(psi))
    o, z = (np.ones_like(c), np.zeros_like(c))
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)

def window_heading(root):
    f = np.asarray(root, dtype=np.float64)[..., :, FORWARD_AXIS]
    fx, fz = (float(f[..., 0].sum()), float(f[..., 2].sum()))
    if math.hypot(fx, fz) < 1e-09:
        return 0.0
    return math.atan2(fx, fz)

def canonical_root(root):
    return yaw_matrix(-window_heading(root)) @ np.asarray(root, dtype=np.float64)

def _slerp_quat(q0, q1, w):
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    lin = dot > SLERP_LINEAR_DOT
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_t = np.where(lin, 1.0, np.sin(theta))
    w0 = np.where(lin, 1.0 - w, np.sin((1.0 - w) * theta) / sin_t)
    w1 = np.where(lin, w, np.sin(w * theta) / sin_t)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

def resample_rotations(R, fps, target=TARGET_FPS):
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 4 or R.shape[-2:] != (3, 3) or R.shape[0] < 1:
        raise C.SourceError('REP_RESAMPLE_SHAPE')
    if not (np.isfinite(fps) and 1.0 <= fps <= 1000.0):
        raise C.SourceError('REP_FPS_INVALID')
    T, J = R.shape[:2]
    ratio = fps / target
    k = round(ratio)
    if k >= 1 and abs(ratio - k) < FPS_TOL:
        return R[::k].copy()
    n_out = int(math.floor((T - 1) * target / fps + 1e-09)) + 1
    u = np.arange(n_out) * (fps / target)
    i0 = np.minimum(np.floor(u + 1e-09).astype(int), T - 1)
    w = u - i0
    exact = (w < 1e-09) | (i0 >= T - 1)
    out = R[i0].copy()
    need = np.flatnonzero(~exact)
    if need.size:
        flat = R.reshape(-1, 3, 3)
        q = Rotation.from_matrix(flat).as_quat().reshape(T, J, 4)
        qa, qb = (q[i0[need]], q[i0[need] + 1])
        qs = _slerp_quat(qa, qb, w[need][:, None, None])
        out[need] = Rotation.from_quat(qs.reshape(-1, 4)).as_matrix().reshape(need.size, J, 3, 3)
    return out

def walk_features(pose, fps):
    pose = np.asarray(pose)
    if pose.ndim != 2 or pose.shape[1] != 3 * N_JOINTS or pose.shape[0] < 1:
        raise C.SourceError('REP_POSE_SHAPE')
    if pose.dtype.kind != 'f':
        raise C.SourceError('REP_POSE_DTYPE')
    if not np.isfinite(pose).all():
        raise C.SourceError('REP_POSE_NONFINITE')
    R = rotvec_to_matrix(pose.astype(np.float64).reshape(-1, N_JOINTS, 3))
    R = resample_rotations(R, float(fps))
    return (R[:, 0].astype(np.float32), matrix_to_6d(R[:, 1:]).astype(np.float32))

class WalkStore:

    def __init__(self, roots, locals_):
        if len(roots) != len(locals_) or not roots:
            raise C.SourceError('REP_STORE_SHAPE')
        for r, l6 in zip(roots, locals_):
            if r.ndim != 3 or r.shape[1:] != (3, 3) or l6.shape != (r.shape[0], N_JOINTS - 1, 6) or (r.shape[0] < 1):
                raise C.SourceError('REP_STORE_SHAPE')
        self.roots, self.locals = (list(roots), list(locals_))
        self.lengths = np.array([r.shape[0] for r in roots], dtype=int)

    def __len__(self):
        return len(self.roots)

def window_array(store, i, start, length=WINDOW):
    K = int(store.lengths[i])
    if not 0 <= start < K:
        raise C.SourceError('REP_WINDOW_START')
    n = min(length, K - start)
    X = np.zeros((length, N_JOINTS, 6), dtype=np.float32)
    root = canonical_root(store.roots[i][start:start + n])
    X[:n, 0] = matrix_to_6d(root).astype(np.float32)
    X[:n, 1:] = store.locals[i][start:start + n]
    mask = np.zeros(length, dtype=bool)
    mask[:n] = True
    return (X, mask)

def crop_start(K, rng, length=WINDOW):
    return int(rng.integers(0, K - length + 1)) if K > length else 0

def eval_starts(K, length=WINDOW):
    return list(range(0, int(K), length))

def p1_single_fold(X_src, y_src, site_src, subj_src, X_tgt, warned, configs=Q.CONFIGS, scheme='walk'):
    if X_src.shape[0] != y_src.size or X_src.shape[1] != X_tgt.shape[1]:
        raise C.SourceError('REP_DOWNSTREAM_SHAPE')
    if not np.isin(y_src, ELIGIBLE).all():
        raise C.SourceError('REP_DOWNSTREAM_LABELS')
    scores = Q.inner_scores(X_src, y_src, site_src, subj_src, scheme, configs, warned)
    best = Q.select_config(scores.mean(axis=1), list(range(len(configs))))
    w = Q.train_weights(y_src, subj_src, scheme)
    proba = Q.fit_predict_proba(configs[best], X_src, y_src, w, X_tgt, warned)
    return (proba, configs[best][0], scores)

def noise_columns(n_rows, k=N_NOISE_COLUMNS, seed=NOISE_SEED):
    return np.random.default_rng(list(seed)).standard_normal((n_rows, k))

def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))

def decide_comparison(contrast, integrity_ok=True):
    if not integrity_ok:
        return {'status': 'BLOCKED', 'conditions': None}
    m = contrast['M']
    sites = [float(v['point']) for v in contrast['per_cohort'].values()]
    cond = {'margin_ge_0_03': bool(m['point'] >= MARGIN), 'lower_bound_gt_0': bool(m['ci95'][0] > 0), 'sites_positive': int(sum((d > 0 for d in sites))), 'sites_positive_ge_2': bool(sum((d > 0 for d in sites)) >= MIN_POSITIVE_SITES), 'worst_site_delta': float(min(sites)), 'no_site_below_minus_0_03': bool(min(sites) >= SITE_FLOOR)}
    if not cond['margin_ge_0_03']:
        status = 'NO_DETECTED_GAIN'
    elif cond['lower_bound_gt_0'] and cond['sites_positive_ge_2'] and cond['no_site_below_minus_0_03']:
        status = 'SIGNAL_SUPPORTED_EXPLORATORY'
    else:
        status = 'INCONCLUSIVE'
    return {'status': status, 'conditions': cond}

def program_decision(sn_status):
    if sn_status == 'BLOCKED':
        return 'BLOCKED'
    return 'PARTIAL_EVIDENCE'

def pooled_recall_rows(conf_by_site):
    pooled = sum((np.asarray(c).reshape(-1, 3, 3).sum(axis=0) for c in conf_by_site.values()))
    rows = {}
    for k in range(3):
        row = pooled[k]
        if np.any((row >= SUPPRESS_LOW) & (row <= SUPPRESS_HIGH)) or row.sum() == 0:
            rows[f'grade_{k}'] = 'SUPPRESSED'
        else:
            rows[f'grade_{k}'] = {f'pred_{j}': float(row[j] / row.sum()) for j in range(3)}
    return rows

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
    raise TypeError('REP_RESULT_NOT_AGGREGATE_JSON')
FORBIDDEN_KEY_PARTS = ('subject', 'participant', 'person_id', 'walk_id', 'per_walk', 'embedding', 'state_dict', 'logits_rows', 'prediction_rows', 'password', 'secret', 'token')

def _forbidden_keys(obj, found):
    if isinstance(obj, dict):
        for k, v in obj.items():
            low = str(k).lower()
            found.extend((part for part in FORBIDDEN_KEY_PARTS if part in low))
            _forbidden_keys(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _forbidden_keys(v, found)
    return found

def render(obj):
    data = clean(obj)
    bad = _forbidden_keys(data, [])
    if bad:
        raise C.SourceError('REP_RESULT_FORBIDDEN_KEY:' + bad[0])
    text = json.dumps(data, indent=1, allow_nan=False) + '\n'
    if len(text.encode('utf-8')) > SUMMARY_CAP:
        raise C.SourceError('REP_RESULT_TOO_LARGE')
    return text

def save_aggregate(path, obj):
    path = Path(path)
    if path.exists():
        raise C.SourceError('REP_RESULT_EXISTS')
    text = render(obj)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'x', encoding='utf-8') as fh:
        fh.write(text)
    os.replace(tmp, path)
    return len(text.encode('utf-8'))

class GuardError(RuntimeError):
    pass
_NET = frozenset(('socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyname_ex', 'socket.gethostbyaddr', 'socket.connect', 'socket.bind', 'socket.sendto', 'urllib.Request'))
_PROC = frozenset(('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.spawn', 'os.fork', 'os.forkpty', 'pty.spawn'))
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_MODEL_SUFFIXES = ('.pt', '.pth', '.ckpt', '.safetensors', '.npy', '.npz', '.joblib', '.zip', '.tar', '.tgz')

def _realpath(p):
    if isinstance(p, int) or p is None:
        return None
    try:
        return os.path.realpath(os.path.abspath(os.fsdecode(p)))
    except (TypeError, ValueError, OSError):
        return '\x00unresolvable'

def _under(r, root):
    return r == root or r.startswith(root + os.sep)

def guard_decision(event, args, repo, read_allow, write_allow, rename_allow):
    if event in _NET:
        raise GuardError('REP_GUARD_NETWORK')
    if event in _PROC:
        raise GuardError('REP_GUARD_PROCESS')
    if event == 'pickle.find_class':
        if tuple(args[:2]) not in C.ALLOWED_GLOBALS:
            raise GuardError('REP_GUARD_PICKLE_GLOBAL')
        return
    if event == 'open':
        r = _realpath(args[0] if args else None)
        if r is None:
            return
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = bool(flags & _WRITE_FLAGS) or any((c in mode for c in 'wax+'))
        parent = os.path.dirname(repo)
        if _under(r, parent) and (not _under(r, repo)):
            raise GuardError('REP_GUARD_SIBLING')
        if not _under(r, repo):
            return
        if writing and r not in write_allow:
            raise GuardError('REP_GUARD_REPO_WRITE')
        base = os.path.basename(r).lower()
        if '.local.' in base:
            raise GuardError('REP_GUARD_LOCAL_LEDGER')
        if _under(r, os.path.join(repo, 'data')) and r not in read_allow:
            raise GuardError('REP_GUARD_DATA_PATH')
        if base.endswith(_MODEL_SUFFIXES):
            raise GuardError('REP_GUARD_MODEL_FILE')
        return
    if event == 'os.rename':
        src = _resolve_dir_fd(args[0] if args else None, args[2] if len(args) > 2 else None, repo)
        dst = _resolve_dir_fd(args[1] if len(args) > 1 else None, args[3] if len(args) > 3 else None, repo)
        if src and _under(src, repo) or (dst and _under(dst, repo)):
            if (src, dst) not in rename_allow:
                raise GuardError('REP_GUARD_RENAME')
        return
    if event in _MUTATE_DIR_FD:
        fd_i = _MUTATE_DIR_FD[event]
        r = _resolve_dir_fd(args[0] if args else None, args[fd_i] if fd_i is not None and len(args) > fd_i else None, repo)
        if r and _under(r, repo):
            raise GuardError('REP_GUARD_REPO_MUTATION')
_MUTATE_DIR_FD = {'os.remove': 1, 'os.rmdir': 1, 'os.mkdir': 2, 'shutil.rmtree': 1, 'os.truncate': None, 'os.symlink': None, 'os.link': 2, 'os.chmod': 2, 'os.chown': 3, 'os.utime': 3}

def _resolve_dir_fd(p, dir_fd, repo):
    if p is None or isinstance(p, int):
        return None
    try:
        s = os.fsdecode(p)
    except TypeError:
        return None
    if isinstance(dir_fd, int) and (not isinstance(dir_fd, bool)) and (not os.path.isabs(s)):
        try:
            s = os.path.join(os.readlink('/proc/self/fd/%d' % dir_fd), s)
        except OSError:
            return os.path.join(repo, '\x00unresolvable_dir_fd')
    return _realpath(s)
