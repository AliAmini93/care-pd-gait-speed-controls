"""Portable analysis kernel. See docs/provenance_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import hashlib
import io
import math
import pickle
import pickletools
import re
import sys
from pathlib import Path
import numpy as np
import aggregate_core as IC
import common as C
DOI = '10.5683/SP3/TWIKMK'
ROOT_DIR = None
COHORTS = ('3DGait', 'BMCLab', 'T-SDU-PD', 'PD-GaM')
ROOT_PINS = {'3DGait': ('3DGait.pkl', 926879, 8066161, 'a4fb5b909b4a7d913530a8ac82ce9c99', 'v1.0-v4.0'), 'BMCLab': ('BMCLab.pkl', 926877, 127210064, '8744c40c868de8f32138673cc0b05485', 'v1.0-v4.0'), 'T-SDU-PD': ('T-SDU-PD.pkl', 926878, 27019026, 'a1b978b7bd5e1d4edbb1306218c22d26', 'v1.0-v4.0'), 'PD-GaM': ('PD-GaM.pkl', 1160521, 84704001, '92884135f342f715ab4f5efd21b8ad7b', 'v3.0-v4.0')}
NOT_APPROVED_DATAFILES = (926880,)
CLINICAL_FIELDS = ('UPDRS_GAIT', 'medication', 'other')
MOTION_FIELDS = ('pose', 'trans', 'beta', 'fps')
KNOWN_FIELDS = MOTION_FIELDS + CLINICAL_FIELDS
GRADE_FIELD = 'UPDRS_GAIT'
SourceError = C.SourceError

def root_path(cohort):
    return ROOT_DIR / cohort / ROOT_PINS[cohort][0]

def file_digests(path, chunk=1 << 20):
    md5, sha = (hashlib.md5(), hashlib.sha256())
    n = 0
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            md5.update(block)
            sha.update(block)
            n += len(block)
    return (n, md5.hexdigest(), sha.hexdigest())

def sanitize_reason(text, limit=100):
    return re.sub('[^A-Za-z0-9_.:]', '', str(text))[:limit] or 'UNSPECIFIED'
STRICT_REFUSED_OPCODES = frozenset(C.REFUSED_OPCODES) | {'EMPTY_SET', 'ADDITEMS', 'FROZENSET'}

def format_sniff(head):
    head = bytes(head[:8])
    if head[:1] == b'\x80':
        return 'pickle_protocol_2_or_higher'
    if head[:2] == b'\x1f\x8b':
        return 'gzip'
    if head[:3] == b'BZh':
        return 'bzip2'
    if head[:6] == b'\xfd7zXZ\x00':
        return 'xz'
    if head[:4] == b'\x04"M\x18':
        return 'lz4'
    if head[:1] == b'x' and head[1:2] in (b'\x01', b'^', b'\x9c', b'\xda'):
        return 'zlib'
    if head[:1] in (b'<',):
        return 'html_or_xml'
    if head[:1] in (b'{', b'['):
        return 'json_like'
    return 'other'

def static_scan(data):
    out = {'ok': False, 'reason': None, 'protocol': None, 'globals_seen': [], 'opcode_counts': {}, 'n_opcodes': 0, 'ends_with_stop': False, 'trailing_bytes': None, 'strict_refused_opcodes': []}
    try:
        proto, seen = C.scan_opcodes(data)
    except SourceError as exc:
        out['reason'] = sanitize_reason(exc)
        return out
    out['protocol'] = proto
    out['globals_seen'] = [f'{m}.{a}' for m, a in seen]
    counts, last = ({}, None)
    try:
        for op, _arg, pos in pickletools.genops(data):
            counts[op.name] = counts.get(op.name, 0) + 1
            last = (op.name, pos)
    except Exception:
        out['reason'] = 'PICKLE_OPCODE_CENSUS_FAILED'
        return out
    out['opcode_counts'] = dict(sorted(counts.items()))
    out['n_opcodes'] = int(sum(counts.values()))
    out['ends_with_stop'] = bool(last is not None and last[0] == 'STOP')
    out['trailing_bytes'] = int(len(data) - (last[1] + 1)) if out['ends_with_stop'] else None
    out['strict_refused_opcodes'] = sorted(set(counts) & STRICT_REFUSED_OPCODES)
    if not out['ends_with_stop']:
        out['reason'] = 'PICKLE_NO_FINAL_STOP'
    elif out['trailing_bytes'] != 0:
        out['reason'] = 'PICKLE_TRAILING_BYTES'
    elif out['strict_refused_opcodes']:
        out['reason'] = 'PICKLE_OPCODE_REFUSED_STRICT'
    else:
        out['ok'] = True
    return out
_SAFE_CALLABLES = (C._safe_reconstruct, C._safe_frombuffer, C._safe_scalar, C._safe_dtype)
_STATE_ATOMS = (int, str, type(None))

def _refuse(code):
    raise SourceError(code)

def _is_safe_callable(func):
    return any((func is g for g in _SAFE_CALLABLES))

def array_dtype_ok(dt):
    return bool(isinstance(dt, np.dtype) and (not dt.hasobject) and (dt.names is None) and (dt.subdtype is None) and (dt.kind in 'fiub') and (dt.byteorder in ('=', '<', '|')) and (f'{dt.kind}{dt.itemsize}' in C.ALLOWED_DTYPES))

def _check_array_state(inst, state):
    if type(inst) is not np.ndarray:
        _refuse('PICKLE_BUILD_TARGET_REFUSED')
    if type(state) is not tuple or len(state) != 5:
        _refuse('PICKLE_ARRAY_STATE_REFUSED')
    ver, shape, dt, fortran, raw = state
    if type(ver) is not int or ver != 1 or type(fortran) is not bool:
        _refuse('PICKLE_ARRAY_STATE_REFUSED')
    if type(shape) is not tuple or len(shape) > 6 or any((type(n) is not int or n < 0 for n in shape)):
        _refuse('PICKLE_ARRAY_STATE_REFUSED')
    if not array_dtype_ok(dt):
        _refuse('PICKLE_ARRAY_DTYPE_REFUSED')
    if type(raw) is not bytes or len(raw) != math.prod(shape) * dt.itemsize:
        _refuse('PICKLE_ARRAY_STATE_REFUSED')

def _check_dtype_state(inst, state):
    if not array_dtype_ok(inst):
        _refuse('PICKLE_DTYPE_REFUSED')
    if type(state) is not tuple or not all((type(x) in _STATE_ATOMS for x in state)):
        _refuse('PICKLE_DTYPE_STATE_REFUSED')
    want = inst.__reduce__()[2]
    if len(state) != len(want) or any((type(a) is not type(b) or a != b for a, b in zip(state, want))):
        _refuse('PICKLE_DTYPE_STATE_REFUSED')

def _check_key(key):
    if type(key) not in (str, int):
        _refuse('PICKLE_DICT_KEY_REFUSED')

class StrictUnpickler(pickle._Unpickler):
    dispatch = dict(pickle._Unpickler.dispatch)

    def find_class(self, module, name):
        if (module, name) not in C.ALLOWED_GLOBALS:
            _refuse('PICKLE_GLOBAL_REFUSED')
        return C._RESOLVED[name]

    def persistent_load(self, pid):
        _refuse('PICKLE_PERSISTENT_ID_REFUSED')

    def _strict_reduce(self):
        stack = self.stack
        args = stack.pop()
        func = stack[-1]
        if not _is_safe_callable(func) or type(args) is not tuple:
            _refuse('PICKLE_REDUCE_TARGET_REFUSED')
        stack[-1] = func(*args)

    def _strict_build(self):
        stack = self.stack
        state = stack.pop()
        inst = stack[-1]
        if type(inst) is np.ndarray:
            _check_array_state(inst, state)
        elif isinstance(inst, np.dtype):
            _check_dtype_state(inst, state)
        else:
            _refuse('PICKLE_BUILD_TARGET_REFUSED')
        inst.__setstate__(state)

    def _strict_setitem(self):
        stack = self.stack
        value = stack.pop()
        key = stack.pop()
        target = stack[-1]
        if type(target) is not dict:
            _refuse('PICKLE_SETITEM_TARGET_REFUSED')
        _check_key(key)
        target[key] = value

    def _strict_setitems(self):
        items = self.pop_mark()
        target = self.stack[-1]
        if type(target) is not dict or len(items) % 2:
            _refuse('PICKLE_SETITEM_TARGET_REFUSED')
        for i in range(0, len(items), 2):
            _check_key(items[i])
            target[items[i]] = items[i + 1]

    def _strict_append(self):
        stack = self.stack
        value = stack.pop()
        target = stack[-1]
        if type(target) is not list:
            _refuse('PICKLE_APPEND_TARGET_REFUSED')
        target.append(value)

    def _strict_appends(self):
        items = self.pop_mark()
        target = self.stack[-1]
        if type(target) is not list:
            _refuse('PICKLE_APPEND_TARGET_REFUSED')
        target.extend(items)

def _install_dispatch():
    code = {op.name: ord(op.code) for op in pickletools.opcodes}
    table = StrictUnpickler.dispatch
    for name in STRICT_REFUSED_OPCODES:
        if name in code:
            table[code[name]] = (lambda n: lambda self: _refuse('PICKLE_OPCODE_REFUSED'))(name)
    table[code['REDUCE']] = StrictUnpickler._strict_reduce
    table[code['BUILD']] = StrictUnpickler._strict_build
    table[code['SETITEM']] = StrictUnpickler._strict_setitem
    table[code['SETITEMS']] = StrictUnpickler._strict_setitems
    table[code['APPEND']] = StrictUnpickler._strict_append
    table[code['APPENDS']] = StrictUnpickler._strict_appends
_install_dispatch()
_SCALAR_TYPES = (type(None), bool, int, float, str, bytes)

def tree_check(obj, max_nodes=50000000, max_depth=12):
    stack, n = ([(obj, 0)], 0)
    while stack:
        x, depth = stack.pop()
        n += 1
        if n > max_nodes or depth > max_depth:
            _refuse('TREE_TOO_LARGE_OR_DEEP')
        t = type(x)
        if t in _SCALAR_TYPES:
            continue
        if t is np.ndarray:
            if not array_dtype_ok(x.dtype):
                _refuse('TREE_ARRAY_DTYPE_REFUSED')
            continue
        if isinstance(x, np.generic):
            if x.dtype.kind not in 'fiub':
                _refuse('TREE_SCALAR_REFUSED')
            continue
        if t is dict:
            for k, v in x.items():
                if type(k) not in (str, int):
                    _refuse('TREE_DICT_KEY_REFUSED')
                stack.append((v, depth + 1))
            continue
        if t in (list, tuple):
            stack.extend(((v, depth + 1) for v in x))
            continue
        _refuse('TREE_TYPE_REFUSED')

def strict_loads(data):
    scan = static_scan(data)
    if not scan['ok']:
        _refuse(scan['reason'] or 'PICKLE_STATIC_SCAN_FAILED')
    try:
        obj = StrictUnpickler(io.BytesIO(data)).load()
    except SourceError:
        raise
    except Exception:
        raise SourceError('PICKLE_STRICT_LOAD_FAILED') from None
    tree_check(obj)
    return obj

def trees_equal(a, b, skip_value_keys=CLINICAL_FIELDS):
    stack = [(a, b)]
    while stack:
        x, y = stack.pop()
        if type(x) is not type(y):
            return False
        if type(x) is dict:
            if len(x) != len(y) or set(x) != set(y):
                return False
            for k in x:
                if k in skip_value_keys:
                    continue
                stack.append((x[k], y[k]))
        elif type(x) in (list, tuple):
            if len(x) != len(y):
                return False
            stack.extend(zip(x, y))
        elif type(x) is np.ndarray:
            if x.dtype != y.dtype or x.shape != y.shape:
                return False
            if not np.array_equal(x, y, equal_nan=x.dtype.kind == 'f'):
                return False
        elif isinstance(x, np.generic):
            if x.dtype != y.dtype or not (x == y or (x.dtype.kind == 'f' and x != x and (y != y))):
                return False
        elif type(x) is float:
            if not (x == y or (x != x and y != y)):
                return False
        elif x != y:
            return False
    return True
_FIELD_NAME_RX = re.compile('^[A-Za-z][A-Za-z0-9_]{0,23}$')

def structure_census(obj):
    out = {'top_level_dict': type(obj) is dict}
    if type(obj) is not dict:
        return out
    subj = {'int': 0, 'str': 0, 'other': 0}
    norm, n_bad_subject = ({}, 0)
    walks = rec_dict = walk_key_str = 0
    present = {f: 0 for f in KNOWN_FIELDS}
    unknown, recs_unknown = ({}, 0)
    pose_ok = trans_ok = frames_equal = beta_zero = fps_num = 0
    beta_shape = {'vector_10': 0, 'per_frame_10': 0, 'row_1x10': 0, 'other': 0}
    fps_values, frames, dtypes = ({}, [], set())
    for s, wd in obj.items():
        subj['int' if type(s) is int else 'str' if type(s) is str else 'other'] += 1
        norm[str(s)] = norm.get(str(s), 0) + 1
        if type(wd) is not dict:
            n_bad_subject += 1
            continue
        for w, rec in wd.items():
            walks += 1
            walk_key_str += int(type(w) is str)
            if type(rec) is not dict:
                continue
            rec_dict += 1
            extra = False
            for k in rec:
                if k in present:
                    present[k] += 1
                else:
                    extra = True
                    name = k if type(k) is str and _FIELD_NAME_RX.match(k) else '<non-identifier>'
                    unknown[name] = unknown.get(name, 0) + 1
            recs_unknown += int(extra)
            pose = rec['pose'] if 'pose' in rec else None
            trans = rec['trans'] if 'trans' in rec else None
            p_ok = type(pose) is np.ndarray and pose.dtype.kind == 'f' and (pose.ndim == 2) and (pose.shape[1] == 72)
            t_ok = type(trans) is np.ndarray and trans.dtype.kind == 'f' and (trans.ndim == 2) and (trans.shape[1] == 3)
            pose_ok += int(p_ok)
            trans_ok += int(t_ok)
            if p_ok:
                dtypes.add('pose:' + pose.dtype.str)
            if t_ok:
                dtypes.add('trans:' + trans.dtype.str)
            if p_ok and t_ok and (pose.shape[0] == trans.shape[0]):
                frames_equal += 1
                frames.append(int(pose.shape[0]))
            beta = rec['beta'] if 'beta' in rec else None
            if type(beta) is np.ndarray and beta.dtype.kind in 'fiu':
                n_t = pose.shape[0] if p_ok else -1
                if beta.shape == (10,):
                    beta_shape['vector_10'] += 1
                elif beta.ndim == 2 and beta.shape == (n_t, 10):
                    beta_shape['per_frame_10'] += 1
                elif beta.shape == (1, 10):
                    beta_shape['row_1x10'] += 1
                else:
                    beta_shape['other'] += 1
                beta_zero += int(bool(np.all(beta == 0)))
            else:
                beta_shape['other'] += 1
            fps = C._as_number(rec['fps']) if 'fps' in rec else None
            if fps is not None and fps == fps:
                fps_num += 1
                fps_values[float(fps)] = fps_values.get(float(fps), 0) + 1
    out.update({'subject_key_types': subj, 'subject_key_str_collisions': sum((1 for v in norm.values() if v > 1)), 'subjects_not_dict': n_bad_subject, 'subjects': len(obj), 'walks': walks, 'walk_keys_str': walk_key_str, 'records_dict': rec_dict, 'field_present': present, 'records_with_unknown_fields': recs_unknown, 'unknown_field_names': unknown, 'pose_conforming': pose_ok, 'trans_conforming': trans_ok, 'pose_trans_same_frames': frames_equal, 'beta_shape': beta_shape, 'beta_all_zero': beta_zero, 'fps_numeric': fps_num, 'fps_values': fps_values, 'frames': frames, 'dtypes': sorted(dtypes)})
    return out

def label_class(rec):
    if GRADE_FIELD not in rec:
        return ('ABSENT_KEY', None)
    x = rec[GRADE_FIELD]
    if x is None:
        return ('ABSENT_NONE', None)
    v = C._as_number(x)
    if v is None:
        return ('INVALID', None)
    if isinstance(v, float) and v != v:
        return ('ABSENT_NAN', None)
    if int(v) != v or not 0 <= int(v) <= 3:
        return ('INVALID', None)
    return ('GRADE', int(v))

def compare_grades(root, canonical):
    (ka, va), (kb, vb) = (root, canonical)
    if ka == 'INVALID' or kb == 'INVALID':
        return 'INVALID_VALUE'
    if ka == 'GRADE' and kb == 'GRADE':
        return 'EQUAL_GRADE' if va == vb else 'CHANGED_GRADE'
    if ka == 'GRADE':
        return 'GRADED_ROOT_ONLY'
    if kb == 'GRADE':
        return 'GRADED_CANONICAL_ONLY'
    return 'EQUAL_ABSENT' if ka == kb else 'ABSENT_ENCODING_DIFFERS'

def field_value(rec, field):
    if field not in rec:
        return ('MISSING_KEY',)
    x = rec[field]
    if x is None:
        return ('NONE',)
    if isinstance(x, (bool, np.bool_)):
        return ('BOOL', bool(x))
    if type(x) is str:
        return ('STR', x)
    if type(x) is bytes:
        return ('BYTES', x)
    v = C._as_number(x)
    if v is not None:
        return ('NAN',) if v != v else ('NUM', float(v))
    return ('UNCOMPARABLE', type(x).__name__)
_ABSENT_FORMS = (('MISSING_KEY',), ('NONE',), ('NAN',))

def compare_field(a, b):
    if a[0] == 'UNCOMPARABLE' or b[0] == 'UNCOMPARABLE':
        return 'UNCOMPARABLE'
    if a == b:
        return 'EQUAL'
    if a in _ABSENT_FORMS and b in _ABSENT_FORMS:
        return 'ABSENT_ENCODING_DIFFERS'
    if a in _ABSENT_FORMS or b in _ABSENT_FORMS:
        return 'ONE_SIDE_ABSENT'
    return 'DIFFERS'
LT5, GE5 = ('LT5', 'GE5')
COARSE_TOKENS = (LT5, GE5)

def has_small(*partitions):
    return any((0 < int(v) < IC.SMALL_CELL for p in partitions for v in p.values()))

def coarse_cell(n):
    n = int(n)
    return 0 if n == 0 else LT5 if n < IC.SMALL_CELL else GE5

def render_partition(cells, coarse):
    cells = {k: int(v) for k, v in cells.items()}
    if coarse:
        return {'mode': 'coarse', 'cells': {k: coarse_cell(v) for k, v in cells.items()}}
    if has_small(cells):
        _refuse('DISCLOSURE_EXACT_WITH_SMALL_CELL')
    return {'mode': 'exact', 'cells': cells}

def count_out(n, coarse):
    n = int(n)
    if coarse:
        return coarse_cell(n)
    if 0 < n < IC.SMALL_CELL:
        _refuse('DISCLOSURE_EXACT_WITH_SMALL_CELL')
    return n

def share_band(k, n):
    if n <= 0:
        return 'UNDEFINED'
    s = k / n
    return 'GE_0.95' if s >= 0.95 else 'LE_0.50' if s <= 0.5 else 'BETWEEN_0.50_AND_0.95'

def share_out(k, n, coarse):
    if n <= 0:
        return None
    return share_band(k, n) if coarse else round(k / n, 4)

def band5(n):
    lo = int(n) // 5 * 5
    return f'{lo}_to_{lo + 4}'

def disclosure_problems(obj, path=''):
    problems = []
    if isinstance(obj, dict):
        if 'cells' in obj and isinstance(obj['cells'], dict):
            coarse = obj.get('mode') == 'coarse'
            for k, v in obj['cells'].items():
                if isinstance(v, bool):
                    problems.append(f'bool-cell:{path}/{k}')
                elif isinstance(v, int):
                    if 0 < v < IC.SMALL_CELL or (coarse and v != 0):
                        problems.append(f'cell:{path}/{k}')
                elif v not in COARSE_TOKENS:
                    problems.append(f'cell-token:{path}/{k}')
        for k, v in obj.items():
            problems.extend(disclosure_problems(v, f'{path}/{k}'))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            problems.extend(disclosure_problems(v, f'{path}[{i}]'))
    return problems

def quantiles(values, min_n=IC.SMALL_CELL, digits=4):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size < min_n:
        return None
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return {'q25': round(float(q25), digits), 'median': round(float(q50), digits), 'q75': round(float(q75), digits)}
