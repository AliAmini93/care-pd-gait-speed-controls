"""Shared constants, hash checks and the fail-closed pickle reader.

The CARE-PD pickles are untrusted. They are read only through RestrictedUnpickler,
after an opcode scan, with no CARE-PD code. Subject and walk IDs stay in memory;
nothing participant-level is ever printed or written by any module in this package.
This is a mitigation, not a general guarantee that hostile pickles are harmless.
"""
from __future__ import annotations

import hashlib
import io
import pickle
import pickletools
from pathlib import Path

import numpy as np

HF_COMMIT = "d53fe929aaacbd6822bcad5ef2b1d215401800b7"
HF_BASE = ("https://huggingface.co/datasets/vida-adl/CARE-PD/resolve/"
           f"{HF_COMMIT}/Canonicalized_SMPL_pickles/")
# cohort -> (file, bytes, sha256, Dataverse md5)
FILES = {
    "3DGait": ("3DGait_canonical.pkl", 8060989,
               "c765e5612b5b1993ef7cd71d0e93211c7a416c7557faef5de0c837dfd511acfc",
               "ad607ea73805fd05de31c1b40e11734a"),
    "BMCLab": ("BMCLab_canonical.pkl", 127164814,
               "d3af967a35b523682d7b5039dc5c11344e56416652e112753b92de142288ed80",
               "bbce5fcd6b0bf6e67d9fa8b1cfb646d3"),
    "PD-GaM": ("PD-GaM_canonical.pkl", 84704001,
               "c85b6363443e5a202206dd669f14f0152bb91ec92e11c54b6ac951f7fd862402",
               "9a70b546db390839298a2e0ebea21a75"),
    "T-SDU-PD": ("T-SDU-PD_canonical.pkl", 26996976,
                 "ff5b30d82d3496909d4c6ef2769861a4c18889b8ea260a0ab2795f8efc3673c9",
                 "b52bea45592ef6cdbeeb00432c054b49"),
}
COHORTS = tuple(FILES)
FIRST_COHORT = "3DGait"
TOTAL_BYTES = 246926780

# Expected aggregates of the pinned files, checked before any model is fitted. See docs/protocol.md.
EXPECTED_SUBJECTS = {"3DGait": 43, "PD-GaM": 30, "BMCLab": 23, "T-SDU-PD": 14}
EXPECTED_CLASS_WALKS = {
    "3DGait": (24, 42, 14, 10),
    "PD-GaM": (783, 635, 248, 34),
    "BMCLab": (341, 276, 164, 0),
    "T-SDU-PD": (96, 118, 167, 0),
}
TOTAL_WALKS_RANGE = (2950, 2953)
CLASS_CELL_TOLERANCE = 1
WALK_KEYS = {"pose", "trans", "beta", "fps", "UPDRS_GAIT", "medication", "other"}
POSE_COLS = 72
TRANS_COLS = 3
MAX_LOW_SPEED_SHARE = 0.10
LOW_SPEED = 0.01


class SourceError(Exception):
    """Raised with a fixed, non-identifying reason code."""


def hash_file(path):
    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
            n += len(chunk)
    return n, sha.hexdigest(), md5.hexdigest()


def verify_file(path, cohort):
    _, size, sha, md5 = FILES[cohort]
    got_size, got_sha, got_md5 = hash_file(path)
    if got_size != size:
        raise SourceError(f"SIZE_MISMATCH:{cohort}")
    if got_sha != sha:
        raise SourceError(f"SHA256_MISMATCH:{cohort}")
    if got_md5 != md5:
        raise SourceError(f"MD5_MISMATCH:{cohort}")
    return got_size, got_sha, got_md5


# ---- fail-closed pickle handling ------------------------------------------
ALLOWED_DTYPES = {"f2", "f4", "f8", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "b1"}
ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
}
# Opcodes that the documented format does not need and that widen the attack surface.
REFUSED_OPCODES = {"PERSID", "BINPERSID", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX",
                   "EXT1", "EXT2", "EXT4", "NEXT_BUFFER", "READONLY_BUFFER"}
STRING_PUSH = {"UNICODE", "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8"}
MEMO_PUT = {"PUT", "BINPUT", "LONG_BINPUT"}
MEMO_GET = {"GET", "BINGET", "LONG_BINGET"}


def scan_opcodes(data):
    """Check opcodes without building anything. Returns (protocol, globals seen).

    STACK_GLOBAL is resolved from the two string pushes before it; anything else
    there is refused. String values are held only to resolve globals and are never
    returned unless they name an allowed global.
    """
    recent, memo, seen, proto = [], {}, set(), None
    try:
        for op, arg, _pos in pickletools.genops(data):
            name = op.name
            if name in REFUSED_OPCODES:
                raise SourceError("PICKLE_OPCODE_REFUSED")
            if name == "PROTO":
                proto = arg
            elif name == "FRAME":
                continue
            elif name in STRING_PUSH:
                recent = (recent + [arg])[-2:]
            elif name == "MEMOIZE":
                memo[len(memo)] = recent[-1] if recent else None
            elif name in MEMO_PUT:
                memo[int(arg)] = recent[-1] if recent else None
            elif name in MEMO_GET:
                recent = (recent + [memo.get(int(arg))])[-2:]
            elif name == "GLOBAL":
                module, _, attr = arg.partition(" ")
                if (module, attr) not in ALLOWED_GLOBALS:
                    raise SourceError(f"PICKLE_GLOBAL_REFUSED:{module}.{attr}"[:100])
                seen.add((module, attr))
                recent = (recent + [None])[-2:]
            elif name == "STACK_GLOBAL":
                pair = tuple(recent[-2:])
                if len(pair) != 2 or pair not in ALLOWED_GLOBALS:
                    raise SourceError(f"PICKLE_GLOBAL_REFUSED:{pair}"[:100])
                seen.add(pair)
                recent = (recent + [None])[-2:]
            else:
                recent = (recent + [None])[-2:]
    except SourceError:
        raise
    except Exception:
        raise SourceError("PICKLE_OPCODE_SCAN_FAILED") from None
    return proto, sorted(seen)


def _safe_reconstruct(subtype, shape, dtype_like):
    if subtype is not np.ndarray or shape != (0,) or dtype_like != b"b":
        raise SourceError("PICKLE_RECONSTRUCT_ARGS_REFUSED")
    return np.ndarray.__new__(np.ndarray, (0,), np.dtype("b"))


def _safe_dtype(spec, align=False, copy=True):
    if type(spec) is not str or spec not in ALLOWED_DTYPES:
        raise SourceError("PICKLE_DTYPE_REFUSED")
    return np.dtype(spec)


def _safe_scalar(dtype, raw):
    if not isinstance(dtype, np.dtype) or dtype.kind not in "fiub":
        raise SourceError("PICKLE_SCALAR_REFUSED")
    if type(raw) is not bytes or len(raw) != dtype.itemsize:
        raise SourceError("PICKLE_SCALAR_REFUSED")
    return np.frombuffer(raw, dtype=dtype, count=1)[0]


def _safe_frombuffer(buf, dtype, shape, order):
    if not isinstance(dtype, np.dtype) or dtype.kind not in "fiub" or order not in ("C", "F"):
        raise SourceError("PICKLE_FROMBUFFER_REFUSED")
    if type(buf) not in (bytes, bytearray) or type(shape) is not tuple:
        raise SourceError("PICKLE_FROMBUFFER_REFUSED")
    return np.frombuffer(buf, dtype=dtype).reshape(shape, order=order).copy()


_RESOLVED = {
    "_reconstruct": _safe_reconstruct,
    "_frombuffer": _safe_frombuffer,
    "scalar": _safe_scalar,
    "dtype": _safe_dtype,
    "ndarray": np.ndarray,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED_GLOBALS:
            raise SourceError("PICKLE_GLOBAL_REFUSED")
        return _RESOLVED[name]

    def persistent_load(self, pid):
        raise SourceError("PICKLE_PERSISTENT_ID_REFUSED")


def safe_loads(data):
    """Scan, then load through the restricted reader. Any surprise fails closed."""
    scan_opcodes(data)
    try:
        return RestrictedUnpickler(io.BytesIO(data)).load()
    except SourceError:
        raise
    except Exception:
        raise SourceError("PICKLE_RESTRICTED_LOAD_FAILED") from None


def load_cohort(cohort, raw_dir):
    """Hash-check, then safely load one cohort's pickle. Returns the nested dict."""
    path = Path(raw_dir) / FILES[cohort][0]
    if not path.is_file():
        raise SourceError(f"FILE_MISSING:{cohort}")
    verify_file(path, cohort)
    return safe_loads(path.read_bytes())


# ---- schema reading ---------------------------------------------------------
def _as_number(x):
    """Return a python number for an int/float or numpy scalar, else None."""
    if isinstance(x, bool):
        return None
    if type(x) in (int, float):
        return x
    if isinstance(x, np.generic) and x.dtype.kind in "fiu":
        return x.item()
    if isinstance(x, np.ndarray) and x.ndim == 0 and x.dtype.kind in "fiu":
        return x.item()
    return None


def read_label(x):
    """Returns an int in 0..3, or None for an absent label. Anything else raises."""
    if x is None:
        return None
    v = _as_number(x)
    if v is None or v != v or int(v) != v or not 0 <= int(v) <= 3:
        raise SourceError("LABEL_OUT_OF_RANGE")
    return int(v)


def read_fps(x):
    v = _as_number(x)
    if v is None or not 1 <= v <= 1000 or v != v:
        raise SourceError("FPS_INVALID")
    return float(v)


def walks_of(cohort_obj):
    """Yield (subject_key, walk_dict), checking only container structure."""
    if type(cohort_obj) is not dict or not cohort_obj:
        raise SourceError("SCHEMA_TOP_LEVEL_NOT_DICT")
    for subject, walks in cohort_obj.items():
        if type(subject) not in (str, int) or type(walks) is not dict or not walks:
            raise SourceError("SCHEMA_SUBJECT_ENTRY")
        for walk_id, rec in walks.items():
            if type(walk_id) is not str or type(rec) is not dict:
                raise SourceError("SCHEMA_WALK_ENTRY")
            if not WALK_KEYS.issubset(rec):
                raise SourceError("SCHEMA_KEYS_ABSENT")
            yield subject, rec


def read_arrays(rec):
    """Validated float64 pose (T,72) and trans (T,3). Non-finite values are kept for counting."""
    pose, trans = rec["pose"], rec["trans"]
    if type(pose) is not np.ndarray or type(trans) is not np.ndarray:
        raise SourceError("SCHEMA_ARRAY_TYPE")
    if pose.dtype.kind != "f" or trans.dtype.kind != "f":
        raise SourceError("SCHEMA_ARRAY_DTYPE")
    if pose.ndim != 2 or pose.shape[1] != POSE_COLS:
        raise SourceError("SCHEMA_POSE_SHAPE")
    if trans.ndim != 2 or trans.shape[1] != TRANS_COLS:
        raise SourceError("SCHEMA_TRANS_SHAPE")
    if pose.shape[0] != trans.shape[0] or pose.shape[0] < 2:
        raise SourceError("SCHEMA_FRAME_COUNT")
    return pose.astype(np.float64), trans.astype(np.float64)
