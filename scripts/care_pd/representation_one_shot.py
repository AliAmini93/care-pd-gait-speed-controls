"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy
import sklearn
import torch
import common as C
import p0_core as P
import p1_core as Q
import representation_core as RC
import representation_native as RN
import representation_public_encoder as PE
TORCH_VERSION = '2.9.1+cu128'
VRAM_CAP_GIB = 4.0
TRAIN_BUDGET_S = 16200
LIMITS = ['F2_PARTIAL: the published-encoder arms P and S+P were not run (P BLOCKED at F0); this is not the full plan.', 'Four previously inspected CARE-PD cohorts; source-only LOCO; exploratory, not independent-site evidence.', "Only each site's reported 0/1/2 label is predicted (A0: no common clinical construct identified).", 'N is newly supervised on three source cohorts per fold; S+N stacks out-of-site N log-probabilities with S in the unchanged P1 downstream family; capacity differs from S (about 179k trained N parameters per model).', 'Rotation-only N may still encode cadence or speed through joint kinematics; see the label-free speed rank correlation diagnostic.', 'Bootstrap intervals resample persons within the four fixed sites with predictions fixed; they cover neither model refits nor new-site variation.', 'Joints 22/23 (SMPL hands) are included as the card requires; they were not fitted to markers in MoCap cohorts.']

def build_inputs(objs):
    t = Q.build_table(objs)
    roots, locs, durations = ([], [], [])
    for name in C.COHORTS:
        for _, rec in C.walks_of(objs[name]):
            pose, trans = C.read_arrays(rec)
            if not (np.isfinite(pose).all() and np.isfinite(trans).all()):
                continue
            fps = C.read_fps(rec['fps'])
            r, l6 = RC.walk_features(pose, fps)
            roots.append(r)
            locs.append(l6)
            durations.append(pose.shape[0] / fps)
    if len(roots) != t['label'].size or not np.array_equal(np.asarray(durations), t['duration']):
        raise C.SourceError('REP_ROW_ALIGNMENT')
    return (t, RC.WalkStore(roots, locs))

def fps_summary(objs):
    out = {}
    for name in C.COHORTS:
        out[name] = sorted({float(C.read_fps(rec['fps'])) for _, rec in C.walks_of(objs[name])})
    return out
