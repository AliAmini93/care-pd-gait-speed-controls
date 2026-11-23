"""Portable analysis kernel. See docs/provenance_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import numpy as np

def integrate_root_xz(raw):
    raw = np.asarray(raw, dtype=np.float64)
    ang = np.concatenate([[0.0], np.cumsum(raw[:, 0])])[1:]
    c, s = (np.cos(2.0 * ang), np.sin(2.0 * ang))
    vx, vz = (raw[:, 1], raw[:, 2])
    step = np.column_stack([vx * c - vz * s, vz * c + vx * s])
    return np.concatenate([np.zeros((1, 2)), np.cumsum(step, axis=0)], axis=0)

def within_participant_spread(ratio, subject):
    ratio, subject = (np.asarray(ratio, float), np.asarray(subject))
    out = []
    for s in np.unique(subject):
        r = ratio[subject == s]
        r = r[np.isfinite(r) & (r > 0)]
        if r.size >= 2:
            out.append(r.max() / r.min() - 1.0)
    return (float(np.median(out)) if out else None, len(out))
