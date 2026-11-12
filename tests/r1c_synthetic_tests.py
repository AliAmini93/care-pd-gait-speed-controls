"""Fabricated-array D1/D0 tests. No CARE-PD file read. Compatible Python 3.11."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'care_pd'))
sys.path.insert(1, str(Path(__file__).resolve().parent))
import gait_site_r1c as R

def check(label, condition):
    if not condition:
        raise AssertionError(label)
    print('PASS ' + label)

def main():
    y = np.array([0, 0, 1, 1, 2, 2])
    q = np.array([0.1, 0.3, 0.55, 0.75, 1.2, 1.85])
    a, b, score = R.best_thresholds([y], [q])
    check('oracle finds separable ordered grades', score == 1.0)
    check('ordered thresholds', a <= b)
    check('oracle predictions equal known labels', np.array_equal(R.threshold_pred(q, a, b), y))
    check('degenerate constant q is handled', np.isfinite(R.best_thresholds([y], [np.zeros_like(q)])[2]))
    ys = [y, y, y]
    qs = [q, q + 0.02, q - 0.02]
    a, b, score = R.best_thresholds(ys, qs, max_grid=65)
    check('three-source mean score', 0.0 <= score <= 1.0)
    check('source mean threshold ordering', a <= b)
    proba = np.array([[0.8, 0.15, 0.05], [0.05, 0.9, 0.05], [0.05, 0.15, 0.8]])
    check('q uses class probabilities', np.allclose(R.ordinal_q(proba), [0.25, 1.0, 1.75]))
    check('AUC perfect when grading ordered', R.rank_metrics([0, 1, 2], [0.1, 0.9, 1.9])['auc_y_ge_1'] == 1.0)
    check('AUC missing class safe', R.rank_metrics([0, 0], [0.1, 0.2])['auc_y_ge_2'] is None)
    cohort = np.repeat(np.array([0, 1]), 6)
    subject = np.repeat(np.arange(6), 2)
    pose = np.column_stack([subject, np.arange(12), np.arange(12) ** 2]).astype(float)
    alt = R.permute_pose_cross_person(pose, cohort, subject, seed=20261103)
    check('pose output shape preserved', alt.shape == pose.shape)
    check('each pseudo-pose stays in same cohort', all((int(i / 6) == int(v / 6) for i, v in enumerate(alt[:, 1]))))
    check('recipient never borrows its own subject', np.all(alt[:, 0] != pose[:, 0]))
    check('pose deterministic', np.array_equal(alt, R.permute_pose_cross_person(pose, cohort, subject, seed=20261103)))
    check('permutation is label-free', 'label' not in R.permute_pose_cross_person.__code__.co_varnames)
    check('serialized bool preserved', isinstance(R.clean(True), bool))
    check('non-finite aggregate rendered as null', R.clean(np.nan) is None)
    print('R1C_SYNTHETIC_TESTS_PASS')
if __name__ == '__main__':
    sys.exit(main())
