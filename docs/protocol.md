# Protocol

This document describes what the code computes. Where it differs from the code, the code is authoritative. It reports no results.

## Chronology

1. The P0 rules were fixed internally before any score existed.
2. P0 ran on the real files.
3. P1 was designed after P0's aggregate results were seen. Its rules were fixed internally before its own analysis.
4. P1 ran on the same people.

Neither stage was publicly registered. P1 is a robustness check on P0, not an independent replication.

## Data

Four canonicalized SMPL files from CARE-PD, pinned by commit, size, SHA-256 and MD5 in `scripts/care_pd/common.py`: the cohorts are 3DGait, BMCLab, PD-GaM and T-SDU-PD.

A walk is **eligible** if its label is 0, 1 or 2. The stop checks expect 2953 labelled walks, of which 2908 are eligible, from 107 of 110 people. Class 3 is excluded from both training and testing because two cohorts have none.

| Cohort | People with labelled walks | Walks by class 0, 1, 2, 3 |
|---|---:|---|
| 3DGait | 43 | 24, 42, 14, 10 |
| PD-GaM | 30 | 783, 635, 248, 34 |
| BMCLab | 23 | 341, 276, 164, 0 |
| T-SDU-PD | 14 | 96, 118, 167, 0 |

These are the expected counts for the P0 check, which allows a difference of up to 1 walk per cell; the pinned files hold 45 class-3 walks in total. P1 additionally requires the totals above.

## Stop checks

No score is computed if any of these fails:

- a file's size, SHA-256 or MD5 differs from its pin, or the restricted reader refuses it;
- expected keys or array shapes are absent;
- a pose or translation value is non-finite;
- subject or class counts differ from those expected, or the total falls outside 2950 to 2953;
- more than 10% of a cohort's walks have speed at or below 0.01, or a cohort's speed has zero IQR.

P1 also stops on any library version other than NumPy 1.26.4, SciPy 1.16.0 and scikit-learn 1.6.1, on non-finite or wrongly shaped P1 features, on other totals, or on a cohort missing a class. Before any new metric exists, the P0 model refit through P1's path must reproduce P0's mean score for `D`, `S`, `Sz`, `P` and `PS` within `1e-9`.

## Input arms

Notation: a walk has `T` frames at `f` frames per second.

| Arm | Definition | Use of the target cohort |
|---|---|---|
| `S` | Summed step length of the root `(x, z)` path, first smoothed by a moving average with edge padding over `max(1, round(f/2))` frames, then divided by `T/f` | Inductive |
| `S_alt` | Median frame-to-frame step of the same smoothed path over fully covered windows, times `f` | Inductive |
| `D` | Duration, `T/f` | Inductive |
| `S_D` | `S` and `D` | Inductive |
| `P` | Standard deviation and p95 minus p5 of the 69 non-root axis-angle components: 138 values | Inductive |
| `PS` | `P` and `S` | Inductive |
| `PS_alt` | `P` and `S_alt` | Inductive |
| `K` | 21 kinematic summaries, listed in `KIN_NAMES` in `p1_core.py`, with no speed, path or duration term | Inductive |
| `KS` | `K` and `S` | Inductive |
| `Sz` | `S`, median/IQR-scaled within each cohort | Transductive: uses the target's unlabelled walks |
| `PzSz` | Each `P` column scaled the same way, then `Sz` | Transductive: uses the target's unlabelled walks |

Units are those of the files and are not verified as metres. Released body-shape parameters are not used.

## Evaluation

- **Folds.** Leave one cohort out: train on three cohorts and test on the fourth.
- **Metric.** Walk-level macro-F1 over classes 0, 1 and 2. `M` averages it equally over the held-out cohorts.
- **Baselines.** `C` predicts the training majority class. `F_U` is the asymptotic expected score of uniform random guessing, given held-out class shares `p_c`: `(1/3) Σ_c 2 p_c (1/3) / (p_c + 1/3)`. It is a plug-in reference, and a trained model can fall below it.
- **Intervals.** People are resampled within each cohort, and the same draws apply to every arm: 2000 draws, seed `20261030`, same draw order in P0 and P1. Limits are the 2.5th and 97.5th percentiles. Predictions stay fixed, so intervals exclude refit variance and do not cover unseen sites.

## P0

Every fitted arm uses `StandardScaler` fitted on the training cohorts, then `LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000)`, with no tuning.

**Rules, on unrounded values:**

- **H1** (`S1` in the output): `M(S) - F_U` is at least 0.05 and its lower 95% bound is above 0.
- **H2** (`C1` in the output): `M(PS) - M(S)` is at least 0.05 and its lower 95% bound is above 0.

| H1 | H2 | Verdict |
|---|---|---|
| Pass | Fail | `SPEED_FLOOR_HOLDS` |
| Pass | Pass | `POSE_ADDS_BEYOND_SPEED` |
| Fail | Pass | `POSE_ONLY_SIGNAL` |
| Fail | Fail | `AT_FLOOR` |

The 0.05 margin is a screening heuristic, not a significance rule. `Sz` can qualify the wording of a failed H1 but cannot change the verdict.

**Descriptive only:** a null (200 draws) that shuffles walk labels inside each training cohort and so ignores clustering by person, a within-person Spearman control of speed against label, and person-averaged posteriors.

## P1

Six candidates are standard-scaled on the training walks and trained with class-balanced walk weights. Their order is the tie-break order.

| Name | Model |
|---|---|
| `LR_C1` | Logistic regression, `C=1.0`, the P0 model |
| `LR_C001` | Logistic regression, `C=0.01` |
| `RF_leaf25` | Random forest, 300 trees, `max_features="sqrt"`, minimum leaf 25 |
| `RF_leaf5` | Random forest, same, minimum leaf 5 |
| `HGB_L7_it200` | Histogram gradient boosting, 7 leaves, 200 iterations, minimum leaf 20, L2 1.0 |
| `HGB_L15_it100` | Histogram gradient boosting, 15 leaves, 100 iterations, minimum leaf 50, L2 10.0 |

**Selection.** Done separately for each input arm and outer fold:

1. Train each candidate on two source cohorts and score the third, over all three rotations.
2. Take the highest unweighted mean. Within `1e-12`, the earlier candidate wins.
3. Refit on all three source cohorts and predict the held-out cohort once.

Held-out labels and features enter no selection, scaling, weighting or stopping decision. The only exception is the transductive arms, which scale using the target's unlabelled walks.

**Primary contrast.** `Δ = M(PS) - M(S)`, both selected from all six candidates, so the speed control is not handicapped.

**Rules, on unrounded values:**

- **Screen:** `Δ` is at least 0.05, its lower 95% bound is above 0, and the per-cohort difference is non-negative on at least two cohorts.
- **Null gate, which applies only after the screen passes:** the one-sided rank value of `Δ` is at most 0.05, from at least 39 completed draws of 100. Within each cohort, each person is assigned another person as a donor by permutation, and each person's walks are relabelled by sampling with replacement from their donor's walks. Held-out labels stay real, and the selected configurations are refit without repeating selection.
- **Guard:** `M(PS)` minus `M(S)` under the fixed `LR_C1` model, with the same margin and bound. It can only qualify a positive result.

| Outcome | Verdict | Qualifier |
|---|---|---|
| Screen, null gate and guard pass | `P1_GO_BEYOND_SPEED` | None |
| Screen and null gate pass, guard fails | `P1_GO_BEYOND_SPEED` | `GAIN_DEPENDS_ON_WEAKER_CONTROL` |
| Screen passes, null rank value above 0.05 | `P1_SCREEN_PASS_NOT_ABOVE_NULL` | None |
| Screen passes, fewer than 39 null draws | `P1_SCREEN_PASS_NULL_UNAVAILABLE` | None |
| Screen fails | `P1_NO_GO_NOT_BEYOND_SPEED` | `POSE_ARM_WORSE`, `SMALL_POSITIVE` or `INCONCLUSIVE` |

## Exploratory outputs

Everything else is exploratory and unadjusted for multiplicity:

- other contrasts and per-family policies;
- per-candidate scores, which select nothing;
- person-weighted training and evaluation;
- rank correlation given speed and duration, and speed terciles;
- balanced accuracy predicting the cohort from features, with person-grouped 5-fold logistic regression and no severity label;
- inner-against-outer selection stability;
- confusion counts per cohort;
- refit sensitivity over seeds 1 to 5.

**Known weakness.** On fabricated data with a sharp speed-only signal, the rank analysis given speed still credited the pose arm with a small positive difference. Only differences from the speed arm should be read, and a positive value does not establish information beyond speed.

## Output privacy

Result files hold aggregates only. Subject keys become anonymous integers in memory, and no subject identifier, walk identifier, prediction or per-walk value is written. The tests assert this on fabricated data.
