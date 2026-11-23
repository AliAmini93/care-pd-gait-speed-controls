# Synthetic test t40: why its fabricated world was changed (2026-11-24)

Test `t40_known_incremental_rotation_signal_is_detected` lives in `tests/representation_synthetic_tests.py`. It checks two things on fabricated walks:

- the rotation-only graph-temporal encoder `N` detects a planted trunk-rotation signal: four-cohort mean macro-F1 `n > 0.5`;
- adding `N` to speed helps: `(S+N) − S > 0.05`.

The thresholds, both assertions, every analysis kernel, the `null` world (negative control t39) and the original `signal` world (used by the structural fold tests) are **unchanged**. Only t40 now draws its campaign from a new world, `signal_untied`.

## Observed failure

On GitHub Actions, after the hosted runner image moved from ubuntu-24.04 `20261101.360.1` to `20261116.133.1`:
- t40 failed with `n = 0.49144129196394`, deterministically: the same value twice on the same commit.
- The torch, NumPy, SciPy and scikit-learn wheels were identical, and the test file was identical.
- The previous image had passed.

On the local CPU verification machine, the same commit gives `n = 0.5192190697417177` with 1 or 2 threads, and the same with `ATEN_CPU_CAPABILITY=default`.

## Root cause: an exact tie, not a detection failure

In the `signal` world each cohort has nine persons, one for every pair of speed class `cv` and rotation class `r` in {0, 1, 2}. The grade is a function of `u = cv + r` (0 if u ≤ 1; 1 if u = 2; 2 if u ≥ 3). `N` sees only rotations, so it can at best learn `r`. Exact enumeration with the class-balanced training weights:

| rotation class r | grades of its three persons | weighted grade mass (0 : 1 : 2) | Bayes decision |
|---|---|---|---|
| 0 | 0, 0, 1 | 2/3 : 1/3 : 0 | 0 |
| 1 | 0, 1, 2 | **1/3 : 1/3 : 1/3** | **tie (0, 1 or 2)** |
| 2 | 1, 2, 2 | 0 : 1/3 : 2/3 | 2 |

So even a perfect rotation detector scores, per held-out cohort, **5/9 = 0.556** if the tie in stratum r = 1 breaks toward grade 1, and **4/9 = 0.444** otherwise. The four-cohort mean lies between 0.444 and 0.556 and straddles the 0.5 threshold. Which side it lands on is decided by floating-point tie-breaking, not by whether `N` detected the signal.

The measurements show exactly this:
- The local per-cohort values of `N` are 0.5213, 0.4444, 0.5556 and 0.5556.
- The CI value is lower by exactly **1/36 = (5/9 − 4/9)/4**: one held-out cohort's tie broke the other way on the new runner image.

The speed-only arm faces the mirror-image tie (at `cv = 1`).

## The replacement world

`signal_untied` keeps the same persons, walks, durations, rotation and speed construction, and noise. Only the grade rule changes: `u = 2r + cv` (0 if u ≤ 1; 1 if 2 ≤ u ≤ 4; 2 if u ≥ 5). Enumeration with class-balanced weights:

| view | strata and weighted grade mass | Bayes macro-F1 |
|---|---|---|
| `N` (rotation only) | r=0: 1 : 1/5 : 0 → 0; r=1: 0 : 3/5 : 0 → 1; r=2: 0 : 1/5 : 1 → 2 | **0.783, no tie** |
| `S` (speed only) | one tie, at cv = 1 between grades 0 and 2; both choices give the same score | 0.300 |
| `S+N` | every (cv, r) cell has a single grade | 1.000 |

Rotation still carries the signal and speed is still incremental. Now, though, a detecting `N` (about 0.78) and a non-detecting `N` are separated by the unchanged 0.5 threshold. A non-detecting `N` scores at most about 0.33 if it guesses, and 0.24 if it predicts a constant.

## Verification (fabricated data, CPU)

| run | world | `n` | `(S+N) − S` |
|---|---|---|---|
| test seed 5, 2 threads | signal_untied | 0.7833 (all four cohorts) | 0.519 |
| test seed 5, 1 thread | signal_untied | 0.7833 (identical) | 0.519 |
| test seed 5, `ATEN_CPU_CAPABILITY=default` | signal_untied | 0.7833 (identical) | 0.519 |
| seed 6 | signal_untied | 0.7021 | 0.466 |
| seed 7 | signal_untied | 0.7833 | 0.430 |
| seed 8 | signal_untied | 0.7081 | 0.385 |
| seed 5, negative control | null (rotation unrelated to grade) | 0.1667 | −0.044 |

**Fixture identity.** The `null` and `signal` worlds produced by the new test file are array-identical to those of the previous file for seeds 5, 21 and 41.

**Limits.** This analysis is about a fabricated test fixture. It says nothing about clinical data. It does not prove that CPU arithmetic is identical across machines; it shows only that the test no longer depends on tie-breaking.
