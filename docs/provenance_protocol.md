# Data provenance and frozen-encoder probe analyses

These commands port four further analysis families of the original study: record-level comparisons of the
publishers' original annotated files with the canonical release, a speed-source sensitivity of the P1 contrast, a
label-free pre-check of the benchmark's encoder inputs, and a probe of the benchmark's frozen MoMask encoder against
a speed scalar. All four reuse cohorts that earlier stages had already examined, so every result is exploratory.
None is a clinical method, an independent confirmation or a basis for deployment. No results are included.

All commands run through `scripts/care_pd/run_provenance.py` with explicit `--data-dir` and `--out-dir` folders
outside this checkout. Each writes one aggregate JSON file and never overwrites it.

## Commands and order

| Command | Computation | Needs | Protocol class | Output in `--out-dir` |
|---|---|---|---|---|
| `fetch-encoder-inputs` | Download and verify the v1.3 HumanML3D inputs, the MoMask checkpoint and 16 normalization files | `--accept-care-pd-terms` | download | `fetch_encoder_inputs.json` |
| `encoder-precheck` | Label-free comparability of those inputs with the canonical files; checkpoint and normalization-file checks | fetched inputs, canonical files | metadata, no model | `encoder_precheck.json` |
| `encoder-probe` | Frozen-encoder probe against input speed under leave-one-cohort-out | fetched inputs, canonical files, P0, P1 | scored, source-only | `encoder_probe.json` |
| `fetch-originals --first` / `--rest` | Download, verify and strictly decode the original annotated files (3DGait first; then BMCLab, T-SDU-PD, PD-GaM) | `--accept-care-pd-terms`; `--rest` needs a passed first download and a first parity aggregate | download | `fetch_originals_first.json`, `fetch_originals_rest.json` |
| `root-parity --first` / `--rest` | Record-level label, field and trajectory comparison of original, canonical and HumanML3D records | that stage's download aggregate, canonical files, HumanML3D inputs | metadata, no model | `root_parity_first.json`, `root_parity_rest.json` |
| `root-speed` | P1 contrast refitted with speed measured on the original root trajectories | both download aggregates, canonical files, P0, P1 | scored, source-only, post hoc | `root_speed.json` |

Run P0 and P1 first (`run_p0.py`, `run_p1.py`) in the same `--out-dir`, or pass `--p0-result` and `--p1-result`.

```bash
python -I -B scripts/care_pd/run_provenance.py fetch-encoder-inputs --data-dir DATA --out-dir OUT --accept-care-pd-terms
python -I -B scripts/care_pd/run_provenance.py encoder-precheck --data-dir DATA --out-dir OUT
python -I -B scripts/care_pd/run_provenance.py encoder-probe --data-dir DATA --out-dir OUT
python -I -B scripts/care_pd/run_provenance.py fetch-originals --first --data-dir DATA --out-dir OUT --accept-care-pd-terms
python -I -B scripts/care_pd/run_provenance.py root-parity --first --data-dir DATA --out-dir OUT
python -I -B scripts/care_pd/run_provenance.py fetch-originals --rest --data-dir DATA --out-dir OUT --accept-care-pd-terms
python -I -B scripts/care_pd/run_provenance.py root-parity --rest --data-dir DATA --out-dir OUT
python -I -B scripts/care_pd/run_provenance.py root-speed --data-dir DATA --out-dir OUT
```

**Exit codes:** 0 complete; 2 usage error or terms not accepted; 10 a result already exists; 20 an input,
environment, gate, parity or privacy stop. Where the original analysis wrote an aggregate for a failed gate (a failed
download, a failed pre-check, a blocked sensitivity run), the command writes it too and exits with 20. A stopped run
needs a new `--out-dir`. Verified local input files are reused without a new download.

## Inputs and how to obtain them

The terms, licence conflict and citation duties described in the README apply to every file below. The fetch
commands do nothing without `--accept-care-pd-terms`. That flag records your own decision that your use is permitted;
it grants no rights. Downloads go only to the pinned HTTPS hosts, enforced by an audit-hook allowlist, and every file
is kept only if its pinned size and digest match. All other commands block the network before reading data.

| Input | Source | Pins | Location under `--data-dir` |
|---|---|---|---|
| Canonical v4.0 files | Same four `*_canonical.pkl` files as P0/P1 (README) | size, SHA-256, MD5 in `common.py` | top level |
| Original annotated SMPL files | Borealis Dataverse doi:10.5683/SP3/TWIKMK, dataset version 4.0, datafiles 926879 (3DGait), 926877 (BMCLab), 926878 (T-SDU-PD), 1160521 (PD-GaM) | size and publisher MD5 in `root_core.ROOT_PINS`; SHA-256 in `root_speed_sensitivity.ROOT_SHA256` | `original/<cohort>/<cohort>.pkl` |
| HumanML3D-format inputs | Same Dataverse record, version 1.3, datafiles 926934, 926933, 926929, 926928 | size and MD5 in `encoder_inputs.DV_FILES` | `humanml3d_v1.3/<cohort>/HumanML3D_collected.npz` |
| MoMask RVQ-VAE checkpoint and 16 normalization files | github.com/TaatiTeam/CARE-PD at commit `9bc22b0258b15369cd8b62f2de20c70b3754349f` | size and git blob SHA-1 in `encoder_inputs.GH_FILES` | `care_pd_9bc22b0/<repository path>` |

Before downloading an original file, `fetch-originals` re-reads the publisher record and stops if the dataset is no
longer at released version 4.0, if a guestbook, terms of access or access request appear, or if the file record
(identifier, name, size, MD5, restriction, tabular flag) changed. The original PD-GaM file of dataset versions
1.0-2.0 (datafile 926880) is refused. Each downloaded original is decoded twice: through the existing restricted
reader of `common.py` and through a stricter reader (`root_core.strict_loads`) that validates every REDUCE, BUILD
and container operation. Both trees must agree. A digest match proves integrity, not harmlessness. The
fetch aggregate reports opcode names and an order-of-magnitude band, not exact opcode counts, because exact counts
can reveal record numbers.

## What each family estimates

### Record-level comparisons (`root-parity`): data provenance, no model fitted

Records are joined by identity on (participant key as text, walk key), never by position, label or value. Records
whose participant keys collide as text are set aside and counted.

- **Labels and fields.** Each joined record's `UPDRS_GAIT` value is classified on both sides (grade 0-3, absent as
  None, NaN or missing key, invalid) and compared. Grade 0 is a grade, not a missing value. `LABEL_PARITY_FAIL`
  if any grade changed or appears on one side only; `LABEL_PARITY_PARTIAL` for invalid values, one-sided graded
  records, identity collisions or no graded pair; otherwise `LABEL_PARITY_PASS`. Absent-encoding differences are
  reported, not counted as mismatches. Medication and `other` fields are compared per record. The subset scored by
  `encoder-probe` (canonical grade 0-2, joined to a usable input) is checked separately.
- **Trajectories.** Label fields are never passed to these functions. Frame correspondence between original and
  canonical records is found by identity of body pose (columns 3-72, tolerance 1e-4) at equal length, an offset of
  at most 30 frames or a stride of 2-6; equal-length walks without a pose match are compared frame by frame and
  flagged. A walk matches a product when its per-step root displacement profile
  correlates at least 0.99 with the original's and the median absolute log deviation of the per-step ratio from its
  walk median is at most 0.01 (one scale per walk). The HumanML3D comparison uses the official frame rule (stride
  1, 5, 1, 1 for 3DGait, BMCLab, T-SDU-PD, PD-GaM) and is planar for T-SDU-PD, whose official conversion discards
  the vertical path. Per cohort: `MATCH` if at least 95% of at least 5 scored walks match, `NO_MATCH` at 50% or
  below, else `AMBIGUOUS`; `INSUFFICIENT` with fewer than 5 scored walks. Pre-specified mechanism probes (time shift; orientation-coupled translation, 3-D or
  planar) are attributed only when at least 95% of non-matching walks share one. The PD-GaM HumanML3D inputs derive
  from an earlier original file that this command does not fetch, so that comparison is descriptive only.
- **Disclosure.** Within each cohort, if any clinical cell holds 1-4 records, every non-zero clinical and
  trajectory count becomes `LT5` or `GE5`. Per-walk statistics appear only as quartiles over at least 5 walks.
- **Not available here:** the original analysis also compared two rank statistics with a separately archived
  aggregate of another family. That aggregate is not distributed, so the field reports
  `NOT_AVAILABLE_IN_PORTABLE_RELEASE`.

### Encoder pre-check (`encoder-precheck`): label-free

No label is read. Per cohort it reports the join between HumanML3D keys and canonical walks, coverage of walks and
participants, clean-array share, frame-count relations and the rank agreement between speed from the input's root
channels and the P0 speed proxy. Pass thresholds: walk coverage at least 0.90, participant coverage at least 0.95,
clean arrays at least 0.99, rank agreement at least 0.70. It also loads the checkpoint strictly (weights only) and
requires 16 finite normalization files with positive standard deviations. The probe does not depend on this
result. In the original analysis the probe design was amended after this pre-check failed one criterion.

### Speed-source sensitivity (`root-speed`): source-only, post hoc

The question is whether the P1 pose-plus-speed versus speed-only contrast changes when speed is measured on the
publisher's original root trajectories instead of canonical `trans`, with canonical pose held fixed. Four arms use
identical eligible walks, labels, participants, folds, training weights, the six frozen candidates and shared
bootstrap draws:

| Arm | Features |
|---|---|
| A | canonical speed (P1 arm S) |
| B | the same speed function, frames, frame rate and window on the original root `trans` |
| C | canonical pose statistics plus canonical speed (P1 arm PS) |
| D | the same canonical pose columns as C, with only the speed column replaced by B |

D is a controlled substitution, not a homogeneous original-file representation. Gates run in order and fail
closed: library versions and the user's P1 aggregate (pinned to its SHA-256 at start); the user's download
aggregates; P1 pre-score checks; record-level equality of labels, medication, `other`, frame rate, frame count,
body pose, row order and duration; finiteness, P0 speed stops, a horizontal-plane check (median x-z share of the
smoothed 3-D path at least 0.90, at most 5% of walks below), a unit band (median original/canonical speed ratio
within 0.8-1.25) and rank agreement of at least 0.5; a substitution check; then **baseline parity**: A and C are
refitted and must reproduce every compared value of your own P1 aggregate within 1e-5 after six-decimal rounding.
Only then are B and D fitted. Primary estimand: difference-in-differences of macro-F1, (M(D) - M(B)) - (M(C) - M(A)),
equal-cohort mean, participant-cluster bootstrap within held-out cohorts (2,000 draws, seed 20261030). The frozen
decision rule uses margin 0.05. A secondary policy refits B and D with the candidate index that A and C selected.
The comparison is post hoc: it was defined after the P1 contrast and the record-level comparisons had been seen.

### Frozen-encoder probe (`encoder-probe`): source-only, with two disclosed exceptions

The benchmark's MoMask RVQ-VAE is assembled from the vendored official modules, loaded strictly and run in
evaluation mode. Each walk is encoded at its own length (frames beyond a multiple of four dropped; at least 8
frames), and the 512-dimensional codes are averaged over time. The benchmark instead cuts fixed-length clips and
votes over them; this difference is deliberate. Arms: `S_in` (speed from the encoder's own input channels),
`S_v4` (P0 speed proxy from canonical `trans`), `E` (embedding), `E_noroot` and `E_rootonly` (the four root channels,
or all other channels, set to the normalization mean before encoding), and `S_in_plus_E`. All arms use the six
frozen P1 candidates with inner leave-one-source-cohort-out selection, policy `all`, walk class weights and
2,000 participant-cluster bootstrap draws. Grade 3 is excluded.

- **Normalization.** In the fold that holds out cohort X, every walk, source and held-out alike, is normalized
  with the `LODO_X` statistics. The dataset providers computed those from the other labelled cohorts plus a fifth,
  unlabelled cohort (KIEL), never from X. Selection and fitting therefore never use statistics of the held-out
  cohort.
- **Exception 1, target-supervised reference.** `within_cohort_reference_fixed_LR` uses participant-grouped
  five-fold validation inside each cohort, with that cohort's own normalization statistics and the fixed logistic
  candidate. It shows whether the embedding carries any label information at all; it is not a transfer estimate.
- **Exception 2, checkpoint origin.** The training data of the shipped checkpoint are undocumented. If it was
  trained or fine-tuned on CARE-PD motion, the encoder may have seen unlabelled motion of the held-out cohort.
- **Decision rules.** For a paired difference in M with interval [lo, hi]: `BEYOND` if the difference is at least
  0.05, lo > 0 and at least three cohorts are non-negative; `BELOW` if hi < 0; otherwise `NO_DETECTED_DIFFERENCE`.
  Applied to `E - S_in` (primary, sets the status), `E - S_v4` (the contrast planned before the pre-check) and
  `S_in_plus_E - S_in`. The root-channel rule on `E - E_noroot` reports `ROOT_CHANNEL_DEPENDENT`,
  `NOT_ROOT_CHANNEL_DEPENDENT` or `INCONCLUSIVE`.
- **Fixed protocol constant.** The three-cohort summary averages BMCLab, PD-GaM and T-SDU-PD, the cohorts that
  passed the original pre-check. That set is part of the frozen design and is not recomputed from your pre-check.
- **Speed-arm parity.** `speed_arm_parity` compares `S_v4` with the speed-arm M of your own P1 aggregate (the field
  keeps its original name, `archived_S_M`) within 1.1e-6, and reports whether the eligible set has the expected
  2,908 walks and 107 participants of the pinned files.

## Output format

Every aggregate passes the frozen renderer: no identifier-like or row-level keys, counts of 1-4 shown as `LT5`,
bounded sizes and finite numbers only. The renderer requires the top-level keys `card` and `card_sha256`. Here
`card` names the command (for example `root-parity-first`) and `card_sha256` is the SHA-256 of the kernel module
that ran. `run_commit` is the `RUN_COMMIT_SHA` environment variable when set. Status strings and keys inside the
frozen kernels keep their original spelling, for example `C2_PRECHECK_PASS`, `C3X_ENCODER_<verdict>_INPUT_SPEED`,
`c3x_subset`, `gate_c2` and `H_EXPLORATORY_SENSITIVITY_COMPLETE`. A kernel stop prints a line starting with
`INDEP_STOP:`, and a refused network event raises `INDEP_NETWORK_REFUSED`.

## Port and verification limits

- **Port.** Retained kernel functions and classes are AST-identical to the frozen source after docstring removal.
  Private route checks, private paths, private aggregate hashes and historical outcome values are omitted or
  replaced by same-named portable functions and attributes. Every omission, replacement and modified line is listed
  in [q2_port_exceptions.tsv](q2_port_exceptions.tsv). Normalized AST hashes of the unchanged symbols are appended to
  [kernel_contract.tsv](kernel_contract.tsv).
- **Not verified on real data.** None of these commands has been run on the real files in this portable layout.
  Real-data equality with the original analyses is **not verified**.
- **Tests.** Fabricated-data suites run on CPU with stubbed downloads, fabricated pickles, fabricated HumanML3D
  arrays, fabricated normalization files and a randomly initialised encoder. No released checkpoint or data file
  is used. Several tests reduce learners or bootstrap draws for speed and say so.
- **Environment.** Exact agreement needs Python 3.11, NumPy 1.26.4, SciPy 1.16.0, scikit-learn 1.6.1, torch
  2.9.1, einops 0.8.1 and one BLAS/OpenMP thread. The encoder runs on CPU in float32.
- **Pickles.** Original and canonical files are pickles. The restricted readers reduce the risk but do not make a
  hostile pickle safe. Use the pinned files and keep `python -I`.
