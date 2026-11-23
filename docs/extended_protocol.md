# Portable exploratory follow-up analyses

The original P0/P1 modules and entry points are preserved. The additional kernels
implement the original follow-up algorithms; their private execution routes,
frozen-card dependencies, local aggregate filenames and machine-specific paths
have been replaced by explicit portable orchestration. No results are included.
All follow-ups reuse previously inspected cohorts and are exploratory. They are
not independent confirmation, a new clinical method or evidence for deployment.

## Coverage and ordering

Run P0 and P1 first, using `run_p0.py` and `run_p1.py`. Then use
`scripts/care_pd/run_followup.py FAMILY --data-dir DATA --out-dir OUTPUT`, where
both directories are outside this code checkout. The output is `OUTPUT/FAMILY.json`.
The command never downloads anything. Each aggregate must be generated locally
from separately obtained data; there are no bundled upstream result files.
Data-provenance and frozen-encoder probe commands use a separate entry point,
`run_provenance.py`, documented in [provenance_protocol.md](provenance_protocol.md).

| Family | Computation | Required upstream aggregates | Device |
|---|---|---|---|
| `r1c` | Noise and same-cohort cross-person pose permutation, three seeds and learner-family policies; source-estimated cutpoints; held-out-label oracle and rank diagnostics | P0, P1 | CPU |
| `r8a` | BMCLab participant × medication cells; walk, cell reweighting, cell probability/vote pooling, participant-cell weighting; paired participant bootstrap | P0, P1 | CPU |
| `r8b` | Within-person partial rank associations, speed-like and noise controls, paired medication-cell and two-walk descriptions | P0 | CPU |
| `r8c` | Nested participant-grouped within-cohort CV, matched LOCO reference, paired contrasts and BMCLab cell scoring | P0, P1 | CPU |
| `r8d` | Three frozen CNN/GRU candidates, pose/full inputs and two seeds; inner source-only selection | P0, P1 | CUDA required by CLI |
| `representation` | Rotation-only graph-temporal encoder, out-of-source stacking with speed, three-column noise control and encoder exclusion record | P0, P1, R8-D | CUDA required by CLI |
| `abcd` | Final source-gated and ungated joint-rotation/cyclic residuals, variance penalty, participant shrinkage/unshrunk control; support and observability gates | P0, P1 | CPU |
| `abcd_a` through `abcd_d` | Original sequential three-cohort development procedures and fabricated-world validity screens | P0, P1; each is a separate exploratory development run | CPU |

Defaults are `OUTPUT/p0.json`, `OUTPUT/p1_strong_models.json` and, for
`representation`, `OUTPUT/r8d.json`. Override these only with explicit
`--p0-result`, `--p1-result` and `--r8d-result` paths. Upstream status, dataset
revision, canonical input pins, bootstrap settings, candidate configurations and
P1-to-P0 hash lineage are checked before data access. Frozen P0 and selected-P1
numerical parity checks then run before follow-up scores. A missing or corrupt
upstream file, wrong library version or failed parity check stops the command.
Completed outputs are never overwritten. Exit codes are 0 (complete), 10
(existing result) and 20 (input, environment or parity stop).

```bash
python -I -B scripts/care_pd/run_followup.py r1c --data-dir /path/to/care_pd --out-dir /path/to/results
python -I -B scripts/care_pd/run_followup.py r8a --data-dir /path/to/care_pd --out-dir /path/to/results
python -I -B scripts/care_pd/run_followup.py r8b --data-dir /path/to/care_pd --out-dir /path/to/results
python -I -B scripts/care_pd/run_followup.py r8c --data-dir /path/to/care_pd --out-dir /path/to/results
python -I -B scripts/care_pd/run_followup.py r8d --data-dir /path/to/care_pd --out-dir /path/to/results
python -I -B scripts/care_pd/run_followup.py representation --data-dir /path/to/care_pd --out-dir /path/to/results
python -I -B scripts/care_pd/run_followup.py abcd --data-dir /path/to/care_pd --out-dir /path/to/results
```

The development arms remain separate from the final four-cohort analysis. Their
routines use only BMCLab, PD-GaM and T-SDU-PD. The final environmental-agreement
candidate is unscored when its original support rule fails. That rule counts
held-out grades and must not be interpreted as a source-only eligibility rule.
The participant shrinkage validation is not fully nested. Those limitations are
preserved; this release neither repairs the analyses nor retroactively validates
their development estimates.

## Frozen mathematical contract

Kernel ASTs were compared with the frozen source used by the original analyses.
The portable kernels retain their mathematical operations, configurations,
thresholds, seeds, sampling and selection rules. The translation removes legacy
`main` functions, private route checks, private path constants, private aggregate
hashes, private audit-hook installation and docstrings. It adds the explicit
release wrapper and upstream schema/lineage contract. The documentary encoder
record replaces one private-report pathname by its substantive input limitation.
The public R8-B wrapper additionally omits the historical study-outcome constant
and its dependent validation function. Instead it recomputes the frozen P0
within-person control with ten bootstrap draws and compares every deterministic
whole-study and per-cohort field, requiring exact JSON types and equality against
the user's validated, locally generated P0 output. Missing fields, cohorts,
status, undefined counts and corrupt types fail closed. Bootstrap intervals use
different draw counts: their schema, finite bounds and ordering are checked,
without asserting interval equality. The sparse all-undefined P0 schema is also
checked. No historical measured association values are bundled. This establishes
local input consistency; it provides weaker historical-run assurance than the
removed outcome pin. The scientific estimators and their eligibility gates are
unchanged. Input-schema counts (including medication-cell structure and
eligible-person counts) are retained as frozen data-integrity gates; they are
not fitted performance or association outcomes. No scientific kernel is retuned
or corrected. `kernel_contract.tsv` records
normalized AST hashes for auditing subsequent source changes.

The default draw counts remain 2,000 for P0/P1, within-cohort, sequence, graph and
final residual comparisons; 500 for diagnostics; 1,000 for BMCLab cells and
within-person analyses. Prediction-fixed participant resampling does not include
refitting uncertainty or uncertainty over unseen institutions. Noise/permutation
and oracle comparisons retain their distinct source-only, transductive and
in-sample protocol labels. No clinical grading instrument is harmonized here.

## Synthetic verification and limitations

`python -I -B tests/run_all.py` orchestrates all old and new synthetic suites,
including corruption cases. It exits unsuccessfully on a failed suite or an
unexpected skip. Tests use fabricated arrays/pickles, a stubbed downloader,
reduced training configurations where explicitly named, full frozen learner
configuration checks, and CPU learned-model exercises. No real data or external
checkpoint is needed or downloaded. The isolated archive check additionally
blocks filesystem reads from the original research and code checkouts.

The verification environment is Python 3.11.15, NumPy 1.26.4, SciPy 1.16.0,
scikit-learn 1.6.1 and torch 2.9.1+cu128. CUDA is unavailable on the verification
host. GPU kernels and complete CUDA CLI scoring are **not executed or verified**
there. CPU synthetic training is not evidence of CUDA determinism or equivalence
to original GPU results. The original graph runner used deterministic float32,
TF32 disabled, a 4 GiB VRAM cap and a 16,200-second training budget. The sequence
runner retains its two-seed design and CUDA nondeterminism caveat.

**Real-data equality of this portable layout has not been verified.** Synthetic
passes and source equivalence establish narrower properties, not reproduction of
any reported clinical result. Data must be obtained separately under the
custodians' terms. The pinned revision is
`d53fe929aaacbd6822bcad5ef2b1d215401800b7`. The documented conflict between
CC BY-NC 4.0 text and CC BY-NC-ND 4.0 metadata remains unresolved. The MIT licence
covers this code only and does not resolve rights to data or derivatives.

## Figure and documentary scope

No manuscript, figure assets, result aggregates or original figure generators
are distributed. The protocol schematic, cohort-contrast plot, diagnostic plot,
evaluation-setting plot and rated-object diagram have **no runnable public
regeneration entry point in this release**. Their original generators depend on
private evidence manifests, exact aggregate hashes and publication layout
modules; the rated-object generator additionally requires documentary crosswalks,
a citation ledger and a pinned bibliography, and deliberately refuses a changed
bibliography. They were not converted or synthetically verified here. Numerical
analysis code can produce local aggregate inputs, but that does not constitute
exact figure regeneration. Rated-object and instrument provenance rests on
source publications and documentation, not a computational harmonization proof.

The published-encoder record is inert: ACTOR and TEMOS remain blocked and no
weights are fetched. The broader input-provenance exclusions also include a
virtual-camera choice for MotionAGFormer and unestablished checkpoint/input
conventions for MotionCLIP. The only pretrained-encoder comparison is the separate
frozen MoMask probe (`run_provenance.py encoder-probe`), whose source-only scope and two
disclosed exceptions are documented in [provenance_protocol.md](provenance_protocol.md); no
other pretrained encoder is evaluated.

Historical development labels are not emitted by the portable ABCD output. Computed decisions, controls, contrasts and never-scored outputs retain their frozen definitions.
