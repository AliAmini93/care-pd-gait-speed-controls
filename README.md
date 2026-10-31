# CARE-PD gait speed controls

Code for measuring what hand-built pose features add to a one-number walking-speed baseline when gait-severity labels are predicted across CARE-PD cohorts.

It compares a one-number speed proxy with pose and kinematic summaries under leave-one-cohort-out evaluation. Both arms get the same model selection, which uses source cohorts only, and uncertainty comes from a paired bootstrap over people.

**This repository holds code only.** It contains no data, results, trained models or figures. This code does not support diagnosis, prognosis or clinical deployment, and it does not establish speed as a biomarker.

## What it does

| Stage | Question | Model | Selection |
|---|---|---|---|
| P0 | Does speed alone beat a uniform guesser? Does adding 138 pose statistics improve on speed? | One fixed logistic regression | None |
| P1 | Does the second answer change with stronger learners? | Six logistic, random-forest and gradient-boosting candidates | Inner leave-one-cohort-out over source cohorts only, per arm and per fold |

P1 came after P0's results had been seen, and both stages use identical participants. Treat P1 as a sensitivity analysis of P0; it does not replicate P0 independently. Decision rules were fixed internally before scoring and were not publicly registered. [docs/protocol.md](docs/protocol.md) gives the full method.

## Data

No CARE-PD file is included or licensed here. Obtain the data yourself from its official sources and read its terms first:

- Terms of use: <https://neurips2025.care-pd.ca/terms-of-use>
- Hugging Face: <https://huggingface.co/datasets/vida-adl/CARE-PD>
- Borealis Dataverse: <https://doi.org/10.5683/SP3/TWIKMK>

**The licence statements differ.** When checked on 2026-10-31, the Terms of Use text said CC BY-NC 4.0, while the Hugging Face licence field and the Dataverse licence said CC BY-NC-ND 4.0. Whether your use is permitted is your decision, and nothing here gives legal advice or grants rights. The terms also forbid attempts to identify or contact participants, and they require citation of the database and of the original paper for each cohort you use.

The analysis reads these four files from `Canonicalized_SMPL_pickles/` at Hugging Face commit `d53fe929aaacbd6822bcad5ef2b1d215401800b7`. Each file's size, SHA-256 and MD5 are pinned in [scripts/care_pd/common.py](scripts/care_pd/common.py), and any mismatch stops the run.

`3DGait_canonical.pkl`, `BMCLab_canonical.pkl`, `PD-GaM_canonical.pkl`, `T-SDU-PD_canonical.pkl`

If you are entitled to use the data, the optional downloader fetches those exact files. It does nothing without an explicit flag:

```bash
python -I scripts/care_pd/fetch_canonical.py --first --data-dir /path/to/care_pd --accept-care-pd-terms
python -I scripts/care_pd/schema_check.py --first --data-dir /path/to/care_pd
python -I scripts/care_pd/fetch_canonical.py --rest --data-dir /path/to/care_pd --accept-care-pd-terms
```

Keep data and results outside version control. `.gitignore` is a backstop, not a guarantee.

## Quick start: tests only, no data needed

Use any Python 3.11 environment. Run from the repository root.

```bash
pip install -r requirements.txt

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -I tests/p0_smoke_tests.py
python -I tests/p1_smoke_tests.py
python -I tests/cli_smoke_tests.py
```

Each script ends with its own `*_PASS` line. Tests use fabricated data only and do not download CARE-PD files.

## Running on the real files

P1 needs a completed P0 result in the same `--out-dir`.

```bash
python -I scripts/care_pd/schema_check.py --all --data-dir /path/to/care_pd
python -I scripts/care_pd/run_p0.py --data-dir /path/to/care_pd --out-dir /path/to/results
python -I scripts/care_pd/run_p1.py --data-dir /path/to/care_pd --out-dir /path/to/results
```

| File in `--out-dir` | Contents |
|---|---|
| `p0.json` | Aggregate P0 metrics, intervals, verdict and provenance |
| `p1_strong_models.json` | Aggregate P1 metrics, intervals, verdict and provenance |
| `p1_attempts.local.tsv` | Attempt record: one retry after a pre-score stop, none after scoring starts |

**Exit codes:** 0 complete, 10 refused because a result or scored attempt exists, 20 a stop check failed. A completed result is never overwritten; use a new `--out-dir` for a new run.

**Stop checks run before any model is fitted.** They cover the file hashes, the restricted pickle reader, subject and class counts, non-finite values and low-speed shares. P1 also requires the exact library versions below and must reproduce P0's five mean scores through its own code path within `1e-9`.

**Cost.** The original runs used one CPU thread and an 8 GiB address-space limit, with time limits of 1 hour for P0 and 3 hours for P1. The null analyses have time budgets of 40 and 90 minutes. A slower machine may finish fewer null draws, and the output records the number completed. The entry points default to one BLAS and OpenMP thread unless you set otherwise.

### What has and has not been verified

- **Algorithm modules.** `common.py`, `p0_core.py` and `p1_core.py` differ from the code behind the original private runs only in comments, docstrings and how the data path is passed in. This was checked with a syntax-tree comparison, and with byte-identical aggregate output from both versions on fabricated data.
- **Entry points.** `run_p0.py`, `run_p1.py`, `schema_check.py` and `fetch_canonical.py` are portable rewrites of the original scripts. They have been tested on fabricated pickles and a stubbed network only. **They have not been run on the real CARE-PD files in this layout.**
- **Reproducing a number.** Expect exact agreement only with the same library versions and thread settings, and treat any difference as something to investigate.

## Repository layout

Only experiment code, the CARE-PD data-integrity gate, relevant synthetic tests and reproducibility documentation are maintained here. Manuscript figure-generation and layout utilities are excluded.


```
scripts/care_pd/common.py           pins, schema reading, fail-closed pickle reader
scripts/care_pd/p0_core.py          speed and pose features, P0 arms, metrics, bootstrap, verdict
scripts/care_pd/p1_core.py          kinematic features, candidates, source-only selection, null, verdict
scripts/care_pd/run_p0.py           P0 command-line entry point
scripts/care_pd/run_p1.py           P1 command-line entry point
scripts/care_pd/schema_check.py     pre-metric checks, aggregate output only
scripts/care_pd/fetch_canonical.py  optional download; requires an explicit flag
tests/                              synthetic tests; no real data
docs/protocol.md                    method, thresholds and verdict rules
```

## Tested environment

Python 3.11.15, NumPy 1.26.4, SciPy 1.16.0 and scikit-learn 1.6.1. `run_p1.py` stops on any other NumPy, SciPy or scikit-learn version. `run_p0.py` records its versions and does not enforce them. GitHub Actions runs the synthetic tests on Python 3.11 with the versions pinned in `requirements.txt`; it downloads no dataset.

## Limitations

- **Four cohorts.** Intervals cover sampling of people within these cohorts. They do not cover refits or unseen sites.
- **Labels.** Gait items that may underlie these labels (MDS-UPDRS 3.10, and item 29 of the 1987 UPDRS) mention speed, so some agreement between speed and label is expected by construction. The item behind each cohort's labels is not settled, and the integer labels are pooled as released.
- **Speed proxy.** Units are those of the files and are not verified as metres. The proxy divides a smoothed path length by total walk time, which introduces a bias tied to walk length. It was kept as specified.
- **Scope.** The inputs are hand-built per-walk summaries. No trained encoder or temporal model is run, so nothing here applies to learned representations.
- **Untrusted pickles.** The dataset files are pickles. They are read after an opcode scan through a restricted unpickler, which reduces the risk but does not make hostile pickles safe. Use the pinned files, keep `python -I`, and do not run scripts from inside the data folder.

## Licence and citation

The original code is released under the [MIT License](LICENSE). That licence does not cover CARE-PD or any other data.

To cite this software, use [CITATION.cff](CITATION.cff). If you use CARE-PD, its terms require citing the database (Adeli et al., *CARE-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment*, NeurIPS 2025) and the original paper for each cohort you use.

Parts of this code and documentation were written with an LLM coding agent, as the commit metadata records.
