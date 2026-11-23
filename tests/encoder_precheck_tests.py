"""Fabricated-only tests of the label-free encoder-input pre-check and the encoder-precheck command.

No clinical file, no network. The command-line test uses fabricated canonical pickles, HumanML3D collections,
normalization files and a randomly initialised checkpoint, all pinned for this process only.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "care_pd"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_precheck as K  # noqa: E402
import aggregate_core as IC  # noqa: E402

PASS = 0


def ok(name, cond):
    global PASS
    if not cond:
        print(f"FAIL {name}")
        sys.exit(1)
    PASS += 1
    print(f"PASS {name}")


class FakeNpz(dict):
    @property
    def files(self):
        return list(self.keys())


def fabricate(n_subjects=12, walks_each=6, fps=30.0, seed=0, shuffle_speed=False):
    """A fabricated cohort: a v4-like nested dict and a matching npz-like mapping with a known speed relation."""
    rng = np.random.default_rng(seed)
    cohort, npz = {}, FakeNpz()
    speeds = []
    for s in range(n_subjects):
        cohort[f"S{s:02d}"] = {}
        for w in range(walks_each):
            t = int(rng.integers(60, 120))
            v = float(rng.uniform(0.3, 1.5))
            speeds.append(v)
            z = np.arange(t) * v / fps
            trans = np.column_stack([np.zeros(t), np.full(t, 0.9), z])
            cohort[f"S{s:02d}"][f"w{w}"] = {"trans": trans, "fps": fps, "UPDRS_GAIT": "NEVER_READ"}
    order = list(range(len(speeds)))
    if shuffle_speed:
        rng.shuffle(order)
    i = 0
    for s in range(n_subjects):
        for w in range(walks_each):
            t = cohort[f"S{s:02d}"][f"w{w}"]["trans"].shape[0]
            raw = rng.normal(0, 0.01, size=(3, t - 1, K.FEAT)).astype(np.float32)
            raw[:, :, 1] = 0.0
            raw[:, :, 2] = speeds[order[i]] / fps
            npz[f"S{s:02d}__w{w}"] = raw
            i += 1
    return cohort, npz


def main():
    ok("key: plain", K.parse_key("A1__walk_3") == ("A1", "walk_3", False, False))
    ok("key: down suffix is stripped", K.parse_key("A1__walk_3_down2") == ("A1", "walk_3", False, True))
    ok("key: trimmed is flagged", K.parse_key("A1__walk_3_TRIMMED")[2] is True)
    ok("key: malformed is refused", K.parse_key("A1_walk") is None and K.parse_key("a__b__c") is None
       and K.parse_key("__x") is None and K.parse_key(5) is None)
    ok("first_view takes index 0 of the augmentation axis",
       K.first_view(np.zeros((4, 10, K.FEAT), np.float32)).shape == (10, K.FEAT))
    ok("first_view refuses wrong width, integers and single frames",
       K.first_view(np.zeros((10, 5), np.float32)) is None and K.first_view(np.zeros((10, K.FEAT), np.int32)) is None
       and K.first_view(np.zeros((1, K.FEAT), np.float32)) is None)
    raw = np.zeros((50, K.FEAT), np.float32)
    raw[:, 1], raw[:, 2] = 0.03, 0.04
    ok("speed from input channels", abs(K.speed_from_humanml(raw, 30.0) - 1.5) < 1e-6)
    ok("rank agreement: monotone is 1, reversed is -1",
       abs(K.rank_corr(np.arange(9.0), np.arange(9.0) ** 3) - 1) < 1e-12 and abs(K.rank_corr(np.arange(9.0), -np.arange(9.0)) + 1) < 1e-12)
    ok("rank agreement: undefined cases give None", K.rank_corr([1, 2], [1, 2]) is None and K.rank_corr(np.ones(9), np.arange(9.0)) is None)

    speed_fn = lambda trans, fps: float(np.linalg.norm(np.diff(trans[:, [0, 2]], axis=0), axis=1).sum() / (trans.shape[0] / fps))
    cohort, npz = fabricate()
    index = K.canonical_index(cohort)
    ok("index never touches the label field", all(len(v) == 3 for v in index.values()) and len(index) == 72)
    rep = K.cohort_report(npz, index, speed_fn)
    ok("aligned cohort: full coverage", rep["walk_coverage"] == 1.0 and rep["participant_coverage"] == 1.0)
    ok("aligned cohort: frame relation is minus one", rep["frames_npz_minus_v4_median"] == -1.0 and rep["aug_axis_max"] == 3)
    ok("aligned cohort: speed definitions agree", rep["speed_rank_agreement"] > 0.99 and K.cohort_passes(rep))
    cohort_b, npz_b = fabricate(shuffle_speed=True, seed=1)
    rep_b = K.cohort_report(npz_b, K.canonical_index(cohort_b), speed_fn)
    ok("misaligned walks are detected by the speed agreement", rep_b["speed_rank_agreement"] < 0.5 and not K.cohort_passes(rep_b))
    npz_c = FakeNpz({k: v for k, v in list(npz.items())[:40]})
    npz_c["ZZ__w0"] = np.zeros((3, 20, K.FEAT), np.float32)
    npz_c["S00__w0_down2"] = npz["S00__w0"]
    npz_c["S00__w1_trimmed"] = npz["S00__w1"]
    npz_c["broken"] = np.zeros(3, np.float32)
    rep_c = K.cohort_report(npz_c, index, speed_fn)
    ok("partial cohort: coverage below threshold fails", rep_c["walk_coverage"] < K.MIN_WALK_COVERAGE and not K.cohort_passes(rep_c))
    ok("partial cohort: skipped keys are counted with the disclosure rule",
       rep_c["n_keys_unparsed"] == "LT5" and rep_c["n_keys_trimmed"] == "LT5" and rep_c["n_keys_without_v4_walk"] == "LT5")
    chosen, _ = K.join(["S00__w0_down2", "S00__w0"], index)
    ok("a walk prefers its un-suffixed key", chosen[("S00", "w0")][0] == "S00__w0" and chosen[("S00", "w0")][2] == 2)
    bad = FakeNpz(npz)
    k0 = next(iter(bad))
    arr = np.array(bad[k0])
    arr[0, 0, 0] = np.nan
    bad[k0] = arr
    ok("non-finite arrays are counted, not used", K.cohort_report(bad, index, speed_fn)["n_arrays_nonfinite"] == "LT5")

    ok("normalization files: good pair", K.stats_report(np.zeros(K.FEAT, np.float32), np.ones(K.FEAT, np.float32)))
    ok("normalization files: zero std, wrong shape and NaN are refused",
       not K.stats_report(np.zeros(K.FEAT, np.float32), np.zeros(K.FEAT, np.float32))
       and not K.stats_report(np.zeros(5, np.float32), np.ones(5, np.float32))
       and not K.stats_report(np.full(K.FEAT, np.nan, np.float32), np.ones(K.FEAT, np.float32)))

    import torch
    import momask_encoder as M
    torch.manual_seed(0)
    sd = M.RVQVAE().state_dict()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ck.tar"
        torch.save({"net": sd, "ep": 49, "total_it": 12345}, p)
        ck = K.checkpoint_report(torch.load(p, map_location="cpu", weights_only=True))
    ok("checkpoint: strict load and counters", ck["strict_load_ok"] and ck["tensor_count_as_expected"]
       and ck["scalar_counters"] == {"ep": 49, "total_it": 12345})
    sd2 = dict(sd)
    sd2.pop(next(iter(sd2)))
    ok("checkpoint: a missing tensor fails the strict load", not K.checkpoint_report({"net": sd2})["strict_load_ok"])
    ok("checkpoint: wrong container fails", not K.checkpoint_report({"model": sd})["strict_load_ok"])

    reports = {c: rep for c in K.COHORTS}
    ok("decision: pass", K.decide(reports, True, ck) == "C2_PRECHECK_PASS")
    ok("decision: overlap failure", K.decide(dict(reports, BMCLab=rep_c), True, ck) == "C2_PRECHECK_FAIL_OVERLAP")
    ok("decision: normalization failure", K.decide(reports, False, ck) == "C2_PRECHECK_FAIL_NORMALIZATION_FILES")
    ok("decision: checkpoint failure", K.decide(reports, True, {"strict_load_ok": False}) == "C2_PRECHECK_FAIL_CHECKPOINT")
    out = {"card": "encoder-precheck", "card_sha256": "0" * 64, "run_commit": "a" * 40, "protocol_class": "METADATA", "seed": 1,
           "status": "C2_PRECHECK_PASS", "labels_read": False, "thresholds": {"min_walk_coverage": 0.9},
           "cohorts": dict(reports, BMCLab=rep_c), "cohort_passes": {c: True for c in K.COHORTS},
           "normalization_files_ok": {c: True for c in K.COHORTS}, "checkpoint": ck, "compute": IC.Receipt().block()}
    ok("would-be aggregate passes the renderer", len(IC.render(out)) < IC.MAX_BYTES)
    test_cli()
    print(f"ENCODER_PRECHECK_TESTS_PASS n={PASS}")


def cohort_inputs(seed, shuffle_speed=False):
    """A fabricated canonical cohort with pose and labels added, and its HumanML3D collection."""
    cohort, npz = fabricate(seed=seed, shuffle_speed=shuffle_speed)
    rng = np.random.default_rng(seed + 100)
    for s, walks in cohort.items():
        for w, rec in walks.items():
            t = rec["trans"].shape[0]
            rec.update({"pose": rng.normal(0, 0.2, (t, 72)).astype(np.float32), "beta": np.zeros(10, np.float32),
                        "UPDRS_GAIT": int(rng.integers(0, 3)), "medication": None, "other": None})
    return cohort, dict(npz)


def test_cli():
    import torch
    import encoder_inputs as F
    import momask_encoder as M
    import provenance_fixtures as FX
    import run_provenance as RP
    saved = FX.snapshot()
    try:
        with tempfile.TemporaryDirectory() as d:
            data, out = Path(d) / "data", Path(d) / "out"
            argv = ["encoder-precheck", "--data-dir", str(data), "--out-dir", str(out)]
            code, text = FX.quiet_main(RP, argv)
            ok("cli: missing inputs stop before any data is parsed", code == 20 and not out.exists())
            FX.layout(data)
            canon = {}
            for i, c in enumerate(F.COHORTS):
                canon[c], npz = cohort_inputs(10 + i)
                FX.pin_npz(c, npz)
            FX.pin_canonical(data, canon)
            torch.manual_seed(0)
            FX.pin_repository_files({"net": M.RVQVAE().state_dict(), "ep": 49, "total_it": 12345})
            code, text = FX.quiet_main(RP, argv)
            res = out / RP.result_name("encoder-precheck")
            agg = json.loads(res.read_text()) if res.is_file() else {}
            ok("cli: aligned fabricated inputs pass the frozen pre-check", code == 0 and agg.get("status") == "C2_PRECHECK_PASS"
               and all(agg["cohort_passes"].values()) and agg["checkpoint"]["strict_load_ok"]
               and all(agg["normalization_files_ok"].values()) and len(agg["normalization_files_ok"]) == 8)
            ok("cli: labels are never read and the aggregate holds no identifier or path",
               agg.get("labels_read") is False and all(x not in res.read_text() for x in ("S00", "w0", d)))
            ok("cli: an existing result is never overwritten", FX.quiet_main(RP, argv)[0] == 10)
            canon["BMCLab"], npz = cohort_inputs(99, shuffle_speed=True)
            FX.pin_npz("BMCLab", npz)
            FX.pin_canonical(data, canon)
            out2 = Path(d) / "out2"
            code, text = FX.quiet_main(RP, ["encoder-precheck", "--data-dir", str(data), "--out-dir", str(out2)])
            agg = json.loads((out2 / RP.result_name("encoder-precheck")).read_text())
            ok("cli: misaligned walks fail the overlap gate, the aggregate is kept and the exit code is 20",
               code == 20 and agg["status"] == "C2_PRECHECK_FAIL_OVERLAP" and agg["cohort_passes"]["BMCLab"] is False)
            stats = F.gh_path(f"{F.STATS}/LODO_BMCLab/Mean.npy")
            raw = stats.read_bytes()
            stats.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            out3 = Path(d) / "out3"
            code, text = FX.quiet_main(RP, ["encoder-precheck", "--data-dir", str(data), "--out-dir", str(out3)])
            ok("cli: a repository file that no longer matches its pin stops with no aggregate",
               code == 20 and not (out3 / RP.result_name("encoder-precheck")).exists())
    finally:
        FX.restore(saved)


if __name__ == "__main__":
    main()
