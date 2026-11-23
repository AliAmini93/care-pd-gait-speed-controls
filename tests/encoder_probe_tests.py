"""Fabricated-only tests of the frozen-encoder probe and the encoder-probe command. Random encoder weights.

No clinical file, no network, no released checkpoint. The speed-arm parity reference is a fabricated stand-in for
the user's own P1 value; no historical value is used.
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
import encoder_probe as A  # noqa: E402
import aggregate_core as IC  # noqa: E402
import momask_encoder as M  # noqa: E402
import p1_core as P1  # noqa: E402
import torch  # noqa: E402

PASS = 0
NAMES = ["3DGait", "BMCLab", "PD-GaM", "T-SDU-PD"]
FAB_S_M = 0.5            # fabricated stand-in for the speed-arm M of the user's own P1 aggregate


def ok(name, cond):
    global PASS
    if not cond:
        print(f"FAIL {name}")
        sys.exit(1)
    PASS += 1
    print(f"PASS {name}")


def fabricate(seed=0, per_cohort=10, walks=6, frames=48, speed_drives_label=True):
    """Four fabricated cohorts. The grade is a noisy function of speed; every non-root channel is label-free noise."""
    rng = np.random.default_rng(seed)
    label, cohort, subject, s, raws = [], [], [], [], []
    sid = 0
    for ci in range(4):
        for _ in range(per_cohort):
            base = rng.uniform(0.4, 1.4)
            for _ in range(walks):
                v = float(np.clip(base + rng.normal(0, 0.08), 0.1, 2.0))
                y = int(v < 0.75) + int(v < 1.05) if speed_drives_label else int(rng.integers(0, 3))
                raw = rng.normal(0, 1.0, size=(frames, 263)).astype(np.float32)
                raw[:, 1] = rng.normal(0, 0.002, frames)
                raw[:, 2] = v / 30.0 + rng.normal(0, 0.002, frames)
                label.append(y); cohort.append(ci); subject.append(sid); s.append(v); raws.append(raw)
            sid += 1
    t = {"label": np.array(label), "cohort": np.array(cohort), "subject": np.array(subject), "S_v4": np.array(s)}
    t["S_in"] = np.array([float(np.linalg.norm(r[:, 1:3], axis=1).mean() * 30.0) for r in raws])
    return t, raws


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    sd = M.RVQVAE().state_dict()
    for k in sd:
        if "codebook" in k:
            sd[k] = torch.randn_like(sd[k]) * 0.5
    model = M.load_strict(sd)
    ident = lambda name: (np.zeros(263, np.float32), np.ones(263, np.float32))

    # --- input ablations through the encoder ---
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 263)).astype(np.float32)
    x2 = x.copy(); x2[:, list(M.ROOT_CHANNELS)] += 3.0
    x3 = x.copy(); x3[:, 10:] += 3.0
    emb = lambda a, mode: M.walk_embedding(model, M.variant(a, mode))
    ok("no_root zeroes exactly the four root channels",
       np.all(M.variant(x, "no_root")[:, :4] == 0) and np.array_equal(M.variant(x, "no_root")[:, 4:], x[:, 4:]))
    ok("root_only keeps exactly the four root channels",
       np.array_equal(M.variant(x, "root_only")[:, :4], x[:, :4]) and np.all(M.variant(x, "root_only")[:, 4:] == 0))
    ok("E_noroot ignores a change of the root channels", np.allclose(emb(x, "no_root"), emb(x2, "no_root"), atol=0))
    ok("E_rootonly ignores a change of the other channels", np.allclose(emb(x, "root_only"), emb(x3, "root_only"), atol=0))
    ok("the full embedding responds to both changes",
       not np.allclose(emb(x, "full"), emb(x2, "full")) and not np.allclose(emb(x, "full"), emb(x3, "full")))
    ok("embedding is deterministic and 512-D", np.array_equal(emb(x, "full"), emb(x, "full")) and emb(x, "full").shape == (512,))
    ok("a walk below the minimum length is not encoded", M.walk_embedding(model, x[:7]) is None)
    ok("normalize uses the given statistics", np.allclose(M.normalize(np.full((2, 263), 3.0), np.ones(263), np.full(263, 2.0)), 1.0))

    # --- fold-wise outer loop equals the frozen outer loop when the features do not depend on the fold ---
    t, raws = fabricate()
    y, coh, subj = t["label"], t["cohort"], t["subject"]
    X = np.column_stack([t["S_v4"], np.random.default_rng(2).normal(size=y.size)])
    w1, w2 = [0], [0]
    ref = P1.outer_run(X, y, coh, subj, "walk", P1.CONFIGS, w1)
    got = A.outer_run_foldwise(P1, [X] * 4, y, coh, subj, "walk", P1.CONFIGS, w2, 4)
    same = all(np.array_equal(r["inner"], g["inner"]) and r["selected"] == g["selected"]
               and all(np.array_equal(a, b) for a, b in zip(r["proba"], g["proba"])) and np.array_equal(r["test_mask"], g["test_mask"])
               for r, g in zip(ref, got))
    ok("fold-wise loop reproduces the frozen P1 outer_run exactly", same)
    Xb = [X + (ci == 0) * 5.0 for ci in range(4)]
    got_b = A.outer_run_foldwise(P1, Xb, y, coh, subj, "walk", P1.CONFIGS[:1], [0], 4)
    ref_b = P1.outer_run(X, y, coh, subj, "walk", P1.CONFIGS[:1], [0])
    ok("a fold uses its own feature matrix", np.allclose(got_b[1]["proba"][0], ref_b[1]["proba"][0])
       and not np.array_equal(got_b[0]["inner"], ref_b[0]["inner"]) or np.allclose(got_b[0]["proba"][0], ref_b[0]["proba"][0]))

    # --- concordance tables ---
    yy = np.array([0, 0, 1, 1, 2, 2]); ss = np.array([0, 0, 1, 1, 2, 2]); people = np.array([0, 1, 2])
    sc = np.array([0.1, 0.2, 0.5, 0.6, 0.9, 1.0])
    c, n = A.pair_tables(yy, sc, ss, people)
    ok("perfect ranking gives concordance 1", A.concordance(c, n) == 1.0)
    c, n = A.pair_tables(yy, -sc, ss, people)
    ok("reversed ranking gives concordance 0", A.concordance(c, n) == 0.0)
    c, n = A.pair_tables(yy, np.zeros(6), ss, people)
    ok("a constant score gives concordance one half", A.concordance(c, n) == 0.5)
    y2 = np.array([0, 2, 0, 2]); s2 = np.array([0, 0, 1, 1])
    c, n = A.pair_tables(y2, np.array([0.9, 0.1, 0.2, 0.8]), s2, np.array([0, 1]))
    ok("pairs inside one participant are excluded", n[0, 0] == 0 and n[1, 1] == 0 and n.sum() == 4)
    c, n = A.pair_tables(yy, sc, ss, people)
    ones = np.ones((3, 3))
    ok("unit bootstrap weights reproduce the point value", np.allclose(A.concordance(c, n, ones), A.concordance(c, n)))
    w = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    val = A.concordance(c, n, w)
    ok("a draw with one participant only is undefined, not zero", np.isfinite(val[0]) and np.isnan(val[1]))

    # --- frozen decision rules ---
    ok("verdict: beyond needs margin, positive bound and three cohorts",
       A.verdict(0.06, 0.01, 0.1, [0.1, 0.1, 0.0, -0.01]) == "BEYOND"
       and A.verdict(0.06, 0.01, 0.1, [0.2, 0.2, -0.01, -0.01]) == "NO_DETECTED_DIFFERENCE"
       and A.verdict(0.04, 0.01, 0.1, [0.1] * 4) == "NO_DETECTED_DIFFERENCE"
       and A.verdict(0.06, -0.01, 0.1, [0.1] * 4) == "NO_DETECTED_DIFFERENCE")
    ok("verdict: below needs an upper bound under zero",
       A.verdict(-0.1, -0.2, -0.01, [-0.1] * 4) == "BELOW" and A.verdict(-0.1, -0.2, 0.0, [-0.1] * 4) == "NO_DETECTED_DIFFERENCE")
    ok("ablation rule", A.dependence_verdict(0.1, 0.02, 0.2) == "ROOT_CHANNEL_DEPENDENT"
       and A.dependence_verdict(0.0, -0.03, 0.03) == "NOT_ROOT_CHANNEL_DEPENDENT"
       and A.dependence_verdict(0.03, -0.02, 0.09) == "INCONCLUSIVE" and A.dependence_verdict(0.06, -0.01, 0.2) == "INCONCLUSIVE")
    sm = A.summary(0.5, np.array([0.4, 0.5, 0.6, np.nan]))
    ok("summary ignores and counts undefined draws", sm["draws_undefined"] == 1 and sm["ci95"][0] < 0.5 < sm["ci95"][1])
    gf = A.grouped_folds(np.array([5, 5, 7, 7, 9, 9, 11]), 2)
    ok("grouped folds never split a participant", gf.tolist() == [0, 0, 1, 1, 0, 0, 1])

    # --- end to end on fabricated cohorts with a planted speed signal (two fast configs, few draws) ---
    A.ARCHIVED_S_M = FAB_S_M
    meta = {"card": "encoder-probe", "card_sha256": "0" * 64, "run_commit": "a" * 40, "protocol_class": "SCORED", "seed": 1}
    counts = {n: {"n_walks_v4": 60, "n_walks_joined": 60, "n_grade3_excluded": 0, "n_unlabelled": 0, "n_unusable_input": 0} for n in NAMES}
    out = A.analyse(meta, t, raws, counts, model, ident, P1, M, NAMES, n_boot=200, configs=P1.CONFIGS[:2])
    out["compute"] = IC.Receipt().block(threads=1)
    text = IC.render(out)
    ok("end to end: aggregate passes the renderer", len(text) < IC.MAX_BYTES // 2)
    ok("end to end: all arms and contrasts are present",
       set(out["arms"]) == set(A.ARMS) and len(out["contrasts"]) == len(A.CONTRASTS) == len(out["concordance_contrasts"]))
    m = lambda arm: out["arms"][arm]["macro_f1"]["M"]["point"]
    ok("planted signal: speed arms recover the grade", m("S_v4") > 0.7 and m("S_in") > 0.7)
    ok("planted signal: without root channels the encoder has nothing", m("E_noroot") < 0.5 and m("E_rootonly") > m("E_noroot"))
    ok("planted signal: verdicts come from the frozen vocabulary",
       out["verdicts"]["primary_E_vs_S_in"] in ("BEYOND", "BELOW", "NO_DETECTED_DIFFERENCE")
       and out["verdicts"]["original_primary_E_vs_S_v4"] in ("BEYOND", "BELOW", "NO_DETECTED_DIFFERENCE")
       and isinstance(out["verdicts"]["primary_and_original_agree"], bool)
       and out["verdicts"]["root_channel_ablation"] in ("ROOT_CHANNEL_DEPENDENT", "NOT_ROOT_CHANNEL_DEPENDENT", "INCONCLUSIVE")
       and out["status"] == "C3X_ENCODER_" + out["verdicts"]["primary_E_vs_S_in"] + "_INPUT_SPEED")
    cc = out["between_participant_concordance"]
    ok("planted signal: speed ranks participants well", cc["S_in"]["M"]["point"] > 0.8 and 0.3 < cc["E_noroot"]["M"]["point"] < 0.7)
    blk = out["contrasts"]["E_minus_S_in"]
    three = np.mean([blk["per_cohort"][n]["point"] for n in A.C2_PASSING])
    ok("three-cohort sensitivity is the mean over the cohorts that passed gate c2",
       abs(blk["M_three_c2_passing_cohorts"]["point"] - three) < 1e-6 and "3DGait" not in A.C2_PASSING)
    ok("input frame rate: same frames keeps the rate, fivefold downsampling divides it",
       abs(A.input_fps(30.0, 99, 100) - 30.0) < 1e-9 and abs(A.input_fps(150.0, 199, 1000) - 30.0) < 1e-9)
    ok("parity block is honest on a fabricated set", out["speed_arm_parity"]["walk_set_equals_archived"] is False
       and out["speed_arm_parity"]["exact_parity"] is False)
    ok("parity block reports the supplied P1 value, never a bundled one", out["speed_arm_parity"]["archived_S_M"] == FAB_S_M
       and "no historical value" in out["design"]["gate_c2"] and not hasattr(A, "main"))
    t0, raws0 = fabricate(seed=3, speed_drives_label=False)
    out0 = A.analyse(meta, t0, raws0, counts, model, ident, P1, M, NAMES, n_boot=200, configs=P1.CONFIGS[:2])
    ok("null world: no arm is declared beyond speed", out0["verdicts"]["primary_E_vs_S_in"] != "BEYOND"
       and out0["verdicts"]["fusion_S_in_plus_E_vs_S_in"] != "BEYOND")
    test_cli(sd)
    print(f"ENCODER_PROBE_TESTS_PASS n={PASS}")


def cli_inputs(seed=5, subjects=8, walks=4, frames=40, fps=30):
    """Fabricated canonical cohorts and HumanML3D collections in which speed drives the grade."""
    rng = np.random.default_rng(seed)
    canon, npz = {}, {}
    for ci, name in enumerate(NAMES):
        canon[name], npz[name] = {}, {}
        for s in range(subjects):
            subj = f"P{ci}{s:02d}"
            canon[name][subj] = {}
            for w in range(walks):
                v = float(np.clip(0.45 + 0.12 * s + rng.normal(0, 0.03), 0.1, 2.0))
                y = int(v < 0.75) + int(v < 1.05)
                z = np.arange(frames) * v / fps
                trans = np.column_stack([np.zeros(frames), np.full(frames, 0.9), z]).astype(np.float32)
                canon[name][subj][f"w{w}"] = {"pose": rng.normal(0, 0.2, (frames, 72)).astype(np.float32), "trans": trans,
                                              "beta": np.zeros(10, np.float32), "fps": fps, "UPDRS_GAIT": y,
                                              "medication": None, "other": None}
                raw = rng.normal(0, 1.0, size=(3, frames - 1, 263)).astype(np.float32)
                raw[:, :, 1] = rng.normal(0, 0.002, (3, frames - 1))
                raw[:, :, 2] = v / fps + rng.normal(0, 0.002, (3, frames - 1))
                npz[name][f"{subj}__w{w}"] = raw
    return canon, npz


def test_cli(sd):
    import encoder_inputs as F
    import provenance_fixtures as FX
    import run_provenance as RP
    saved, saved_analyse, saved_sm = FX.snapshot(), A.analyse, A.ARCHIVED_S_M
    try:
        with tempfile.TemporaryDirectory() as d:
            data, out = Path(d) / "data", Path(d) / "out"
            argv = ["encoder-probe", "--data-dir", str(data), "--out-dir", str(out)]
            code, text = FX.quiet_main(RP, argv)
            ok("cli: refuses without the user's P0 and P1 before reading data", code == 20
               and "RELEASE_UPSTREAM_MISSING" in text and not (out / RP.result_name("encoder-probe")).exists())
            FX.layout(data)
            canon, npz = cli_inputs()
            FX.pin_canonical(data, canon)
            for name in F.COHORTS:
                FX.pin_npz(name, npz[name])
            FX.pin_repository_files({"net": sd})
            FX.upstream(out, s_m=0.4321)
            A.analyse = lambda *a: saved_analyse(*a, n_boot=200, configs=P1.CONFIGS[:2])
            code, text = FX.quiet_main(RP, argv)
            res = out / RP.result_name("encoder-probe")
            agg = json.loads(res.read_text()) if res.is_file() else {}
            ok("cli: fabricated end-to-end probe completes", code == 0 and str(agg.get("status", "")).startswith("C3X_ENCODER_")
               and agg.get("card") == "encoder-probe" and agg.get("release_scope") == RP.SCOPE
               and set(agg.get("arms", {})) == set(A.ARMS))
            ok("cli: the speed-arm parity reference is the user's own P1 value",
               agg["speed_arm_parity"]["archived_S_M"] == 0.4321 and agg["speed_arm_parity"]["exact_parity"] is False)
            ok("cli: eligible walks and participants follow the fabricated join",
               agg["n_walks_eligible"] == 128 and agg["n_participants_eligible"] == 32)
            ok("cli: aggregate holds no identifier or local path", all(x not in res.read_text() for x in ("P000", "__w", d)))
            ok("cli: an existing result is never overwritten", FX.quiet_main(RP, argv)[0] == 10)
    finally:
        A.analyse, A.ARCHIVED_S_M = saved_analyse, saved_sm
        FX.restore(saved)


if __name__ == "__main__":
    main()
