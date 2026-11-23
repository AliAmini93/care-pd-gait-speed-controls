"""Fabricated-only tests of the aggregate renderer, the network tripwire and the strict restricted pickle reader.

No clinical file and no network. Covers: renderer refusals (identifier-like keys, small counts, sizes, non-finite
values), the small-cell rule, the audit-hook network tripwire; the strict reader against malicious pickles
(forbidden globals, dynamic import, REDUCE on a class, BUILD on a container, tampered dtype and array states, object
and unicode dtypes, EXT/NEWOBJ/PERSID, sets, dict keys, truncated and trailing bytes, HTML), absence of imports while
decoding, agreement with common.safe_loads on legitimate pickles, tree equality and the label-free structure census.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "care_pd"
sys.path.insert(0, str(SCRIPTS))
import aggregate_core as IC  # noqa: E402
import common as C  # noqa: E402
import root_core as G  # noqa: E402
import root_fetch as D  # noqa: E402

PASS = 0


def ok(name, cond):
    global PASS
    if not cond:
        print(f"FAIL {name}")
        sys.exit(1)
    PASS += 1
    print(f"PASS {name}")


def exits(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def refused(fn, *args, code=None):
    try:
        fn(*args)
    except G.SourceError as exc:
        return code is None or str(exc).startswith(code)
    except Exception:
        return False
    return False


class Sentinel:
    """A clinical value that raises on ANY attribute access (repr, eq, hash, class lookups through getattr...)."""

    def __getattribute__(self, name):
        raise AssertionError("clinical value touched")


def fab_cohort(rng, n_subj=6, walks=2, int_keys=True, clinical=None):
    out = {}
    for s in range(n_subj):
        key = 100 + s if int_keys else f"S{s:03d}"
        out[key] = {}
        for w in range(walks):
            T = 40 + 3 * s + w
            rec = {"pose": rng.normal(size=(T, 72)).astype(np.float32),
                   "trans": np.cumsum(rng.normal(0, 0.01, size=(T, 3)), axis=0).astype(np.float32),
                   "beta": np.zeros(10, dtype=np.float32), "fps": 30}
            if clinical is None:
                rec.update({"UPDRS_GAIT": int((s + w) % 4) if w else None, "medication": "on" if w else None,
                            "other": None})
            else:
                rec.update({"UPDRS_GAIT": clinical(), "medication": clinical(), "other": clinical()})
            out[key][f"walk_{w}"] = rec
    return out


class Reduce:
    def __init__(self, fn, args, state=None):
        self.r = (fn, args) if state is None else (fn, args, state)

    def __reduce__(self):
        return self.r


def malicious():
    """name -> (bytes, expected refusal code prefix or None for any)."""
    import builtins
    import importlib
    out = {}
    out["global os.system"] = (pickle.dumps(Reduce(os.system, ("true",)), protocol=2), "PICKLE_GLOBAL_REFUSED")
    out["stack_global builtins.eval"] = (pickle.dumps(Reduce(builtins.eval, ("1+1",)), protocol=4), "PICKLE_GLOBAL_REFUSED")
    out["dynamic __import__"] = (pickle.dumps(Reduce(builtins.__import__, ("os",)), protocol=4), "PICKLE_GLOBAL_REFUSED")
    out["dynamic importlib.import_module"] = (pickle.dumps(Reduce(importlib.import_module, ("os",)), protocol=3),
                                              "PICKLE_GLOBAL_REFUSED")
    out["numpy non-allowlisted attr"] = (b"\x80\x02cnumpy.core.multiarray\nfromstring\nq\x00.", "PICKLE_GLOBAL_REFUSED")
    out["numpy.load"] = (pickle.dumps(Reduce(np.load, ("x",)), protocol=4), "PICKLE_GLOBAL_REFUSED")
    out["EXT1"] = (b"\x80\x02\x82\x01.", "PICKLE_OPCODE_REFUSED")
    out["EXT2"] = (b"\x80\x02\x83\x01\x00.", "PICKLE_OPCODE_REFUSED")
    out["EXT4"] = (b"\x80\x02\x84\x01\x00\x00\x00.", "PICKLE_OPCODE_REFUSED")
    out["NEWOBJ on allowlisted class"] = (b"\x80\x02cnumpy\nndarray\n)\x81.", "PICKLE_OPCODE_REFUSED")
    out["INST"] = (b"(inumpy\nndarray\n.", "PICKLE_OPCODE_REFUSED")
    out["BINPERSID"] = (b"\x80\x02X\x01\x00\x00\x00aQ.", "PICKLE_OPCODE_REFUSED")
    out["PERSID"] = (b"Pa\n.", "PICKLE_OPCODE_REFUSED")
    # allowlisted class called directly through REDUCE: passes the existing static scan, strict reader refuses
    out["REDUCE numpy.ndarray class"] = (b"\x80\x02cnumpy\nndarray\nK\x03\x85R.", "PICKLE_REDUCE_TARGET_REFUSED")
    out["BUILD on a dict"] = (b"\x80\x02}q\x00}q\x01b.", "PICKLE_BUILD_TARGET_REFUSED")
    out["BUILD on a list"] = (b"\x80\x02]q\x00}q\x01b.", "PICKLE_BUILD_TARGET_REFUSED")
    out["dtype state with object flag"] = (
        pickle.dumps(Reduce(np.dtype, ("f8", False, True), (3, "<", None, None, None, -1, -1, 63)), protocol=4),
        "PICKLE_DTYPE_STATE_REFUSED")
    out["dtype state with fields"] = (
        pickle.dumps(Reduce(np.dtype, ("f8", False, True), (3, "<", None, ("a",), {"a": (1, 0)}, 8, 1, 0)), protocol=4),
        None)
    out["big-endian dtype state"] = (
        pickle.dumps(Reduce(np.dtype, ("f8", False, True), (3, ">", None, None, None, -1, -1, 0)), protocol=4),
        "PICKLE_DTYPE_STATE_REFUSED")
    rec = np.core.multiarray._reconstruct
    out["array state wrong length"] = (
        pickle.dumps(Reduce(rec, (np.ndarray, (0,), b"b"), (1, (4,), np.dtype("f8"), False, b"\x00" * 8)), protocol=4),
        "PICKLE_ARRAY_STATE_REFUSED")
    out["array state list payload"] = (
        pickle.dumps(Reduce(rec, (np.ndarray, (0,), b"b"), (1, (2,), np.dtype("f8"), False, [1.0, 2.0])), protocol=4),
        "PICKLE_ARRAY_STATE_REFUSED")
    out["object dtype array"] = (pickle.dumps(np.array([1, "a"], dtype=object), protocol=4), "PICKLE_DTYPE_REFUSED")
    out["unicode dtype array"] = (pickle.dumps(np.array(["ab", "c"]), protocol=4), "PICKLE_DTYPE_REFUSED")
    out["set via EMPTY_SET"] = (pickle.dumps({"k": {1, 2}}, protocol=4), "PICKLE_OPCODE_REFUSED_STRICT")
    out["frozenset"] = (pickle.dumps(frozenset({1}), protocol=4), "PICKLE_OPCODE_REFUSED_STRICT")
    out["float dict key"] = (pickle.dumps({1.5: 1}, protocol=4), "PICKLE_DICT_KEY_REFUSED")
    out["tuple dict key"] = (pickle.dumps({(1, 2): 1}, protocol=4), "PICKLE_DICT_KEY_REFUSED")
    out["protocol 2 array uses _codecs.encode"] = (pickle.dumps(np.zeros(2, np.float32), protocol=2), "PICKLE_GLOBAL_REFUSED")
    legit = pickle.dumps({"a": np.zeros(3, np.float32)}, protocol=4)
    out["truncated"] = (legit[:-7], None)
    out["trailing bytes after STOP"] = (legit + b"\x00\x00", "PICKLE_TRAILING_BYTES")
    out["html page"] = (b"<!DOCTYPE html><html><body>Sign in</body></html>", None)
    out["empty"] = (b"", None)
    return out


# ---- aggregate renderer and network tripwire ----------------------------------------------------------------------
def base():
    return {"card": "demo-command", "card_sha256": "0" * 64, "run_commit": "a" * 40, "protocol_class": "METADATA",
            "status": "OK", "compute": {"device": "cpu", "elapsed_s": 1.0, "peak_rss_mb": 1.0, "peak_gpu_allocated_mb": 0.0}}


def test_aggregate():
    good = base()
    good["cohorts"] = {"A": {"n_walks": 80, "n_participants": 40, "macro_f1": 0.5}, "B": {"n_walks": 0}}
    text = IC.render(good)
    ok("valid aggregate renders", json.loads(text)["cohorts"]["A"]["n_walks"] == 80)
    for name, mutate in (
        ("forbidden id key", lambda d: d.update(walk_id_list=[1])),
        ("forbidden per-walk key", lambda d: d.update(per_walk_scores=[0.1])),
        ("forbidden embedding key", lambda d: d.update(embedding=[0.1])),
        ("small count", lambda d: d["cohorts"]["A"].update(n_participants=3)),
        ("boolean count", lambda d: d["cohorts"]["A"].update(n_participants=True)),
        ("long list", lambda d: d.update(values=list(range(IC.MAX_LIST + 1)))),
        ("large dict", lambda d: d.update(table={f"k{i}": i for i in range(IC.MAX_DICT + 1)})),
        ("long string", lambda d: d.update(note="x" * (IC.MAX_STR + 1))),
        ("non-finite", lambda d: d.update(value=float("nan"))),
        ("missing top key", lambda d: d.pop("compute")),
        ("object type", lambda d: d.update(value={1, 2})),
    ):
        bad = base()
        bad["cohorts"] = {"A": {"n_walks": 80, "n_participants": 40}}
        mutate(bad)
        ok(f"renderer refuses {name}", exits(lambda b=bad: IC.render(b)))
    ok("count rule suppresses 1..4", [IC.count(i) for i in (0, 1, 4, 5, 9)] == [0, "LT5", "LT5", 5, 9])
    sup = base()
    sup["n_unmatched"] = IC.count(3)
    ok("suppressed marker is accepted", "LT5" in IC.render(sup))
    big = base()
    big["blocks"] = [{"text": "y" * 200} for _ in range(60)]
    big["more"] = {f"k{i}": [{"t": "z" * 200} for _ in range(60)] for i in range(30)}
    ok("renderer refuses oversize output", exits(lambda: IC.render(big)))

    # network tripwire (subprocess, so the hook does not stay in this interpreter)
    code = ("import sys; sys.path.insert(0, %r); import aggregate_core as IC; IC.block_network(); import socket\n"
            "try:\n socket.getaddrinfo('example.invalid', 443); print('OPEN')\n"
            "except PermissionError: print('BLOCKED')\n") % str(SCRIPTS)
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
    ok("network tripwire blocks name resolution", r.stdout.strip() == "BLOCKED")
    code2 = code.replace("IC.block_network()", "IC.block_network(('allowed.invalid',))")
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code2], capture_output=True, text=True)
    ok("network tripwire blocks a host outside the allowlist", r.stdout.strip() == "BLOCKED")
    ok("host rule: exact name", IC.host_allowed("borealisdata.ca", ("borealisdata.ca",)))
    ok("host rule: suffix needs a subdomain", IC.host_allowed("a.b.example.org", (".example.org",))
       and not IC.host_allowed("example.org", (".example.org",)) and not IC.host_allowed("badexample.org", (".example.org",)))
    ok("host rule: address literal is not allowed", not IC.host_allowed("93.184.216.34", ("borealisdata.ca", ".example.org")))
    code3 = ("import sys; sys.path.insert(0, %r); import aggregate_core as IC; IC.block_network(('allowed.invalid',))\n"
             "import urllib.request\n"
             "for u in ('http://allowed.invalid/x', 'https://other.invalid/x'):\n"
             " try:\n  urllib.request.urlopen(u, timeout=2); print('OPEN')\n"
             " except PermissionError: print('BLOCKED')\n"
             " except Exception as e: print('OTHER', type(e).__name__)\n") % str(SCRIPTS)
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code3], capture_output=True, text=True)
    ok("network tripwire blocks plain http and foreign https requests", r.stdout.split() == ["BLOCKED", "BLOCKED"])
    ok("portable module defines no private result location",
       not any(hasattr(IC, k) for k in ("LOCAL", "RESULT_DIR", "BRANCH", "require_route", "save_aggregate", "result_path")))


# ---- strict reader, tree equality and census ----------------------------------------------------------------------
def test_strict_reader():
    IC.block_network()                      # empty allowlist: any real network event in these tests raises
    rng = np.random.default_rng(20261113)
    imports = []
    sys.addaudithook(lambda ev, args: imports.append(args[0]) if ev == "import" else None)
    n0 = len(imports)
    __import__("colorsys")                  # positive control: a fresh import must be recorded by the hook
    ok("import audit hook records a fresh import (positive control)", len(imports) > n0 and "colorsys" in imports)

    # --- 1. strict reader: legitimate pickles -----------------------------------------------------------------
    cohort = fab_cohort(rng)
    for proto in (3, 4, 5):
        data = pickle.dumps(cohort, protocol=proto)
        n0 = len(imports)
        obj = G.strict_loads(data)
        ok(f"legit protocol {proto}: strict load equals the original (all values, incl. clinical)",
           G.trees_equal(obj, cohort, skip_value_keys=()))
        ok(f"legit protocol {proto}: strict and existing loaders agree", G.trees_equal(obj, C.safe_loads(data), ()))
        ok(f"legit protocol {proto}: no module imported while decoding", len(imports) == n0)
        scan = G.static_scan(data)
        ok(f"legit protocol {proto}: static scan ok, ends at STOP, no trailing bytes",
           scan["ok"] and scan["ends_with_stop"] and scan["trailing_bytes"] == 0 and scan["protocol"] == proto)
    scalars = {"f": np.float64(1.5), "i": np.int64(3), "b": np.bool_(True), "u": np.zeros((2, 2), np.uint8),
               "nested": [(1, 2.0, None, "x", b"y")], 7: True}
    data = pickle.dumps(scalars, protocol=4)
    ok("numpy scalars, bool/uint8 arrays, tuples, bytes and int keys decode", G.trees_equal(G.strict_loads(data), scalars, ()))
    fortran = {"a": np.asfortranarray(rng.normal(size=(5, 4)))}
    ok("Fortran-ordered array decodes equal", G.trees_equal(G.strict_loads(pickle.dumps(fortran, protocol=3)), fortran, ()))

    # --- 2. strict reader: malicious or out-of-scope pickles -----------------------------------------------------
    for name, (blob, code) in malicious().items():
        n0 = len(imports)
        ok(f"refused: {name}", refused(G.strict_loads, blob, code=code))
        ok(f"no import attempted: {name}", len(imports) == n0)
    blob = malicious()["REDUCE numpy.ndarray class"][0]
    ok("existing static scan alone would pass REDUCE-on-class (the strict layer is what refuses it)",
       C.scan_opcodes(blob)[0] == 2)
    blob = malicious()["trailing bytes after STOP"][0]
    ok("existing scan ignores trailing bytes; strict census flags them",
       C.scan_opcodes(blob)[0] == 4 and G.static_scan(blob)["reason"] == "PICKLE_TRAILING_BYTES")
    ok("format sniff: html", G.format_sniff(b"<!DOCTYPE") == "html_or_xml")
    ok("format sniff: pickle", G.format_sniff(pickle.dumps({}, protocol=4)) == "pickle_protocol_2_or_higher")
    ok("format sniff: zlib / gzip", G.format_sniff(b"\x78\x9cabc") == "zlib" and G.format_sniff(b"\x1f\x8b") == "gzip")
    ok("reason sanitizer keeps code identifiers only", G.sanitize_reason("A:b.c d<e>'f'") == "A:b.cdef")
    ok("tree check refuses an object array", refused(G.tree_check, {"a": np.array([None], dtype=object)}))
    ok("tree check refuses a set", refused(G.tree_check, {"a": {1}}))
    ok("tree check refuses a dtype leaf", refused(G.tree_check, [np.dtype("f8")]))

    # --- 3. tree equality ----------------------------------------------------------------------------------------
    a = fab_cohort(np.random.default_rng(1))
    b = fab_cohort(np.random.default_rng(1))
    ok("identical trees are equal", G.trees_equal(a, b))
    b[100]["walk_1"]["UPDRS_GAIT"] = 3
    ok("clinical value differences are NOT inspected by the loader-agreement check", G.trees_equal(a, b))
    ok("but they are seen when the skip list is empty", not G.trees_equal(a, b, skip_value_keys=()))
    b = fab_cohort(np.random.default_rng(1))
    b[101]["walk_0"]["trans"] = b[101]["walk_0"]["trans"].copy()
    b[101]["walk_0"]["trans"][3, 1] += 1e-3
    ok("a changed motion value is detected", not G.trees_equal(a, b))
    b = fab_cohort(np.random.default_rng(1))
    del b[102]["walk_1"]["other"]
    ok("a missing clinical key is detected (presence is compared)", not G.trees_equal(a, b))
    b = {k: dict(reversed(list(v.items()))) for k, v in reversed(list(fab_cohort(np.random.default_rng(1)).items()))}
    ok("reordered keys are still equal", G.trees_equal(a, b))

    # --- 4. label-free structure census --------------------------------------------------------------------------
    sent = fab_cohort(np.random.default_rng(2), clinical=Sentinel)
    try:
        cs = G.structure_census(sent)
        touched = False
    except AssertionError:
        touched = True
    ok("census never touches a clinical value (sentinel raises on any access)", not touched)
    ok("census counts", cs["walks"] == 12 and cs["subjects"] == 6 and cs["subject_key_types"]["int"] == 6
       and cs["pose_conforming"] == 12 and cs["beta_all_zero"] == 12 and cs["fps_values"] == {30.0: 12})
    ok("census: all documented fields present", all(v == 12 for v in cs["field_present"].values()))
    ok("schema accepted for the documented layout", D.schema_ok(cs))
    shuffled = {k: dict(reversed(list(v.items()))) for k, v in reversed(list(sent.items()))}
    cs2 = G.structure_census(shuffled)
    ok("census is invariant to key order", {k: v for k, v in cs2.items() if k != "frames"} ==
       {k: v for k, v in cs.items() if k != "frames"} and sorted(cs2["frames"]) == sorted(cs["frames"]))
    bad = fab_cohort(np.random.default_rng(3))
    bad[100]["walk_0"]["pose"] = bad[100]["walk_0"]["pose"][:, :69]
    del bad[101]["walk_1"]["trans"]
    bad[102]["walk_0"]["extra_field"] = 1
    csb = G.structure_census(bad)
    ok("wrong root schema detected (pose width, missing trans)", not D.schema_ok(csb) and csb["pose_conforming"] == 11
       and csb["field_present"]["trans"] == 11 and csb["records_with_unknown_fields"] == 1)
    ok("top-level list is not a cohort", not D.schema_ok(G.structure_census([bad])))
    mixed = {1: {"w": a[100]["walk_0"]}, "1": {"w": a[100]["walk_1"]}}
    ok("int/str subject keys colliding after str() are flagged", G.structure_census(mixed)["subject_key_str_collisions"] == 1
       and not D.schema_ok(G.structure_census(mixed)))
    summary = D.census_out(cs)
    ok("census summary: totals are bands, no exact small counts", summary["walks_band"] == "10_to_14"
       and summary["subjects_band"] == "5_to_9" and summary["n_records_nonconforming_motion"] == 0)
    with tempfile.TemporaryDirectory() as td:
        saved = G.ROOT_DIR
        G.ROOT_DIR = Path(td) / "original"
        try:
            ok("root files live under the configured portable folder",
               all(G.root_path(c) == Path(td) / "original" / c / G.ROOT_PINS[c][0] for c in G.COHORTS))
        finally:
            G.ROOT_DIR = saved


def main():
    test_aggregate()
    test_strict_reader()
    print(f"ROOT_CORE_TESTS_PASS n={PASS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
