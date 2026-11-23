"""Fabricated-only tests of the pinned original-file download and its command line. No network, no clinical file.

Covers: publisher-record checks, the redirect guard, the fail-closed download path (checksum, size, content type,
redirects, login, rate limit, retries, no overwrite, file modes), process() (reuse, mismatch, gating, decode verdicts
of unsafe pickles with matching digests), the opcode-count hardening, the rest-stage prerequisites, and the
fetch-originals command with a stubbed opener, temporary folders and fabricated files.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import stat
import subprocess
import sys
import tempfile
import urllib.error
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "care_pd"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_core as IC  # noqa: E402
import provenance_fixtures as FX  # noqa: E402
import root_core as G  # noqa: E402
import root_core_tests as T0  # noqa: E402
import root_fetch as D  # noqa: E402
import run_provenance as RP  # noqa: E402

PASS = 0


def ok(name, cond):
    global PASS
    if not cond:
        print(f"FAIL {name}")
        sys.exit(1)
    PASS += 1
    print(f"PASS {name}")


# ---- fake network layer -------------------------------------------------------------------------------------
class FakeResp:
    def __init__(self, body, url="https://borealisdata.ca/api/access/datafile/1", ctype="application/octet-stream",
                 clen="auto", status=200):
        self.body, self.url, self.status = io.BytesIO(body), url, status
        self.headers = {"Content-Type": ctype}
        if clen == "auto":
            self.headers["Content-Length"] = str(len(body))
        elif clen is not None:
            self.headers["Content-Length"] = str(clen)

    def read(self, n=-1):
        return self.body.read(n)

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def open(self, req, timeout=None):
        self.calls += 1
        if not self.script:
            raise AssertionError("unexpected network call")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def http_error(code):
    return urllib.error.HTTPError("https://borealisdata.ca/x", code, "x", {}, None)


def dataset_doc(**over):
    latest = {"versionNumber": 4, "versionMinorNumber": 0, "versionState": "RELEASED", "termsOfAccess": None,
              "fileAccessRequest": False, "license": {"name": "CC BY-NC-ND 4.0"}}
    latest.update(over.pop("latest", {}))
    data = {"id": 1, "latestVersion": latest}
    data.update(over)
    return {"status": "OK", "data": data}


def file_doc(cohort, **over):
    logical, fid, size, md5 = G.ROOT_PINS[cohort][:4]
    df = {"id": fid, "filename": logical, "filesize": size, "md5": md5, "restricted": None, "tabularData": False,
          "fileAccessRequest": False}
    df.update(over.pop("df", {}))
    data = {"label": logical, "restricted": False, "dataFile": df}
    data.update(over)
    return {"status": "OK", "data": data}


def json_resp(obj, ctype="application/json;charset=UTF-8"):
    return FakeResp(json.dumps(obj).encode(), url="https://borealisdata.ca/api/files/1", ctype=ctype)


def test_kernels():
    # --- publisher record checks ------------------------------------------------------------------------------
    ok("dataset record: unchanged -> no problem", D.check_dataset_meta(dataset_doc())[0] == []
       and D.check_dataset_meta(dataset_doc())[1] == "CC BY-NC-ND 4.0")
    ok("dataset record: guestbook stops", "GUESTBOOK_PRESENT" in D.check_dataset_meta(dataset_doc(guestbookId=7))[0])
    ok("dataset record: new version stops", "LATEST_VERSION_CHANGED" in D.check_dataset_meta(dataset_doc(latest={"versionNumber": 5}))[0])
    ok("dataset record: terms of access stop", "TERMS_OF_ACCESS_PRESENT" in D.check_dataset_meta(dataset_doc(latest={"termsOfAccess": "sign"}))[0])
    ok("dataset record: draft stops", "VERSION_NOT_RELEASED" in D.check_dataset_meta(dataset_doc(latest={"versionState": "DRAFT"}))[0])
    ok("dataset record: unreadable stops", D.check_dataset_meta({"status": "ERROR"})[0] == ["DATASET_RECORD_UNREADABLE"])
    for cohort_name in G.COHORTS:
        ok(f"file record {cohort_name}: unchanged -> no problem", D.check_file_meta(file_doc(cohort_name), cohort_name) == [])
    ok("file record: restricted stops", "FILE_RESTRICTED" in D.check_file_meta(file_doc("3DGait", restricted=True), "3DGait"))
    ok("file record: changed MD5 stops", "FILE_DIGEST_CHANGED" in D.check_file_meta(file_doc("3DGait", df={"md5": "0" * 32}), "3DGait"))
    ok("file record: other datafile stops", "FILE_IDENTITY_CHANGED" in D.check_file_meta(file_doc("3DGait", df={"id": 926880}), "3DGait"))
    ok("file record: tabular stops", "FILE_TABULAR" in D.check_file_meta(file_doc("3DGait", df={"tabularData": True}), "3DGait"))
    ok("old PD-GaM datafile is not approved", 926880 in G.NOT_APPROVED_DATAFILES and G.ROOT_PINS["PD-GaM"][1] == 1160521)
    ok("pins equal the publisher table", [G.ROOT_PINS[c][1:4] for c in G.COHORTS] == [
        (926879, 8066161, "a4fb5b909b4a7d913530a8ac82ce9c99"), (926877, 127210064, "8744c40c868de8f32138673cc0b05485"),
        (926878, 27019026, "a1b978b7bd5e1d4edbb1306218c22d26"), (1160521, 84704001, "92884135f342f715ab4f5efd21b8ad7b")])
    ok("fetch_json: JSON accepted", D.fetch_json("https://borealisdata.ca/a", FakeOpener([json_resp({"status": "OK"})])) == {"status": "OK"})
    for name, item, code in (("html", json_resp({}, ctype="text/html"), "API_NOT_JSON"),
                             ("429", http_error(429), "RATE_LIMITED"), ("403", http_error(403), "LOGIN_OR_PERMISSION_REQUIRED"),
                             ("too large", FakeResp(b"{" + b" " * (D.API_CAP + 5) + b"}", ctype="application/json"), "API_TOO_LARGE")):
        try:
            D.fetch_json("https://borealisdata.ca/a", FakeOpener([item]))
            got = None
        except D.Stop as exc:
            got = str(exc)
        ok(f"fetch_json fails closed: {name}", got == code)
    try:
        D.fetch_json("http://borealisdata.ca/a", FakeOpener([]))
        got = None
    except D.Stop as exc:
        got = str(exc)
    ok("fetch_json refuses a non-HTTPS URL without a call", got == "NON_HTTPS_URL")

    # --- redirect guard ---------------------------------------------------------------------------------------
    import urllib.request as UR
    req = UR.Request("https://borealisdata.ca/api/access/datafile/1")
    h = D.GuardedRedirects(D.ALLOWED_HOSTS, 3)
    nxt = h.redirect_request(req, None, 302, "Found", {}, "https://s3.example.scholarsportal.info/b/k?sig=x")
    ok("redirect to the storage host class is followed and recorded without the URL",
       nxt is not None and h.hops == ["*.scholarsportal.info"])
    for name, url in (("http downgrade", "http://borealisdata.ca/x"), ("foreign host", "https://evil.example.com/x"),
                      ("look-alike host", "https://borealisdata.ca.evil.com/x"), ("literal address", "https://10.0.0.1/x")):
        try:
            D.GuardedRedirects(D.ALLOWED_HOSTS, 3).redirect_request(req, None, 302, "Found", {}, url)
            got = None
        except D.Stop as exc:
            got = str(exc)
        ok(f"redirect refused: {name}", got == "REDIRECT_HOST_REFUSED")
    h = D.GuardedRedirects(D.ALLOWED_HOSTS, 3)
    try:
        for _ in range(4):
            h.redirect_request(req, None, 302, "Found", {}, "https://borealisdata.ca/y")
        got = None
    except D.Stop as exc:
        got = str(exc)
    ok("more than three redirects stop", got == "TOO_MANY_REDIRECTS")
    try:
        D.GuardedRedirects(D.API_HOSTS, 0).redirect_request(req, None, 302, "Found", {}, "https://borealisdata.ca/y")
        got = None
    except D.Stop as exc:
        got = str(exc)
    ok("API calls accept no redirect", got == "TOO_MANY_REDIRECTS")

    # --- download path ----------------------------------------------------------------------------------------
    body = pickle.dumps(T0.fab_cohort(np.random.default_rng(4)), protocol=4)
    md5, sha = hashlib.md5(body).hexdigest(), hashlib.sha256(body).hexdigest()
    sleeps = []
    with tempfile.TemporaryDirectory() as td:
        os.umask(0o077)
        dest = Path(td) / "x" / "Fab.pkl"
        dest.parent.mkdir(mode=0o700)
        handler = D.GuardedRedirects(D.ALLOWED_HOSTS, 3)

        def run(script, want_bytes=len(body), want_md5=md5):
            op = FakeOpener(script)
            try:
                res = D.download("https://borealisdata.ca/api/access/datafile/1", dest, want_bytes, want_md5, op,
                                 handler, sleep=sleeps.append)
                return res, None, op.calls
            except D.Stop as exc:
                return None, str(exc), op.calls

        part = dest.with_name(dest.name + ".part")
        for name, script, code in (
            ("checksum mismatch", [FakeResp(body[:-1] + b"\x00")], "MD5_MISMATCH"),
            ("oversize body", [FakeResp(body + b"x", clen=None)], "OVERSIZE"),
            ("short body", [FakeResp(body[:-10], clen=None)], "SIZE_MISMATCH"),
            ("HTML instead of data", [FakeResp(b"<html>login</html>", ctype="text/html; charset=utf-8")], "CONTENT_TYPE_REFUSED"),
            ("JSON error instead of data", [FakeResp(b"{}", ctype="application/json")], "CONTENT_TYPE_REFUSED"),
            ("Content-Length mismatch", [FakeResp(body, clen=len(body) + 1)], "CONTENT_LENGTH_MISMATCH"),
            ("final host not allowlisted", [FakeResp(body, url="https://cdn.example.net/f")], "FINAL_HOST_REFUSED"),
            ("final URL over http", [FakeResp(body, url="http://borealisdata.ca/f")], "FINAL_HOST_REFUSED"),
            ("login required (401)", [http_error(401)], "LOGIN_OR_PERMISSION_REQUIRED"),
            ("forbidden (403)", [http_error(403)], "LOGIN_OR_PERMISSION_REQUIRED"),
            ("not found (404)", [http_error(404)], "NOT_FOUND"),
            ("rate limited (429), no retry", [http_error(429), FakeResp(body)], "RATE_LIMITED"),
            ("network errors exhaust retries", [urllib.error.URLError("x")] * 3, "NETWORK_ERROR_AFTER_RETRIES"),
            ("server errors exhaust retries", [http_error(503)] * 3, "HTTP_ERROR_AFTER_RETRIES"),
        ):
            res, code_got, calls = run(script)
            ok(f"download stops: {name}", res is None and code_got == code and not dest.exists() and not part.exists())
        res, code_got, calls = run([http_error(429), FakeResp(body)])
        ok("rate limit makes exactly one call", calls == 1)
        sleeps.clear()
        res, code_got, calls = run([http_error(503), FakeResp(body)])
        ok("one server error, then success: retried once with a pause", res == (sha, 2) and len(sleeps) == 1)
        ok("downloaded file has mode 0600 and no .part remains",
           stat.S_IMODE(dest.stat().st_mode) == 0o600 and not part.exists() and dest.read_bytes() == body)
        before = dest.read_bytes()
        res, code_got, calls = run([FakeResp(body)])
        ok("never overwrites an existing destination", code_got == "DESTINATION_APPEARED" and dest.read_bytes() == before)

        # --- process(): reuse, mismatch, gating, decode verdicts --------------------------------------------
        saved = (G.ROOT_DIR, dict(G.ROOT_PINS))
        G.ROOT_DIR = Path(td) / "root"
        try:
            def pins(blob):
                G.ROOT_PINS["Fab"] = ("Fab.pkl", 1, len(blob), hashlib.md5(blob).hexdigest(), "v-test")

            def proc(blob_pinned, script, dataset_problems=(), api=None, preexisting=None):
                pins(blob_pinned)
                p = G.root_path("Fab")
                if p.parent.exists():
                    for f in p.parent.iterdir():
                        f.unlink()
                if preexisting is not None:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(preexisting)
                opener = FakeOpener(script)
                api_opener = FakeOpener(api if api is not None else [json_resp(file_doc("Fab"))])
                row, _ = D.process("Fab", opener, D.GuardedRedirects(D.ALLOWED_HOSTS, 3), api_opener,
                                   list(dataset_problems), sleep=sleeps.append)
                return row, opener.calls + api_opener.calls

            row, calls = proc(body, [FakeResp(body)])
            ok("process: fresh download, safe load and schema pass", row["stop"] is None and row["downloaded"]
               and row["sha256"] == sha and row["decode"]["loaders_agree"] and row["decode"]["strict_load_ok"])
            ok("process: directory 0700, file 0600", stat.S_IMODE(G.root_path("Fab").parent.stat().st_mode) == 0o700
               and stat.S_IMODE(G.root_path("Fab").stat().st_mode) == 0o600)
            row, calls = proc(body, [], preexisting=body)
            ok("process: verified local copy is reused with no network call", row["reused_local_copy"] and calls == 0
               and row["stop"] is None)
            row, calls = proc(body, [], preexisting=body[:-1] + b"\x01")
            ok("process: mismatching local copy stops and is left untouched",
               row["stop"] == "EXISTING_FILE_MISMATCH" and calls == 0 and G.root_path("Fab").read_bytes() == body[:-1] + b"\x01")
            row, calls = proc(body, [FakeResp(body)], dataset_problems=["GUESTBOOK_PRESENT"])
            ok("process: dataset gate stops before any file request", row["stop"] == "ACCESS_GATED_OR_RECORD_CHANGED" and calls == 0)
            row, calls = proc(body, [FakeResp(body)], api=[json_resp(file_doc("Fab", restricted=True))])
            ok("process: restricted file record stops before the download", row["stop"] == "PUBLISHER_RECORD_CHANGED"
               and "FILE_RESTRICTED" in row["publisher_file_problems"] and not G.root_path("Fab").exists())
            evil = T0.malicious()["global os.system"][0]
            G.ROOT_PINS["Fab"] = ("Fab.pkl", 1, len(evil), hashlib.md5(evil).hexdigest(), "v-test")
            row, calls = proc(evil, [FakeResp(evil)])
            ok("process: injected unsafe pickle with a MATCHING digest is still blocked",
               row["stop"] == "ROOT_SAFE_LOADER_BLOCKED" and row["integrity_ok"] and not row["decode"]["static_scan_ok"])
            sneaky = T0.malicious()["REDUCE numpy.ndarray class"][0]
            row, calls = proc(sneaky, [FakeResp(sneaky)])
            ok("process: allowlisted-global trick blocked by the strict layer",
               row["stop"] == "ROOT_SAFE_LOADER_BLOCKED" and row["decode"]["static_scan_ok"]
               and row["decode"]["strict_load_reason"] == "PICKLE_REDUCE_TARGET_REFUSED")
            listy = pickle.dumps([T0.fab_cohort(np.random.default_rng(5))], protocol=4)
            row, calls = proc(listy, [FakeResp(listy)])
            ok("process: wrong root schema is reported", row["stop"] == "ROOT_SCHEMA_UNEXPECTED")
            ok("overall status mapping", D.overall_status([{"stop": None}], []) == (D.PASS_STATUS, 0)
               and D.overall_status([{"stop": "ROOT_SAFE_LOADER_BLOCKED"}], [])[1] == 5
               and D.overall_status([{"stop": "ROOT_SCHEMA_UNEXPECTED"}], [])[1] == 7
               and D.overall_status([{"stop": "MD5_MISMATCH"}], [])[1] == 4
               and D.overall_status([{"stop": None}], ["GUESTBOOK_PRESENT"])[1] == 6)
        finally:
            G.ROOT_DIR = saved[0]
            G.ROOT_PINS.clear()
            G.ROOT_PINS.update(saved[1])

        # --- rest-stage prerequisites (portable form of the original stage gate) ------------------------------
        out = Path(td) / "results"
        out.mkdir()
        first, parity = out / RP.result_name("fetch-originals", "first"), out / RP.result_name("root-parity", "first")
        ok("rest stage refused without the first download", not RP.prerequisites_ok(out))
        first.write_text(json.dumps({"status": "ROOT_SAFE_LOADER_BLOCKED"}))
        parity.write_text(json.dumps({"status": "X"}))
        ok("rest stage refused when the first download did not pass", not RP.prerequisites_ok(out))
        first.write_text(json.dumps({"status": D.PASS_STATUS}))
        ok("rest stage allowed after a passed first download and a first parity aggregate", RP.prerequisites_ok(out))
        parity.unlink()
        ok("rest stage refused without the first parity aggregate", not RP.prerequisites_ok(out))
        parity.write_text("{not json")
        ok("rest stage refused when the first parity aggregate is unreadable", not RP.prerequisites_ok(out))

    # --- would-be aggregate --------------------------------------------------------------------------------------
    cs = G.structure_census(T0.fab_cohort(np.random.default_rng(2)))
    row = {"cohort": "3DGait", "logical_file": "3DGait.pkl", "publisher_datafile": 926879, "publisher_versions": "v1.0-v4.0",
           "bytes_pinned": 8066161, "md5_pinned": "a" * 32, "reused_local_copy": False, "downloaded": True,
           "download_attempts": 1, "redirect_host_classes": ["*.scholarsportal.info"], "integrity_ok": True,
           "sha256": "b" * 64, "stop": None,
           "decode": D.decode_checks(pickle.dumps(T0.fab_cohort(np.random.default_rng(6)), protocol=4))[0],
           "structure": D.census_out(cs)}
    out = {"card": "fetch-originals-first", "card_sha256": "0" * 64, "run_commit": "a" * 40, "seed": 1,
           "protocol_class": "DOWNLOAD", "status": D.PASS_STATUS, "labels_read": False, "authority": "x",
           "publisher": {"doi": G.DOI, "dataset_problems": [], "network_used": True,
                         "latest_version_licence_as_listed": "CC BY-NC-ND 4.0", "expected_latest_version": "4.0",
                         "access_endpoint": "x", "allowed_host_classes": ["borealisdata.ca"]},
           "files": [row, {"cohort": "BMCLab", "stop": "NOT_ATTEMPTED_AFTER_EARLIER_STOP"}],
           "local_storage": {"location": "x", "dir_mode": "0700", "file_mode": "0600"}, "compute": IC.Receipt().block()}
    text = IC.render(out)
    ok("would-be aggregate passes the renderer and the validator rules", IC.problems_of(json.loads(text)) == [])
    ok("aggregate carries no subject key, walk key, path or clinical value",
       all(s not in text for s in ("walk_0", "S00", str(Path.home()), "original/", '"on"', "\"medication\": \"")))

    # --- opcode-count hardening ------------------------------------------------------------------------------------
    ok("magnitude band", D.magnitude_band(9875) == "1e3_to_1e4" and D.magnitude_band(0) == "0" and D.magnitude_band(10) == "1e1_to_1e2")
    blob = pickle.dumps(T0.fab_cohort(np.random.default_rng(7)), protocol=4)
    v, census = D.hardened_decode_checks(blob)
    ok("hardened verdict: no exact opcode counts or total", "opcode_counts" not in v and "opcodes_total" not in v
       and v["opcode_names"] and v["opcodes_total_band"].startswith("1e"))
    ok("hardened verdict keeps every safety verdict", v["static_scan_ok"] and v["strict_load_ok"] and v["existing_load_ok"]
       and v["loaders_agree"] and census is not None)
    evil = T0.malicious()["global os.system"][0]
    v2, c2 = D.hardened_decode_checks(evil)
    ok("hardened verdict still blocks an unsafe pickle", not v2["static_scan_ok"] and c2 is None)
    text = json.dumps(v, sort_keys=True)
    ok("no integer next to an opcode name", all(f'"{n}": ' not in text for n in v["opcode_names"]))
    ok("hardening wraps the unchanged original decode checks", D._original_decode_checks is D.decode_checks
       and D.hardened_decode_checks(blob)[1] == D.decode_checks(blob)[1])


# ---- command line ---------------------------------------------------------------------------------------------
def cli(*args):
    return subprocess.run([sys.executable, "-I", "-B", str(SCRIPTS / "run_provenance.py"), *args],
                          capture_output=True, text=True, cwd=REPO)


def test_cli():
    ok("CLI --help runs from a fresh checkout", cli("--help").returncode == 0)
    ok("CLI refuses to guess a data location", cli().returncode == 2 and cli("root-speed").returncode == 2)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        base = ["--data-dir", str(d / "data"), "--out-dir", str(d / "out")]
        ok("fetch-originals needs a stage", cli("fetch-originals", *base, "--accept-care-pd-terms").returncode == 2)
        ok("stages apply only to staged commands", cli("root-speed", "--first", *base).returncode == 2)
        done = cli("fetch-originals", "--first", *base)
        ok("no terms flag: nothing downloaded, nothing written", done.returncode == 2 and "TERMS_NOT_ACCEPTED" in done.stdout
           and not (d / "data").exists() and not (d / "out").exists())
        done = cli("fetch-originals", "--first", "--data-dir", str(REPO / "inside"), "--out-dir", str(d / "out"),
                   "--accept-care-pd-terms")
        ok("a data folder inside the code repository is refused", done.returncode == 20
           and "OUTSIDE_CODE_REPOSITORY" in done.stdout and not (REPO / "inside").exists())

    saved = FX.snapshot()
    saved_make, saved_decode = D.make_opener, D.decode_checks
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            data, out = d / "data", d / "out"
            argv = ["fetch-originals", "--first", "--data-dir", str(data), "--out-dir", str(out), "--accept-care-pd-terms"]
            body = pickle.dumps(T0.fab_cohort(np.random.default_rng(8)), protocol=4)
            G.ROOT_PINS["3DGait"] = ("3DGait.pkl", 926879, len(body), hashlib.md5(body).hexdigest(), "v1.0-v4.0")
            calls = []

            def make(allowed, limit):
                calls.append((tuple(allowed), limit))
                if limit == 0:
                    return FakeOpener([json_resp(dataset_doc()), json_resp(file_doc("3DGait"))]), None
                return FakeOpener([FakeResp(body)]), D.GuardedRedirects(allowed, limit)

            D.make_opener = make
            code, text = FX.quiet_main(RP, argv)
            res = out / RP.result_name("fetch-originals", "first")
            agg = json.loads(res.read_text()) if res.is_file() else {}
            row = (agg.get("files") or [{}])[0]
            ok("fetch-originals --first: fabricated download passes and writes one aggregate",
               code == 0 and agg.get("status") == D.PASS_STATUS and agg.get("card") == "fetch-originals-first"
               and row.get("sha256") == hashlib.sha256(body).hexdigest() and row.get("downloaded") is True)
            ok("fetch-originals: the downloaded file sits in the portable layout with mode 0600",
               (data / "original" / "3DGait" / "3DGait.pkl").read_bytes() == body
               and stat.S_IMODE((data / "original" / "3DGait" / "3DGait.pkl").stat().st_mode) == 0o600)
            ok("fetch-originals: opcode counts are banded (hardened verdict) and labels are not read",
               "opcode_counts" not in row.get("decode", {}) and "opcode_names" in row.get("decode", {})
               and agg.get("labels_read") is False and agg.get("release_scope") == RP.SCOPE)
            ok("fetch-originals: only the pinned hosts are allowlisted",
               calls == [(D.ALLOWED_HOSTS, D.MAX_REDIRECTS), (D.API_HOSTS, 0)])
            ok("fetch-originals: aggregate holds no local path", str(d) not in res.read_text())
            code, text = FX.quiet_main(RP, argv)
            ok("an existing result is never overwritten", code == 10 and "RESULT_EXISTS" in text)
            code, text = FX.quiet_main(RP, ["fetch-originals", "--rest", *argv[2:]])
            ok("fetch-originals --rest stops before any request without the first parity aggregate",
               code == 20 and not (out / RP.result_name("fetch-originals", "rest")).exists())
    finally:
        D.make_opener, D.decode_checks = saved_make, saved_decode
        FX.restore(saved)


def main():
    IC.block_network()                      # empty allowlist: any real network event in these tests raises
    test_kernels()
    test_cli()
    print(f"ROOT_FETCH_TESTS_PASS n={PASS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
