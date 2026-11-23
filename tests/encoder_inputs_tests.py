"""Fabricated-only tests of the pinned encoder-input download and the fetch-encoder-inputs command.

No network: urllib.request.urlopen is replaced by a local stub that serves fabricated bytes. No data file.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "care_pd"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_core as IC  # noqa: E402
import encoder_inputs as F  # noqa: E402
import provenance_fixtures as FX  # noqa: E402
import run_provenance as RP  # noqa: E402

PASS = 0


def ok(name, cond):
    global PASS
    if not cond:
        print(f"FAIL {name}")
        sys.exit(1)
    PASS += 1
    print(f"PASS {name}")


def test_kernels():
    with tempfile.TemporaryDirectory() as tmp:
        saved = (F.DV_DIR, F.GH_DIR)
        F.DV_DIR, F.GH_DIR = Path(tmp) / "humanml3d_v1.3", Path(tmp) / "care_pd_9bc22b0"
        try:
            rows = F.plan()
            ok("plan lists 4 dataverse files and 17 repository files", len(rows) == 21)
            ok("every destination is under the configured data folder", all(str(r["dest"]).startswith(tmp) for r in rows))
        finally:
            F.DV_DIR, F.GH_DIR = saved
        ok("every url is https on an allowed host",
           all(r["url"].startswith("https://") and IC.host_allowed(r["url"].split("/")[2], F.ALLOWED_HOSTS) for r in rows))
        ok("dataverse rows pin md5, repository rows pin blob sha1",
           all((r["md5"] is None) != (r["blob"] is None) for r in rows))
        ok("pins are well-formed", all(len(v[2]) == 32 for v in F.DV_FILES.values())
           and all(len(v[1]) == 40 for v in F.GH_FILES.values()))
        ok("labels are short and unique", len({r["label"] for r in rows}) == 21 and all(len(r["label"]) < 120 for r in rows))
        ok("one checkpoint and sixteen statistics files",
           sum("checkpoint" in r["label"] for r in rows) == 1 and sum("/norm/" in r["label"] for r in rows) == 16)
        ok("repository files come from the pinned commit", F.GH_COMMIT == "9bc22b0258b15369cd8b62f2de20c70b3754349f"
           and all(F.GH_COMMIT in r["url"] for r in rows if r["blob"]))
        p = Path(tmp) / "x.bin"
        data = bytes(range(256)) * 37
        p.write_bytes(data)
        size, md5, sha256, blob = F.digests(p)
        ok("digests: size, md5 and sha256", size == len(data) and md5 == hashlib.md5(data).hexdigest()
           and sha256 == hashlib.sha256(data).hexdigest())
        git = subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True).stdout.strip()
        ok("digests: git blob sha1 equals git hash-object", blob == git)
        ok("verified accepts the right pins", F.verified(p, len(data), md5, None)[0] and F.verified(p, len(data), None, blob)[0])
        ok("verified refuses a wrong md5", not F.verified(p, len(data), "0" * 32, None)[0])
        ok("verified refuses a wrong size", not F.verified(p, len(data) + 1, md5, None)[0])
        ok("verified refuses a wrong blob", not F.verified(p, len(data), None, "0" * 40)[0])
        ok("verified refuses a missing file", F.verified(Path(tmp) / "absent", 1, None, None) == (False, None))
        # a present, already-verified file is not downloaded again (no network is touched here)
        good, sha, attempts = F.fetch("https://borealisdata.ca/never-called", p, len(data), md5, None)
        ok("fetch skips a verified file", good and attempts == 0 and sha == sha256)
    # the aggregate this command writes must pass the renderer (keys, counts, size)
    files = [{"label": r["label"], "bytes_pinned": r["bytes"], "integrity_ok": True,
              "pinned_digest_kind": "md5" if r["md5"] else "git_blob_sha1", "sha256": "a" * 64, "download_attempts": 1}
             for r in rows]
    out = {"card": "fetch-encoder-inputs", "card_sha256": "0" * 64, "run_commit": "a" * 40, "protocol_class": "DOWNLOAD",
           "seed": 1, "status": "C1_FETCH_COMPLETE", "sources": {"dataverse_doi": "10.5683/SP3/TWIKMK"},
           "files_part1": files[:10], "files_part2": files[10:], "n_files": IC.count(21), "n_files_verified": IC.count(21),
           "bytes_verified": 1, "content_parsed": False, "compute": IC.Receipt().block()}
    ok("would-be aggregate passes the renderer", len(IC.render(out)) < IC.MAX_BYTES)
    ok("portable module defines no private location or route", F.DV_DIR is None and F.GH_DIR is None
       and not hasattr(F, "main"))


class Served:
    def __init__(self, body):
        self.body = io.BytesIO(body)

    def read(self, n=-1):
        return self.body.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_cli():
    ok("terms flag is required before any request", RP.parse_args(["fetch-encoder-inputs", "--data-dir", "/x", "--out-dir", "/y"])
       .accept_care_pd_terms is False)
    saved = FX.snapshot()
    real_urlopen, real_sleep = urllib.request.urlopen, time.sleep
    served, requested = {}, []

    def stub(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        requested.append(url)
        if url not in served:
            raise OSError("not served")
        return Served(served[url])

    try:
        urllib.request.urlopen = stub
        time.sleep = lambda s: None
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            data, out = d / "data", d / "out"
            argv = ["fetch-encoder-inputs", "--data-dir", str(data), "--out-dir", str(out)]
            code, text = FX.quiet_main(RP, argv)
            ok("no terms flag: nothing requested, nothing written", code == 2 and "TERMS_NOT_ACCEPTED" in text
               and not requested and not data.exists() and not out.exists())
            for i, cohort in enumerate(F.COHORTS):
                body = f"fabricated humanml3d {cohort} {i}".encode() * 50
                F.DV_FILES[cohort] = (F.DV_FILES[cohort][0], len(body), hashlib.md5(body).hexdigest())
                served[F.DV_ACCESS.format(fid=F.DV_FILES[cohort][0])] = body
            for repo_path in list(F.GH_FILES):
                body = ("fabricated " + repo_path).encode() * 20
                blob = hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()
                F.GH_FILES[repo_path] = (len(body), blob)
                served[F.GH_RAW.format(path=repo_path)] = body
            code, text = FX.quiet_main(RP, argv + ["--accept-care-pd-terms"])
            res = out / RP.result_name("fetch-encoder-inputs")
            agg = json.loads(res.read_text()) if res.is_file() else {}
            ok("fabricated download of 21 pinned files completes", code == 0 and agg.get("status") == "C1_FETCH_COMPLETE"
               and agg.get("n_files") == 21 and agg.get("n_files_verified") == 21 and agg.get("content_parsed") is False)
            ok("files land in the portable layout and match their pins",
               all(F.verified(r["dest"], r["bytes"], r["md5"], r["blob"])[0] for r in F.plan())
               and str(F.plan()[0]["dest"]).startswith(str(data / "humanml3d_v1.3")))
            ok("only the 21 pinned URLs were requested, each once", sorted(requested) == sorted(served))
            ok("aggregate holds labels and digests only, no local path", str(d) not in res.read_text()
               and agg.get("card") == "fetch-encoder-inputs")
            requested.clear()
            code, text = FX.quiet_main(RP, argv + ["--accept-care-pd-terms"])
            ok("an existing result is never overwritten and nothing is requested", code == 10 and not requested)
            out2 = d / "out2"
            bad = F.plan()[0]
            Path(bad["dest"]).unlink()
            served[bad["url"]] = served[bad["url"]][:-1] + b"!"
            code, text = FX.quiet_main(RP, ["fetch-encoder-inputs", "--data-dir", str(data), "--out-dir", str(out2),
                                            "--accept-care-pd-terms"])
            agg = json.loads((out2 / RP.result_name("fetch-encoder-inputs")).read_text())
            ok("a corrupted download is retried, removed and reported incomplete with exit 20",
               code == 20 and agg["status"] == "C1_FETCH_INCOMPLETE" and not Path(bad["dest"]).exists()
               and requested.count(bad["url"]) == F.RETRIES and len(requested) == F.RETRIES)
    finally:
        urllib.request.urlopen, time.sleep = real_urlopen, real_sleep
        FX.restore(saved)


def main():
    IC.block_network()                      # empty allowlist: any real network event in these tests raises
    test_kernels()
    test_cli()
    print(f"ENCODER_INPUTS_TESTS_PASS n={PASS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
