"""Portable analysis kernel. See docs/provenance_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
import aggregate_core as IC
import root_core as G
import common as C
import math
PASS_STATUS = 'G_ROOT_DOWNLOAD_VERIFIED_SAFE_LOAD_PASS'
API_FILE = 'https://borealisdata.ca/api/files/{fid}'
API_DATASET = 'https://borealisdata.ca/api/datasets/:persistentId/?persistentId=doi:' + G.DOI
ACCESS = 'https://borealisdata.ca/api/access/datafile/{fid}'
ALLOWED_HOSTS = ('borealisdata.ca', '.scholarsportal.info')
API_HOSTS = ('borealisdata.ca',)
EXPECTED_VERSION = (4, 0)
MAX_REDIRECTS = 3
ATTEMPTS = 3
PAUSE_S = (20, 40)
CHUNK = 1 << 20
API_CAP = 1 << 20
TIMEOUT = 120
REFUSED_CONTENT = ('text/', 'application/json', 'application/xml', 'application/xhtml')
USER_AGENT = 'research-integrity-fetch/1.0'

class Stop(Exception):
    pass

def host_class(host):
    host = (host or '').lower().rstrip('.')
    if host == 'borealisdata.ca':
        return 'borealisdata.ca'
    if host.endswith('.scholarsportal.info'):
        return '*.scholarsportal.info'
    return 'other'

class GuardedRedirects(urllib.request.HTTPRedirectHandler):

    def __init__(self, allowed, limit):
        super().__init__()
        self.allowed, self.limit, self.hops = (tuple(allowed), int(limit), [])

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlsplit(newurl)
        if parts.scheme != 'https' or not IC.host_allowed(parts.hostname, self.allowed):
            raise Stop('REDIRECT_HOST_REFUSED')
        self.hops.append(host_class(parts.hostname))
        if len(self.hops) > self.limit:
            raise Stop('TOO_MANY_REDIRECTS')
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def make_opener(allowed, limit):
    handler = GuardedRedirects(allowed, limit)
    return (urllib.request.build_opener(handler), handler)

def _content_refused(ctype):
    ctype = (ctype or '').lower().strip()
    return any((ctype.startswith(p) for p in REFUSED_CONTENT))

def _http_stop(exc):
    code = getattr(exc, 'code', None)
    if code in (401, 403):
        return Stop('LOGIN_OR_PERMISSION_REQUIRED')
    if code == 404:
        return Stop('NOT_FOUND')
    if code == 429:
        return Stop('RATE_LIMITED')
    return None

def fetch_json(url, opener):
    if urlsplit(url).scheme != 'https':
        raise Stop('NON_HTTPS_URL')
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'})
    try:
        with opener.open(req, timeout=TIMEOUT) as resp:
            if not (resp.headers.get('Content-Type') or '').lower().startswith('application/json'):
                raise Stop('API_NOT_JSON')
            body = resp.read(API_CAP + 1)
    except Stop:
        raise
    except urllib.error.HTTPError as exc:
        raise _http_stop(exc) or Stop('API_HTTP_ERROR') from None
    except Exception:
        raise Stop('API_UNREACHABLE') from None
    if len(body) > API_CAP:
        raise Stop('API_TOO_LARGE')
    try:
        return json.loads(body.decode('utf-8'))
    except Exception:
        raise Stop('API_NOT_JSON') from None

def check_dataset_meta(js):
    problems = []
    data = js.get('data') if isinstance(js, dict) and js.get('status') == 'OK' else None
    if not isinstance(data, dict):
        return (['DATASET_RECORD_UNREADABLE'], None)
    if data.get('guestbookId') is not None:
        problems.append('GUESTBOOK_PRESENT')
    latest = data.get('latestVersion') if isinstance(data.get('latestVersion'), dict) else {}
    if (latest.get('versionNumber'), latest.get('versionMinorNumber')) != EXPECTED_VERSION:
        problems.append('LATEST_VERSION_CHANGED')
    if latest.get('versionState') != 'RELEASED':
        problems.append('VERSION_NOT_RELEASED')
    if latest.get('termsOfAccess'):
        problems.append('TERMS_OF_ACCESS_PRESENT')
    if latest.get('fileAccessRequest'):
        problems.append('FILE_ACCESS_REQUEST_ENABLED')
    lic = latest.get('license')
    name = lic.get('name') if isinstance(lic, dict) else None
    return (problems, str(name)[:60] if name else None)

def check_file_meta(js, cohort):
    logical, fid, size, md5 = G.ROOT_PINS[cohort][:4]
    data = js.get('data') if isinstance(js, dict) and js.get('status') == 'OK' else None
    if not isinstance(data, dict) or not isinstance(data.get('dataFile'), dict):
        return ['FILE_RECORD_UNREADABLE']
    df = data['dataFile']
    problems = []
    if data.get('restricted') is not False or df.get('restricted') is True:
        problems.append('FILE_RESTRICTED')
    if df.get('fileAccessRequest'):
        problems.append('FILE_ACCESS_REQUEST_ENABLED')
    if df.get('tabularData'):
        problems.append('FILE_TABULAR')
    if df.get('id') != fid or df.get('filename') != logical or data.get('label') != logical:
        problems.append('FILE_IDENTITY_CHANGED')
    if df.get('filesize') != size or df.get('md5') != md5:
        problems.append('FILE_DIGEST_CHANGED')
    return problems

def download(url, dest, want_bytes, want_md5, opener, handler, sleep=time.sleep):
    part = dest.with_name(dest.name + '.part')
    attempts = 0
    while True:
        attempts += 1
        handler.hops = []
        if part.exists():
            part.unlink()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with opener.open(req, timeout=TIMEOUT) as resp:
                final = urlsplit(resp.geturl())
                if final.scheme != 'https' or not IC.host_allowed(final.hostname, ALLOWED_HOSTS):
                    raise Stop('FINAL_HOST_REFUSED')
                if getattr(resp, 'status', 200) != 200:
                    raise Stop('UNEXPECTED_STATUS')
                if _content_refused(resp.headers.get('Content-Type')):
                    raise Stop('CONTENT_TYPE_REFUSED')
                clen = resp.headers.get('Content-Length')
                if clen is not None and str(clen).strip() != str(want_bytes):
                    raise Stop('CONTENT_LENGTH_MISMATCH')
                md5, sha, got = (hashlib.md5(), hashlib.sha256(), 0)
                fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 384)
                with os.fdopen(fd, 'wb') as out:
                    for block in iter(lambda: resp.read(CHUNK), b''):
                        got += len(block)
                        if got > want_bytes:
                            raise Stop('OVERSIZE')
                        md5.update(block)
                        sha.update(block)
                        out.write(block)
            if got != want_bytes:
                raise Stop('SIZE_MISMATCH')
            if md5.hexdigest() != want_md5:
                raise Stop('MD5_MISMATCH')
            os.chmod(part, 384)
            os.link(part, dest)
            part.unlink()
            return (sha.hexdigest(), attempts)
        except Stop:
            if part.exists():
                part.unlink()
            raise
        except urllib.error.HTTPError as exc:
            if part.exists():
                part.unlink()
            stop = _http_stop(exc)
            if stop is not None:
                raise stop from None
            if attempts >= ATTEMPTS:
                raise Stop('HTTP_ERROR_AFTER_RETRIES') from None
        except FileExistsError:
            if part.exists():
                part.unlink()
            raise Stop('DESTINATION_APPEARED') from None
        except PermissionError:
            if part.exists():
                part.unlink()
            raise Stop('NETWORK_REFUSED_BY_TRIPWIRE') from None
        except Exception:
            if part.exists():
                part.unlink()
            if attempts >= ATTEMPTS:
                raise Stop('NETWORK_ERROR_AFTER_RETRIES') from None
        print(f'PROGRESS retry {attempts}')
        sleep(PAUSE_S[min(attempts, len(PAUSE_S)) - 1])

def decode_checks(data):
    head = G.format_sniff(data[:8])
    scan = G.static_scan(data)
    v = {'format': head, 'static_scan_ok': scan['ok'], 'static_scan_reason': scan['reason'], 'pickle_protocol': scan['protocol'], 'allowlisted_globals_seen': scan['globals_seen'], 'opcode_counts': scan['opcode_counts'], 'opcodes_total': scan['n_opcodes'], 'trailing_bytes_after_stop': scan['trailing_bytes'], 'strict_refused_opcodes': scan['strict_refused_opcodes'], 'strict_load_ok': False, 'strict_load_reason': None, 'existing_load_ok': False, 'existing_load_reason': None, 'loaders_agree': False}
    if not scan['ok']:
        return (v, None)
    try:
        obj = G.strict_loads(data)
        v['strict_load_ok'] = True
    except G.SourceError as exc:
        v['strict_load_reason'] = G.sanitize_reason(exc)
        return (v, None)
    try:
        obj2 = C.safe_loads(data)
        v['existing_load_ok'] = True
    except G.SourceError as exc:
        v['existing_load_reason'] = G.sanitize_reason(exc)
        return (v, None)
    v['loaders_agree'] = bool(G.trees_equal(obj, obj2))
    del obj2
    return (v, G.structure_census(obj))

def census_out(cs):
    if cs is None:
        return None
    if not cs['top_level_dict']:
        return {'top_level_dict': False}
    walks = cs['walks']
    q = G.quantiles(cs['frames'], min_n=IC.SMALL_CELL, digits=1)
    fps_mode = max(cs['fps_values'].items(), key=lambda kv: kv[1])[0] if cs['fps_values'] else None
    names = cs['unknown_field_names']
    shown = sorted((k for k, n in names.items() if k != '<non-identifier>' and n >= IC.SMALL_CELL))[:8]
    return {'top_level_dict': True, 'subjects_band': G.band5(cs['subjects']), 'walks_band': G.band5(walks), 'subject_key_types': {k: G.coarse_cell(n) for k, n in cs['subject_key_types'].items()}, 'n_subject_key_str_collisions': IC.count(cs['subject_key_str_collisions']), 'n_subjects_not_dict': IC.count(cs['subjects_not_dict']), 'all_walk_keys_str': cs['walk_keys_str'] == walks, 'all_records_dict': cs['records_dict'] == walks, 'field_present_in_all_records': {f: n == walks for f, n in cs['field_present'].items()}, 'field_absent_records': {f: G.coarse_cell(walks - n) for f, n in cs['field_present'].items()}, 'n_records_with_unknown_fields': IC.count(cs['records_with_unknown_fields']), 'unknown_field_names_shown': shown, 'n_unknown_field_names': IC.count(len(names)), 'all_pose_T_by_72_float': cs['pose_conforming'] == walks, 'all_trans_T_by_3_float': cs['trans_conforming'] == walks, 'all_pose_trans_same_frames': cs['pose_trans_same_frames'] == walks, 'n_records_nonconforming_motion': IC.count(walks - min(cs['pose_conforming'], cs['trans_conforming'], cs['pose_trans_same_frames'])), 'beta_shape': {k: G.coarse_cell(n) for k, n in cs['beta_shape'].items()}, 'all_beta_zero': cs['beta_all_zero'] == walks, 'n_records_beta_nonzero': IC.count(walks - cs['beta_all_zero']), 'all_fps_numeric': cs['fps_numeric'] == walks, 'fps_mode': fps_mode, 'fps_single_value': len(cs['fps_values']) == 1, 'frames_quartiles': q, 'array_dtypes': cs['dtypes'][:6]}

def process(cohort, opener, handler, api_opener, dataset_problems, sleep=time.sleep):
    logical, fid, size, md5, versions = G.ROOT_PINS[cohort]
    if fid in G.NOT_APPROVED_DATAFILES:
        raise Stop('DATAFILE_NOT_APPROVED')
    dest = G.root_path(cohort)
    row = {'cohort': cohort, 'logical_file': logical, 'publisher_datafile': fid, 'publisher_versions': versions, 'bytes_pinned': size, 'md5_pinned': md5, 'reused_local_copy': False, 'downloaded': False, 'download_attempts': 0, 'redirect_host_classes': [], 'integrity_ok': False, 'sha256': 'ABSENT', 'stop': None}
    try:
        dest.parent.mkdir(parents=True, exist_ok=True, mode=448)
        os.chmod(dest.parent, 448)
        if dest.exists():
            got_size, got_md5, got_sha = G.file_digests(dest)
            if got_size != size or got_md5 != md5:
                raise Stop('EXISTING_FILE_MISMATCH')
            row.update(reused_local_copy=True, integrity_ok=True, sha256=got_sha)
        else:
            if dataset_problems:
                raise Stop('ACCESS_GATED_OR_RECORD_CHANGED')
            problems = check_file_meta(fetch_json(API_FILE.format(fid=fid), api_opener), cohort)
            if problems:
                row['publisher_file_problems'] = problems
                raise Stop('PUBLISHER_RECORD_CHANGED')
            sha, attempts = download(ACCESS.format(fid=fid), dest, size, md5, opener, handler, sleep)
            row.update(downloaded=True, download_attempts=attempts, integrity_ok=True, sha256=sha, redirect_host_classes=sorted(set(handler.hops)))
        os.chmod(dest, 384)
    except Stop as exc:
        row['stop'] = str(exc)
        return (row, None)
    except Exception as exc:
        row['stop'] = G.sanitize_reason('UNEXPECTED_' + type(exc).__name__)
        return (row, None)
    data = dest.read_bytes()
    if len(data) != size or hashlib.md5(data).hexdigest() != md5:
        row.update(integrity_ok=False, stop='REREAD_MISMATCH')
        return (row, None)
    verdict, census = decode_checks(data)
    del data
    row['decode'] = verdict
    row['structure'] = census_out(census)
    ok = verdict['static_scan_ok'] and verdict['strict_load_ok'] and verdict['existing_load_ok'] and verdict['loaders_agree']
    if not ok:
        row['stop'] = 'ROOT_SAFE_LOADER_BLOCKED'
    elif not schema_ok(census):
        row['stop'] = 'ROOT_SCHEMA_UNEXPECTED'
    return (row, census)

def schema_ok(cs):
    if cs is None or not cs.get('top_level_dict'):
        return False
    w = cs['walks']
    return bool(w > 0 and cs['subjects_not_dict'] == 0 and (cs['records_dict'] == w) and (cs['walk_keys_str'] == w) and (cs['subject_key_types']['other'] == 0) and (cs['subject_key_str_collisions'] == 0) and all((cs['field_present'][f] == w for f in ('pose', 'trans', 'fps'))) and (cs['pose_conforming'] == w) and (cs['trans_conforming'] == w) and (cs['pose_trans_same_frames'] == w) and (cs['fps_numeric'] == w))

def overall_status(rows, dataset_problems):
    if dataset_problems:
        return ('ACCESS_GATED_OR_RECORD_CHANGED', 6)
    for r in rows:
        if r['stop'] == 'ROOT_SAFE_LOADER_BLOCKED':
            return ('ROOT_SAFE_LOADER_BLOCKED', 5)
        if r['stop'] == 'ROOT_SCHEMA_UNEXPECTED':
            return ('G_ROOT_SCHEMA_UNEXPECTED', 7)
        if r['stop'] is not None:
            return ('G_ROOT_DOWNLOAD_FAILED', 4)
    return (PASS_STATUS, 0)
_original_decode_checks = decode_checks

def magnitude_band(n):
    if not n:
        return '0'
    e = int(math.floor(math.log10(n)))
    return f'1e{e}_to_1e{e + 1}'

def harden(verdict):
    v = dict(verdict)
    counts = v.pop('opcode_counts', {}) or {}
    total = v.pop('opcodes_total', 0) or 0
    v['opcode_names'] = sorted(counts)
    v['opcodes_total_band'] = magnitude_band(total)
    return v

def hardened_decode_checks(data):
    verdict, census = _original_decode_checks(data)
    return (harden(verdict), census)
