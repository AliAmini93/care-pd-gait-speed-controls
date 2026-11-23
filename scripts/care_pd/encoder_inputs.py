"""Portable analysis kernel. See docs/provenance_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import hashlib
import sys
import time
import urllib.request
from pathlib import Path
import aggregate_core as IC
COHORTS = ('3DGait', 'BMCLab', 'PD-GaM', 'T-SDU-PD')
DV_VERSION = '1.3'
DV_ACCESS = 'https://borealisdata.ca/api/access/datafile/{fid}'
DV_FILES = {'3DGait': (926934, 56163606, 'c674cb928691dd45d15ac764d2ae0bff'), 'BMCLab': (926933, 875604234, '5355a38b3e3162a66693ca0472bf0340'), 'PD-GaM': (926929, 587333598, 'f21fd10ac2252c232d06839dcca8ec98'), 'T-SDU-PD': (926928, 187782986, '19d11a5217a7496fcc04c944cb4a9842')}
GH_COMMIT = '9bc22b0258b15369cd8b62f2de20c70b3754349f'
GH_RAW = 'https://raw.githubusercontent.com/TaatiTeam/CARE-PD/' + GH_COMMIT + '/{path}'
CKPT = 'pretext/momask/checkpoints/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/model/net_best_fid.tar'
STATS = 'assets/stats/HumanML3D_norm_data'
GH_FILES = {CKPT: (84070519, '03852ca17a505e05cb6330c204f96b3e3236f63d'), f'{STATS}/3DGait/Mean.npy': (1180, 'e668b0e2d04efcb959521d51d704d75cf737bd90'), f'{STATS}/3DGait/Std_FEAT_BIAS_included.npy': (1180, '1478ff5516e97069f6107ce37bf275e747d04608'), f'{STATS}/BMCLab/Mean.npy': (1180, '68682ffc9ec1625e35173a1c10a6fa38c8e8ba87'), f'{STATS}/BMCLab/Std_FEAT_BIAS_included.npy': (1180, 'f81df9029f711de8c7497273ccd36308c6567ca3'), f'{STATS}/PD-GaM/Mean.npy': (1180, 'f33dc4f7714ff1b049fefc996a096fe6074a2a8c'), f'{STATS}/PD-GaM/Std_FEAT_BIAS_included.npy': (1180, '659101c4f03ff63a7dd188794b916c697541b6c8'), f'{STATS}/T-SDU-PD/Mean.npy': (1180, 'e81b73369a146cabaafcffef93302f70259aeea7'), f'{STATS}/T-SDU-PD/Std_FEAT_BIAS_included.npy': (1180, 'e8c5752952225897716572724c2f4ffc818f6da6'), f'{STATS}/LODO_3DGait/Mean.npy': (1180, '70887d8fa168eb3e4da6edbd3eb63161723d7e0b'), f'{STATS}/LODO_3DGait/Std_FEAT_BIAS_included.npy': (1180, '378902769eefe43d11845c8bb2b37d756c1a41a8'), f'{STATS}/LODO_BMCLab/Mean.npy': (1180, '2d2c3e375ffec56311baca9495e0845eb80ae6bb'), f'{STATS}/LODO_BMCLab/Std_FEAT_BIAS_included.npy': (1180, 'adb0b4a19b6d7598d0dca74791c28296c3dbcef1'), f'{STATS}/LODO_PD-GaM/Mean.npy': (1180, 'ad2fc6289124677be06e3f8f1c19c0513f05393d'), f'{STATS}/LODO_PD-GaM/Std_FEAT_BIAS_included.npy': (1180, '68b6147b9870922b0a9f9d9cb788cb5b3126678c'), f'{STATS}/LODO_T-SDU-PD/Mean.npy': (1180, 'da7a34d6a4cd0ae44db5368d13c70010d69ba89e'), f'{STATS}/LODO_T-SDU-PD/Std_FEAT_BIAS_included.npy': (1180, '3aba88703aa4e3eea40ca3454d3456cd01c7fccb')}
ALLOWED_HOSTS = ('borealisdata.ca', '.scholarsportal.info', 'raw.githubusercontent.com')
DV_DIR = None
GH_DIR = None
CHUNK = 1 << 20
RETRIES = 3

def dv_path(cohort):
    return DV_DIR / cohort / 'HumanML3D_collected.npz'

def gh_path(repo_path):
    return GH_DIR / repo_path

def digests(path):
    size = path.stat().st_size
    md5, sha256, blob = (hashlib.md5(), hashlib.sha256(), hashlib.sha1())
    blob.update(b'blob %d\x00' % size)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            md5.update(chunk)
            sha256.update(chunk)
            blob.update(chunk)
    return (size, md5.hexdigest(), sha256.hexdigest(), blob.hexdigest())

def verified(path, want_bytes, want_md5=None, want_blob=None):
    if not path.is_file():
        return (False, None)
    size, md5, sha256, blob = digests(path)
    good = size == want_bytes and (want_md5 is None or md5 == want_md5) and (want_blob is None or blob == want_blob)
    return (good, sha256)

def download(url, dest, want_bytes):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    req = urllib.request.Request(url, headers={'User-Agent': 'research-integrity-fetch/1.0'})
    with urllib.request.urlopen(req, timeout=120) as resp, open(part, 'wb') as out:
        got = 0
        for chunk in iter(lambda: resp.read(CHUNK), b''):
            out.write(chunk)
            got += len(chunk)
            if got > want_bytes:
                break
    part.replace(dest)

def fetch(url, dest, want_bytes, want_md5=None, want_blob=None):
    good, sha = verified(dest, want_bytes, want_md5, want_blob)
    attempts = 0
    while not good and attempts < RETRIES:
        attempts += 1
        try:
            download(url, dest, want_bytes)
        except PermissionError:
            raise
        except Exception as exc:
            print(f'PROGRESS retry {attempts} after {type(exc).__name__}')
            time.sleep(5 * attempts)
            continue
        good, sha = verified(dest, want_bytes, want_md5, want_blob)
    if not good and dest.exists():
        dest.unlink()
    return (good, sha, attempts)

def plan():
    rows = []
    for cohort in COHORTS:
        fid, size, md5 = DV_FILES[cohort]
        rows.append({'label': f'dataverse_v{DV_VERSION}/HumanML3D/{cohort}', 'url': DV_ACCESS.format(fid=fid), 'dest': dv_path(cohort), 'bytes': size, 'md5': md5, 'blob': None})
    for repo_path, (size, blob) in GH_FILES.items():
        rows.append({'label': 'github_9bc22b0/' + repo_path.replace(STATS, 'norm').replace(CKPT, 'momask_rvq_checkpoint'), 'url': GH_RAW.format(path=repo_path), 'dest': gh_path(repo_path), 'bytes': size, 'md5': None, 'blob': blob})
    return rows
