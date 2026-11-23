"""Portable analysis kernel. See docs/provenance_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import json
import os
import resource
import sys
import time
from pathlib import Path
MAX_BYTES = 262144
MAX_LIST = 64
MAX_DICT = 64
MAX_STR = 240
SMALL_CELL = 5
SUPPRESSED = 'LT5'
CLASSES = ('METADATA', 'DOWNLOAD', 'SCORED', 'EXPLORATORY')
FORBIDDEN_KEY_PARTS = ('_id', 'ids', 'per_walk', 'per_subject', 'per_participant', 'per_person', 'embedding', 'state_dict', 'logit', 'proba_rows', 'prediction_rows', 'label_rows', 'password', 'secret', 'token', 'name_list', 'key_list')
REQUIRED_TOP = ('card', 'card_sha256', 'run_commit', 'protocol_class', 'status', 'compute')

class RouteError(SystemExit):
    pass

def stop(reason, code=20):
    print(f'INDEP_STOP: {reason}')
    raise RouteError(code)
_NET_EVENTS = ('socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr', 'socket.sendto', 'socket.sendmsg', 'socket.bind', 'urllib.Request')

def host_allowed(host, allowed):
    host = (host or '').lower().rstrip('.')
    for a in allowed:
        if a.startswith('.'):
            if host.endswith(a) and len(host) > len(a):
                return True
        elif host == a:
            return True
    return False

def block_network(allowed_hosts=()):
    allowed = tuple((a.lower() for a in allowed_hosts))

    def hook(event, args):
        if event not in _NET_EVENTS:
            return
        if not allowed:
            raise PermissionError('INDEP_NETWORK_REFUSED')
        if event == 'urllib.Request':
            from urllib.parse import urlsplit
            parts = urlsplit(str(args[0]))
            if parts.scheme == 'https' and host_allowed(parts.hostname, allowed):
                return
            raise PermissionError('INDEP_NETWORK_REFUSED')
        if event == 'socket.getaddrinfo':
            if args and isinstance(args[0], (str, bytes)):
                host = args[0].decode() if isinstance(args[0], bytes) else args[0]
                if host_allowed(host, allowed):
                    return
            raise PermissionError('INDEP_NETWORK_REFUSED')
        if event == 'socket.connect':
            return
        raise PermissionError('INDEP_NETWORK_REFUSED')
    sys.addaudithook(hook)

def _check(obj, path, problems):
    if isinstance(obj, dict):
        if len(obj) > MAX_DICT:
            problems.append(f'dict-too-large:{path}')
        for k, v in obj.items():
            if not isinstance(k, str):
                problems.append(f'non-string-key:{path}')
                continue
            low = k.lower()
            if any((part in low for part in FORBIDDEN_KEY_PARTS)):
                problems.append(f'forbidden-key:{path}/{k}')
            if low.startswith('n_') and (not isinstance(v, (dict, list))):
                if isinstance(v, bool) or not (v == SUPPRESSED or (isinstance(v, int) and (v == 0 or v >= SMALL_CELL))):
                    problems.append(f'small-or-invalid-count:{path}/{k}')
            _check(v, f'{path}/{k}', problems)
    elif isinstance(obj, (list, tuple)):
        if len(obj) > MAX_LIST:
            problems.append(f'list-too-long:{path}')
        for i, v in enumerate(obj):
            _check(v, f'{path}[{i}]', problems)
    elif isinstance(obj, str):
        if len(obj) > MAX_STR:
            problems.append(f'string-too-long:{path}')
    elif isinstance(obj, float):
        if obj != obj or obj in (float('inf'), float('-inf')):
            problems.append(f'non-finite:{path}')
    elif obj is None or isinstance(obj, (bool, int)):
        pass
    else:
        problems.append(f'type-not-allowed:{path}')

def count(n):
    n = int(n)
    return n if n == 0 or n >= SMALL_CELL else SUPPRESSED

def problems_of(obj):
    problems = []
    if not isinstance(obj, dict):
        return ['top-level-not-dict']
    for k in REQUIRED_TOP:
        if k not in obj:
            problems.append(f'missing-top-key:{k}')
    _check(obj, '', problems)
    return problems

def render(obj):
    problems = problems_of(obj)
    if problems:
        stop('aggregate refused: ' + ';'.join(sorted(set(problems))[:8]))
    text = json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + '\n'
    if len(text.encode('utf-8')) > MAX_BYTES:
        stop('aggregate exceeds the summary size cap')
    return text

class Receipt:

    def __init__(self):
        self.t0 = time.time()

    def block(self, device='cpu', device_name='cpu', peak_gpu_mb=0.0, threads=1):
        return {'device': device, 'device_name': device_name, 'threads': int(threads), 'elapsed_s': round(time.time() - self.t0, 1), 'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1), 'peak_gpu_allocated_mb': round(float(peak_gpu_mb), 1)}
