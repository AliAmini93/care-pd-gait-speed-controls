"""Portable analysis kernel. See docs/extended_protocol.md for scope, provenance and I/O."""
from __future__ import annotations
import hashlib
import math
import numpy as np
from scipy.spatial.transform import Rotation
N_JOINTS = 24
ENVS = ('E0', 'E1', 'E2', 'E3')
SOURCE_ENVS = (0, 1, 2)
TARGET_ENV = 3
FPS = (30.0, 25.0, 60.0, 50.0)
MEAS_SCALE = (1.0, 0.85, 1.15, 1.05)
SPINE3_TILT = (0.0, 0.04, -0.04, 0.08)
PERSONS_PER_ENV = 24
OCCASIONS = 2
WALKS_PER_OCCASION = 3
WALKS_PER_PERSON = OCCASIONS * WALKS_PER_OCCASION
PERSON_GRADES = ((10, 8, 6), (7, 10, 7), (6, 8, 10), (8, 8, 8))
OCCASION_SD = 0.35
U_CLIP = (-0.49, 2.49)
THRESHOLDS = (0.5, 1.5)
W4_TARGET_THRESHOLDS = (0.0, 1.0)
DURATION_S = (2.5, 9.0)
TIMEBASE = 0.02
BASE_JITTER = 0.004
WORLDS = ('W1', 'W2', 'W3', 'W4')
FALSIFIERS = ('FX_APERIODIC', 'FX_UNIT')
CUE = {'W1': (0.25, 0.03, 0.08), 'W2': (0.125, 0.015, 0.04), 'W3': (0.0, 0.0, 0.0), 'W4': (0.25, 0.03, 0.08), 'FX_APERIODIC': (0.25, 0.03, 0.08), 'FX_UNIT': (0.25, 0.03, 0.08)}
SPURIOUS_KAPPA = {'W2': (1.0, 0.6, 0.2, -0.8)}
FX_UNIT_TARGET_SCALE = 0.01
SEEDS = {'test': 20261106, 'dev': 7700123}
REPLICATES = {'test': 3, 'dev': 1}
WORLD_CODE = {'W1': 1, 'W2': 2, 'W3': 3, 'W4': 4, 'FX_APERIODIC': 5, 'FX_UNIT': 6}
N_NOISE_TONES = 40
L_HIP, R_HIP, SPINE1, L_KNEE, R_KNEE, SPINE2, L_ANKLE, R_ANKLE, SPINE3 = (1, 2, 3, 4, 5, 6, 7, 8, 9)
L_FOOT, R_FOOT, NECK, L_COLLAR, R_COLLAR, HEAD = (10, 11, 12, 13, 14, 15)
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HAND, R_HAND = (16, 17, 18, 19, 20, 21, 22, 23)
AXIS_X, AXIS_Y, AXIS_Z = np.eye(3)

class WorldError(Exception):
    pass

def _rot(axis, angle):
    return np.asarray(angle, dtype=np.float64)[:, None] * np.asarray(axis, dtype=np.float64)[None, :]

def _cycles(rng, n_needed_s, f, sigma, t0):
    n = int(math.ceil((n_needed_s + 2.0) * f)) + 3
    dur = 1.0 / f * np.maximum(0.5, 1.0 + sigma * rng.standard_normal(n))
    amp = np.maximum(0.3, 1.0 + sigma * rng.standard_normal(n))
    starts = t0 + np.concatenate([[0.0], np.cumsum(dur)[:-1]])
    return (starts, dur, amp)

def _phase_amp(t, starts, dur, amp):
    k = np.clip(np.searchsorted(starts, t, side='right') - 1, 0, starts.size - 1)
    phase = 2.0 * math.pi * (k + (t - starts[k]) / dur[k])
    centres = starts + dur / 2.0
    return (phase, np.interp(t, centres, amp))

class _Drivers:

    def __init__(self, phase, t, aperiodic, rng):
        self.phase, self.t, self.aperiodic = (phase, t, aperiodic)
        if aperiodic:
            self.freq = rng.uniform(0.2, 4.0, N_NOISE_TONES)
            self.rng = rng

    def one(self, c):
        if not self.aperiodic:
            return np.sin(self.phase + c)
        ph = self.rng.uniform(0.0, 2.0 * math.pi, N_NOISE_TONES)
        x = np.sin(2.0 * math.pi * self.freq[None, :] * self.t[:, None] + ph[None, :]).sum(axis=1)
        return x / math.sqrt(N_NOISE_TONES)

    def two(self, c):
        if not self.aperiodic:
            return np.sin(2.0 * self.phase + c)
        return self.one(c)

def _walk(rng, world, env, person_rec, u, label):
    c_arm, c_var, c_asym = CUE[world]
    fps = FPS[env]
    v = max(0.25, 1.3 - 0.25 * u + 0.1 * person_rec['xi_speed'] + 0.05 * rng.standard_normal())
    f = 0.7 + 0.3 * v + 0.02 * person_rec['xi_cadence']
    log_arm = -c_arm * (u - 1.0) + 0.1 * person_rec['xi_arm'] + 0.05 * rng.standard_normal()
    sigma = 0.02 + c_var * max(u, 0.0) + 0.005 * abs(rng.standard_normal())
    asym = c_asym * max(u + 0.3 * person_rec['xi_asym'], 0.0)
    kappa = SPURIOUS_KAPPA.get(world)
    jitter = BASE_JITTER * (math.exp(kappa[env] * (u - 1.0)) if kappa is not None else 1.0)
    duration = rng.uniform(*DURATION_S)
    n = max(8, int(duration * fps))
    tau = rng.uniform(-TIMEBASE, TIMEBASE)
    t = np.arange(n) / fps * (1.0 + tau)
    starts, dur, amp = _cycles(rng, t[-1], f, sigma, -rng.uniform(0.0, 1.0 / f))
    phase, a = _phase_amp(t, starts, dur, amp)
    drv = _Drivers(phase, t, world == 'FX_APERIODIC', rng)
    arm = math.exp(log_arm)
    side = person_rec['side']
    a_l = a * (1.0 - asym if side == 0 else 1.0)
    a_r = a * (1.0 - asym if side == 1 else 1.0)
    Ah, Ak, Aa = (0.25 + 0.12 * v, 0.45 + 0.2 * v, 0.15 + 0.05 * v)
    As, Ae = ((0.08 + 0.1 * v) * arm, (0.1 + 0.06 * v) * arm)
    pi = math.pi
    angles = {L_HIP: (AXIS_X, Ah * a_l * drv.one(0.0)), R_HIP: (AXIS_X, Ah * a_r * drv.one(pi)), L_KNEE: (AXIS_X, Ak * a_l * (0.6 * drv.one(-0.5) + 0.25 * drv.two(-1.0))), R_KNEE: (AXIS_X, Ak * a_r * (0.6 * drv.one(pi - 0.5) + 0.25 * drv.two(2 * pi - 1.0))), L_ANKLE: (AXIS_X, Aa * a_l * drv.one(1.2)), R_ANKLE: (AXIS_X, Aa * a_r * drv.one(pi + 1.2)), L_FOOT: (AXIS_X, 0.15 * a_l * drv.one(2.0)), R_FOOT: (AXIS_X, 0.15 * a_r * drv.one(pi + 2.0)), L_SHOULDER: (AXIS_X, As * a_l * drv.one(pi)), R_SHOULDER: (AXIS_X, As * a_r * drv.one(0.0)), L_ELBOW: (AXIS_X, Ae * a_l * 0.5 * drv.one(pi - 0.3)), R_ELBOW: (AXIS_X, Ae * a_r * 0.5 * drv.one(-0.3)), SPINE1: (AXIS_X, 0.02 * drv.two(0.0)), SPINE2: (AXIS_X, 0.02 * drv.two(0.3)), SPINE3: (AXIS_Y, 0.04 * drv.one(0.5)), NECK: (AXIS_X, 0.01 * drv.two(0.6)), HEAD: (AXIS_X, 0.01 * drv.two(0.9)), L_COLLAR: (AXIS_Z, 0.02 * drv.one(pi)), R_COLLAR: (AXIS_Z, 0.02 * drv.one(0.0)), L_WRIST: (AXIS_X, 0.03 * drv.one(pi)), R_WRIST: (AXIS_X, 0.03 * drv.one(0.0)), L_HAND: (AXIS_X, 0.02 * drv.one(pi)), R_HAND: (AXIS_X, 0.02 * drv.one(0.0))}
    heading0 = rng.uniform(0.0, 2.0 * pi)
    heading = heading0 + 0.15 * np.sin(pi * t / max(t[-1], 1e-06))
    pose = np.empty((n, 3 * N_JOINTS))
    root = Rotation.from_rotvec(_rot(AXIS_Y, heading)) * Rotation.from_rotvec(_rot(AXIS_Y, 0.05 * drv.one(0.0))) * Rotation.from_rotvec(_rot(AXIS_X, 0.02 * drv.two(0.0))) * Rotation.from_rotvec(jitter * rng.standard_normal((n, 3)))
    pose[:, 0:3] = root.as_rotvec()
    for j in range(1, N_JOINTS):
        axis, ang = angles[j]
        rest = person_rec['rest'][j].copy()
        if j == SPINE3:
            rest = rest + np.array([0.0, 0.0, SPINE3_TILT[env]])
        r = Rotation.from_rotvec(np.tile(rest, (n, 1))) * Rotation.from_rotvec(_rot(axis, ang)) * Rotation.from_rotvec(jitter * rng.standard_normal((n, 3)))
        pose[:, 3 * j:3 * j + 3] = r.as_rotvec()
    scale = MEAS_SCALE[env]
    speed_t = v * (1.0 + 0.05 * drv.two(0.0))
    dt = (1.0 + tau) / fps
    fwd = np.stack([np.sin(heading), np.cos(heading)], axis=1)
    lat = np.stack([np.cos(heading), -np.sin(heading)], axis=1)
    xz = np.cumsum(speed_t[:, None] * dt * fwd, axis=0) + 0.02 * drv.one(0.0)[:, None] * lat
    trans = np.column_stack([xz[:, 0], 0.9 + 0.012 * drv.two(0.0), xz[:, 1]]) * scale
    truth = {'v': v, 'f_true': f, 'f_apparent': f * (1.0 + tau), 'log_arm': log_arm, 'sigma_cycle': sigma, 'asym': asym, 'side': side, 'jitter': jitter, 'u': u, 'label': label, 'tau': tau, 'phase': phase.astype(np.float32), 'periodic': world != 'FX_APERIODIC'}
    return (pose, trans, fps, truth)

def _digitize(u, thresholds):
    return int(np.searchsorted(np.asarray(thresholds), u, side='right'))

def generate(world, seed_set, replicate):
    if world not in CUE:
        raise WorldError('UNKNOWN_WORLD')
    if seed_set not in SEEDS or not 0 <= replicate < REPLICATES[seed_set]:
        raise WorldError('UNKNOWN_SEED')
    rng = np.random.default_rng([SEEDS[seed_set], WORLD_CODE[world], replicate])
    poses, transes, fpss, labels, envs, persons, occasions, truths = ([], [], [], [], [], [], [], [])
    pid = 0
    for env in range(len(ENVS)):
        grades = np.repeat(np.arange(3), PERSON_GRADES[env])
        if grades.size != PERSONS_PER_ENV:
            raise WorldError('PERSON_TABLE')
        thresholds = W4_TARGET_THRESHOLDS if world == 'W4' and env == TARGET_ENV else THRESHOLDS
        for g in grades:
            rec = {'xi_speed': rng.standard_normal(), 'xi_cadence': rng.standard_normal(), 'xi_arm': rng.standard_normal(), 'xi_asym': rng.standard_normal(), 'side': int(rng.integers(0, 2)), 'rest': 0.03 * rng.standard_normal((N_JOINTS, 3))}
            for occ in range(OCCASIONS):
                u = float(np.clip(g + OCCASION_SD * rng.standard_normal(), *U_CLIP))
                lab = _digitize(u, thresholds)
                for _ in range(WALKS_PER_OCCASION):
                    pose, trans, fps, truth = _walk(rng, world, env, rec, u, lab)
                    if world == 'FX_UNIT' and env == TARGET_ENV:
                        trans = trans * FX_UNIT_TARGET_SCALE
                    poses.append(pose)
                    transes.append(trans)
                    fpss.append(fps)
                    labels.append(lab)
                    envs.append(env)
                    persons.append(pid)
                    occasions.append(occ)
                    truth['source_threshold_label'] = _digitize(u, THRESHOLDS)
                    truths.append(truth)
            pid += 1
    return {'world': world, 'seed_set': seed_set, 'replicate': replicate, 'pose': poses, 'trans': transes, 'fps': np.asarray(fpss), 'label': np.asarray(labels), 'env': np.asarray(envs), 'person': np.asarray(persons), 'occasion': np.asarray(occasions), 'truth': truths}

def digest(data):
    h = hashlib.sha256()
    for key in ('fps', 'label', 'env', 'person', 'occasion'):
        h.update(np.ascontiguousarray(data[key]).tobytes())
    for p, tr in zip(data['pose'], data['trans']):
        h.update(np.ascontiguousarray(p).tobytes())
        h.update(np.ascontiguousarray(tr).tobytes())
    return h.hexdigest()

def split(data):
    src = np.flatnonzero(np.isin(data['env'], SOURCE_ENVS))
    tgt = np.flatnonzero(data['env'] == TARGET_ENV)
    return (src, tgt)
