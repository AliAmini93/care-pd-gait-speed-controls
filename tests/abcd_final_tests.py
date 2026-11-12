"""Fabricated-only regression tests of the portable residual kernels. No real data is read."""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'care_pd'))
import abcd_privacy as AP
import numpy as np
import abcd_common as AC
import abcd_final_one_shot as FO
import abcd_stage_a as SA
import abcd_stage_b as SB
import abcd_worlds as W
import common as C

def _fab():
    data = W.generate('W1', 'dev', 0)
    b = AC.synthetic_bundle(data)
    t = {'label': b['label'], 'cohort': b['site'], 'subject': b['subject']}
    return (b, t, SA.features(b['store']), SB.features(b['store']))

def test_end_to_end_and_size(state):
    b, t, FA, FB = state['fab']
    body = FO.run_final(t, b, FA, FB, None, n_boot=2000)
    state['body'] = body
    ok = body['candidates_scored'] and body['S_parity']['status'] == 'NOT_CHECKED_FABRICATED'
    ok &= set(FO.CANDIDATES) <= set(body['decisions']) and 'S_plus_D' in body['arms']
    out = {'schema': FO.SCHEMA, 'status': 'GAIT_ABCD_FINAL_COMPLETE', **body, 'p0_parity': 'PASS', 'site_decodability_balanced_accuracy_descriptive': {'A_features': 0.5, 'B_features': 0.5}, 'registered': {'candidates': list(FO.CANDIDATES), 'controls': list(FO.CONTROLS)}, 'input_frame_rates_by_cohort': {s: [30.0] for s in C.COHORTS}, 'resources': {'x': 1.0}, 'provenance': {'inputs': {c: {'bytes': f[1], 'sha256': f[2]} for c, f in C.FILES.items()}}, 'limits': FO.LIMITS}
    n = len(AP.render(out).encode('utf-8'))
    return {'pass': bool(ok and n <= 0.6 * AP.SUMMARY_CAP), 'rendered_bytes': n, 'verdict_on_fabricated': body['terminal_verdict'], 'fabricated_decisions': {k: v['status'] for k, v in body['decisions'].items()}}

def test_target_label_blindness(state):
    b, t, FA, FB = state['fab']
    warned = [0]
    p1, _, _ = FO.run_arms(FA, FB, b, True, True, lambda: None, warned)
    b2 = dict(b)
    lab = b['label'].copy()
    m = np.flatnonzero(b['site'] == 3)
    lab[m] = np.random.default_rng(5).permutation(lab[m])
    b2['label'] = lab
    p2, _, _ = FO.run_arms(FA, FB, b2, True, True, lambda: None, warned)
    same = all((np.array_equal(p1[a][3], p2[a][3]) for a in p1))
    return {'pass': bool(same), 'n_arms': len(p1)}

def test_s_parity_logic(state):
    p1 = {'arms_inductive_strong': {'S': {'macro_f1': {'M': {'point': 0.5}, 'per_cohort': {s: {'point': 0.5} for s in C.COHORTS}}}}, 'selected_configs': {'S': {s: {'all': 'RF_leaf25'} for s in C.COHORTS}}}
    good = FO.s_parity(['RF_leaf25'] * 4, np.full(4, 0.5), p1)['status'] == 'PASS'
    bad_sel = FO.s_parity(['RF_leaf25'] * 3 + ['LR_C1'], np.full(4, 0.5), p1)['status'] == 'FAIL'
    bad_val = FO.s_parity(['RF_leaf25'] * 4, np.array([0.5, 0.5, 0.5, 0.5 + 2e-06]), p1)['status'] == 'FAIL'
    return {'pass': bool(good and bad_sel and bad_val)}

def test_verdict_rules(state):

    def d(*s):
        return {f'x{i}': {'status': v} for i, v in enumerate(s)}
    ok = FO.verdict(d('NO_DETECTED_GAIN', 'EMPIRICAL_GAIN_EXPLORATORY'), True) == 'FINAL_EMPIRICAL_GAIN_EXPLORATORY'
    ok &= FO.verdict(d('NO_DETECTED_GAIN', 'INCONCLUSIVE'), True) == 'FINAL_INCONCLUSIVE'
    ok &= FO.verdict(d('NO_DETECTED_GAIN', 'NOT_RUN_ENV_SUPPORT'), True) == 'FINAL_NO_DETECTED_GAIN'
    ok &= FO.verdict(d('EMPIRICAL_GAIN_EXPLORATORY'), False) == 'FINAL_BLOCKED'
    ok &= FO.verdict(d('NOT_RUN_ENV_SUPPORT'), True) == 'FINAL_BLOCKED'
    return {'pass': bool(ok)}

def test_parity_failure_blocks_candidates(state):
    b, t, FA, FB = state['fab']
    p1 = {'arms_inductive_strong': {'S': {'macro_f1': {'M': {'point': 0.123}, 'per_cohort': {s: {'point': 0.123} for s in C.COHORTS}}}}, 'selected_configs': {'S': {s: {'all': 'LR_C1'} for s in C.COHORTS}}}
    body = FO.run_final(t, b, FA, FB, p1, n_boot=50)
    return {'pass': bool(body['S_parity']['status'] == 'FAIL' and (not body['candidates_scored']) and (body['terminal_verdict'] == 'FINAL_BLOCKED') and ('arms' not in body))}

def test_c_support_rule_on_public_counts(state):
    """C's frozen support rule needs >= 20 walks per grade per cohort; the pinned public P0 count of 3DGait grade 2
    is 14 eligible walks, so S_plus_C is NOT_RUN in the final by the frozen rule (pre-registered)."""
    counts = C.EXPECTED_CLASS_WALKS['3DGait']
    return {'pass': bool(min(counts[:3]) < 20), '3DGait_public_eligible_walks_by_grade_012': list(counts[:3])}

def test_historical_metadata_not_emitted(state):
    """Historical development labels are outside the portable output schema."""
    import ast
    tree = ast.parse(Path(FO.__file__).read_text())
    absent = not any(isinstance(n, ast.Name) and n.id == 'DEV_STATUS' for n in ast.walk(tree))
    return {'pass': bool(absent and not hasattr(FO, 'DEV_STATUS') and 'development_status' not in state['body'])}

TESTS = {'F01_end_to_end_fabricated_and_json_size': test_end_to_end_and_size, 'F02_target_cohort_label_blindness_all_arms': test_target_label_blindness, 'F03_s_parity_logic': test_s_parity_logic, 'F04_terminal_verdict_rules': test_verdict_rules, 'F05_parity_failure_blocks_all_candidates': test_parity_failure_blocks_candidates, 'F06_c_support_rule_on_public_counts': test_c_support_rule_on_public_counts, 'F07_historical_metadata_not_emitted': test_historical_metadata_not_emitted}

def main():
    state = {'fab': _fab()}
    for name, fn in TESTS.items():
        result = fn(state)
        if not result['pass']:
            print('FAIL ' + name, flush=True)
            return 1
        print('PASS ' + name, flush=True)
    print('ABCD_FINAL_TESTS_ALL_PASS n=' + str(len(TESTS)), flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
