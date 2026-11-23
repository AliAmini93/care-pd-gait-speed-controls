#!/usr/bin/env python3
"""Run all required synthetic suites and report every one; any failure, skip or empty suite makes the run fail.

Every suite is executed even after an earlier one fails, so that a CI log always shows the status of all suites.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

SUITES=('p0_smoke_tests','p1_smoke_tests','cli_smoke_tests','r1c_synthetic_tests','r1c_integration_tests',
        'r8_a_occasion_grain_synthetic_tests','r8_b_within_person_synthetic_tests',
        'r8_c_insite_ceiling_synthetic_tests','r8_d_sequence_loco_synthetic_tests',
        'representation_synthetic_tests','abcd_worlds_tests','abcd_stage_tests',
        'abcd_final_tests','release_contract_tests','root_core_tests','root_fetch_tests','root_parity_tests',
        'root_speed_sensitivity_tests','encoder_inputs_tests','encoder_precheck_tests','encoder_probe_tests')


def main():
    total=0
    bad=[]
    for name in SUITES:
        command=[sys.executable,'-I','-B',str(Path(__file__).resolve().parent/(name+'.py'))]
        done=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(done.stdout,end='',flush=True)
        lines=done.stdout.splitlines()
        passed=sum(s.startswith('PASS ') for s in lines)
        failed=sum(s.startswith('FAIL ') or s.startswith('SKIP ') for s in lines)
        print(f'SUITE={name} EXIT={done.returncode} PASSED={passed} FAILED_OR_SKIPPED={failed}',flush=True)
        if done.returncode or failed or not passed:
            bad.append(name)
        total+=passed
    if bad:
        print(f'SYNTHETIC_SUITES_FAILED failed_suites={len(bad)} of={len(SUITES)} names={",".join(bad)} passed_checks={total}',flush=True)
        return 1
    print(f'ALL_SYNTHETIC_SUITES_PASS suites={len(SUITES)} checks={total}',flush=True)
    print('CUDA_KERNEL_VERIFICATION=NOT_EXECUTED; synthetic learned-model checks used CPU',flush=True)
    return 0

if __name__=='__main__':
    sys.exit(main())
