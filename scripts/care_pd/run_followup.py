#!/usr/bin/env python3
"""Portable command-line orchestration of the frozen exploratory follow-up kernels.

Only users who separately obtained the pinned CARE-PD data under the custodians'
terms may run these commands on real data. Nothing is downloaded by this script.
All output is local, aggregate-only and outside this code repository.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parent))

import common as C
import p1_core as Q
import release_context as X
import r8_core as R

FAMILIES = ('r1c','r8a','r8b','r8c','r8d','representation','abcd','abcd_a','abcd_b','abcd_c','abcd_d')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('family',choices=FAMILIES)
    p.add_argument('--data-dir',type=Path,required=True,help='separately obtained pinned canonical files, outside this repository')
    p.add_argument('--out-dir',type=Path,required=True,help='local aggregate output directory outside this repository')
    p.add_argument('--p0-result',type=Path,help='completed P0 aggregate; default: <out-dir>/p0.json')
    p.add_argument('--p1-result',type=Path,help='completed P1 aggregate; default: <out-dir>/p1_strong_models.json')
    p.add_argument('--r8d-result',type=Path,help='required by representation; default: <out-dir>/r8d.json')
    return p.parse_args(argv)


def compute(family,objs,t,p0,p1,r8d=None):
    """The original numerical ordering with explicit already-validated inputs."""
    if family=='r1c':
        import gait_site_r1c as M
        feats=Q.arm_features(t)
        scorer=Q.Scorer(t,M.N_BOOT)
        baseline={a:Q.outer_run(feats[a],t['label'],t['cohort'],t['subject'],'walk',Q.CONFIGS,[0]) for a in M.ARMS}
        M.check_baseline_parity(t,baseline,p1,scorer)
        return {'status':'R1C_COMPLETE_EXPLORATORY','D1':M.d1_analysis(t,feats,baseline,scorer),
                'D0':M.d0_analysis(t,feats,baseline,scorer)}
    if family=='r8a':
        import r8_a_occasion_grain as M
        med_site=M.collect_medication(objs[M.SITE])
        ci=C.COHORTS.index(M.SITE)
        if len(med_site)!=int((t['cohort']==ci).sum()):
            raise C.SourceError('R8_A_MEDICATION_ROW_ALIGNMENT')
        feats=Q.arm_features(t)
        warned=[0]
        folds={a:R.single_outer_fold(feats[a],t['label'],t['cohort'],t['subject'],'walk',Q.CONFIGS,warned,ci) for a in M.ARMS}
        te=folds['S']['test_mask']
        for a in M.ARMS:
            if not M.np.array_equal(folds[a]['test_mask'],te):
                raise C.SourceError('R8_A_TEST_MASK_MISMATCH')
        parity=M.check_parity(t,folds,p1,te)
        site_rows=M.np.flatnonzero(t['cohort']==ci)
        med=[med_site[i] for i in range(site_rows.size) if te[site_rows[i]]]
        body=M.analyze(t['label'][te],t['subject'][te],med,{a:R.selected_proba(folds[a]) for a in M.ARMS})
        return {'status':M.STATUS,'p1_parity':parity,**body,'limits':M.LIMITS}
    if family=='r8b':
        import r8_b_within_person as M
        within=X.check_within_parity(t,p0)
        return {'status':M.STATUS,'p0_within_subject_parity':within,
                **M.analyze(t,M.collect_medication(objs['BMCLab'])),'limits':M.LIMITS}
    if family=='r8c':
        import r8_c_insite_ceiling as M
        return {'status':M.STATUS,**M.run(t,M.collect_medication(objs['BMCLab']),p1=p1),'limits':M.LIMITS}
    if family=='r8d':
        import r8_d_sequence_loco as M
        return {'status':M.STATUS,**M.run(t,M.build_sequences(objs),p1=p1,device='cuda'),'limits':M.LIMITS}
    if family=='representation':
        import torch
        import representation_core as RC
        import representation_native as RN
        import representation_one_shot as M
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_num_threads(1)
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0,M.VRAM_CAP_GIB*2**30/total),0)
        t2,store=M.build_inputs(objs)
        # build_inputs uses the same validated row order; no new selection rule.
        Q.check_stops(t2)
        start=time.monotonic()
        def budget():
            if time.monotonic()-start>M.TRAIN_BUDGET_S:
                raise C.SourceError('REP_TIME_BUDGET')
        body=RN.run_campaign(t2,store,'cuda',p1=p1,budget=budget)
        if body['S_parity']['status']!='PASS':
            raise C.SourceError('REP_S_PARITY_NOT_PASS')
        resources={'device':torch.cuda.get_device_name(0),'vram_cap_gib':M.VRAM_CAP_GIB,
                   'peak_vram_allocated_gib':round(torch.cuda.max_memory_allocated()/2**30,3)}
        import representation_public_encoder as PE
        return {'schema':RC.SCHEMA,'status':RC.STATUS_COMPLETE,'program_status':RC.PROGRAM_STATUS,
                **body,'P_published_encoder':PE.p_status_record(),'native_design':{'config':RN.CONFIG},
                'input_frame_rates_by_cohort':M.fps_summary(objs),'resources':resources,'limits':M.LIMITS}
    if family=='abcd':
        import abcd_final_one_shot as M
        import abcd_stage_a as A
        import abcd_stage_b as B
        t2,b=M.build_inputs(objs)
        FA,FB=A.features(b['store']),B.features(b['store'])
        start=time.monotonic()
        def budget():
            if time.monotonic()-start>M.FINAL_BUDGET_S:
                raise C.SourceError('ABCD_FINAL_TIME_BUDGET')
        body=M.run_final(t2,b,FA,FB,p1,budget=budget)
        if not body['candidates_scored']:
            raise C.SourceError('ABCD_FINAL_BASELINE_PARITY')
        body['site_decodability_balanced_accuracy_descriptive']={
            'A_features':M.site_decodability(FA,b),'B_features':M.site_decodability(FB,b)}
        return {'schema':M.SCHEMA,'status':'GAIT_ABCD_FINAL_COMPLETE',**body,
                'limits':M.LIMITS+['Shrinkage validation is not fully nested; development is exploratory.',
                                   'Environmental support uses all cohorts: its candidate is unscored when support fails.']}
    if family.startswith('abcd_'):
        return development(family[-1],objs)
    raise C.SourceError('RELEASE_UNKNOWN_FAMILY')


def development(part,objs):
    """Original exploratory development gates; the final score is a separate command."""
    import importlib
    import numpy as np
    import abcd_common as AC
    import abcd_stage_b as B
    import abcd_stagerun as SR
    import abcd_stagetests as ST
    M=importlib.import_module('abcd_stage_'+part)
    val=ST.validity_checks(part)
    if not all(v['pass'] for v in val.values()):
        raise C.SourceError('ABCD_SYNTHETIC_VALIDITY_FAILED')
    synth=SR.run_synthetic(M,'test')
    selected={s:objs[s] for s in AC.DEV_SITES}
    b=AC.build_bundle(selected,AC.DEV_SITES)
    AC.check_dev_bundle(b)
    F=M.features(b['store'])
    obs=B.observability(F,b) if part!='a' else {}
    blocked=[s for s,v in obs.items() if v['observable_share']<B.OBSERVABILITY_MIN_SHARE]
    reason='PHASE_NOT_OBSERVABLE:'+','.join(blocked) if blocked else None
    support={}
    if part=='c':
        ok,support=M.support(b)
        if reason is None and not ok:
            reason='ENV_SUPPORT:'+','.join(s for s,v in support.items() if not v['support_pass'])
    if reason:
        dev={'observability':obs,'support':support,'scored':False}
        status=AC.stage_status(True,synth['screen'],None,blocked_reason=reason)
    else:
        proxy=B._Cached(F)
        proxy.STAGE,proxy.CANDIDATE,proxy.CONTROLS,proxy.method_fold=M.STAGE,M.CANDIDATE,M.CONTROLS,M.method_fold
        dev=SR.run_dev(proxy,selected,time.monotonic())
        dev.update({'observability':obs,'support':support,'scored':True})
        status=AC.stage_status(True,synth['screen'],dev['screen'])
    return {'schema':'gait_abcd_development_v1','stage':M.STAGE,'status':status['status'],
            'status_detail':status,'synthetic_validity':val,'synthetic':synth,'development':dev,
            'config':M.CFG,'limits':SR.LIMITS}


def main(argv=None):
    a=parse_args(argv)
    try:
        a.data_dir=X.outside_repo(a.data_dir)
        a.out_dir=X.outside_repo(a.out_dir)
        result=a.out_dir/(a.family+'.json')
        if result.exists():
            print('RELEASE_REFUSED_RESULT_EXISTS')
            return 10
        p0_path=a.p0_result or a.out_dir/'p0.json'
        p1_path=None if a.family=='r8b' else (a.p1_result or a.out_dir/'p1_strong_models.json')
        p0,p1=X.prior_results(p0_path,p1_path)
        paths={'p0':p0_path,'p1':p1_path}
        r8d=None
        if a.family=='representation':
            paths['r8d']=a.r8d_result or a.out_dir/'r8d.json'
            r8d=X.read_aggregate(paths['r8d'],'R8_D_SEQUENCE_LOCO_COMPLETE')
        if a.family in ('r8d','representation'):
            import torch
            if torch.__version__.split('+')[0]!='2.9.1':
                raise C.SourceError('RELEASE_TORCH_VERSION')
            if not torch.cuda.is_available():
                raise C.SourceError('RELEASE_CUDA_UNAVAILABLE')
        if a.family.startswith('abcd_'):
            import abcd_common as AC
            objs={s:C.load_cohort(s,a.data_dir) for s in AC.DEV_SITES}
            t=None
        else:
            objs,t=X.load_table(a.data_dir,p0)
        body=compute(a.family,objs,t,p0,p1,r8d)
        body['release_scope']='EXPLORATORY_NONCLINICAL_PORTABLE_RERUN'
        body['release_provenance']=X.provenance(paths)
        a.out_dir.mkdir(parents=True,exist_ok=True)
        if a.family=='representation':
            import representation_core as RC
            RC.save_aggregate(result,body)
        elif a.family.startswith('abcd'):
            import abcd_privacy as AP
            AP.save_aggregate(result,body)
        else:
            R.save_aggregate(result,body)
    except (C.SourceError,OSError,KeyError,ValueError,TypeError) as exc:
        # Never echo a data-derived value or identifier from an exception.
        reason=str(exc) if isinstance(exc,C.SourceError) else type(exc).__name__
        print('RELEASE_STOP:'+reason)
        return 20
    print('RELEASE_AGGREGATES_ONLY_COMPLETE:'+a.family)
    return 0


if __name__=='__main__':
    sys.exit(main())
