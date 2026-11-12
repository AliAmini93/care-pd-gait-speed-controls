"""Synthetic corruption and CLI tests. No clinical files, network, checkpoints or private imports."""
from __future__ import annotations
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'/'care_pd'))
import common as C
import p0_core as P
import p1_core as Q
import release_context as X
import run_followup as F

COUNT=0

def check(value,name):
    global COUNT
    if not value:
        raise AssertionError(name)
    COUNT+=1
    print('PASS '+name,flush=True)


def fixtures(folder):
    files={s:{'name':f[0],'bytes':f[1],'sha256':f[2],'md5':f[3]} for s,f in C.FILES.items()}
    p0={'status':'P0_COMPLETE','n_boot':P.N_BOOT,'seed':P.SEED,
        'arms':{a:{'M':{'point':0.5}} for a in Q.PARITY_ARMS},
        'provenance':{'hf_commit':C.HF_COMMIT,'files':files}}
    p0_path=folder/'p0.json'
    p0_path.write_text(json.dumps(p0))
    p1={'status':'P1_COMPLETE','n_boot':P.N_BOOT,'seed':P.SEED,
        'provenance':{'hf_commit':C.HF_COMMIT,'files':files,'p0_result_sha256':X.digest(p0_path)},
        'candidate_configs':[{'name':n,'family':f,'params':p} for n,f,p in Q.CONFIGS],
        'selected_configs':{a:{s:{'all':Q.CONFIGS[0][0]} for s in C.COHORTS} for a in ('S','PS','KS')},
        'arms_inductive_strong':{a:{'macro_f1':{'M':{'point':0.5},'per_cohort':{s:{'point':0.5} for s in C.COHORTS}}} for a in ('S','PS','KS')}}
    (folder/'p1_strong_models.json').write_text(json.dumps(p1))
    return p0,p1


def quiet_main(argv):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        result=F.main(argv)
    return result,output.getvalue()


def test_corruption():
    real=X.load_table
    def forbidden(*a,**kw):
        raise AssertionError('raw data was opened before upstream validation')
    X.load_table=forbidden
    try:
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for family in F.FAMILIES:
                out=root/family
                rc,text=quiet_main([family,'--data-dir',str(root/'absent'),'--out-dir',str(out)])
                check(rc==20 and 'RELEASE_UPSTREAM_MISSING' in text and not out.exists(),family+': missing upstream refused before raw access')
            out=root/'corrupt'
            out.mkdir()
            p0,p1=fixtures(out)
            argv=['r1c','--data-dir',str(root/'absent'),'--out-dir',str(out)]
            mutations=[('p0 status',0,lambda x:x.update(status='P0_SCHEMA_STOP')),
                       ('dataset revision',0,lambda x:x['provenance'].update(hf_commit='wrong')),
                       ('file pins',0,lambda x:x['provenance']['files']['3DGait'].update(sha256='0'*64)),
                       ('bootstrap count',0,lambda x:x.update(n_boot=1)),
                       ('bootstrap seed',0,lambda x:x.update(seed=0)),
                       ('nonfinite metric',0,lambda x:x['arms']['S']['M'].update(point=float('nan'))),
                       ('p1 lineage',1,lambda x:x['provenance'].update(p0_result_sha256='0'*64)),
                       ('candidate parameters',1,lambda x:x['candidate_configs'][0]['params'].update(C=2.0)),
                       ('unknown selected learner',1,lambda x:x['selected_configs']['S']['BMCLab'].update(all='UNKNOWN'))]
            for label,index,mutation in mutations:
                fixtures(out)
                obj=copy.deepcopy((p0,p1)[index])
                mutation(obj)
                target=out/('p0.json' if index==0 else 'p1_strong_models.json')
                target.write_text(json.dumps(obj))
                rc,text=quiet_main(argv)
                check(rc==20 and 'RELEASE_STOP' in text and not (out/'r1c.json').exists(),label+': fail closed with no metric')
            fixtures(out)
            (out/'r1c.json').write_text('{}')
            rc,text=quiet_main(argv)
            check(rc==10 and 'RESULT_EXISTS' in text,'existing result is never overwritten')
            (out/'r1c.json').unlink()
            calls=[]
            def boundary(*args):
                calls.append(1)
                raise C.SourceError('TEST_RAW_BOUNDARY')
            X.load_table=boundary
            fixtures(out)
            rc,text=quiet_main(argv)
            check(rc==20 and calls==[1] and 'TEST_RAW_BOUNDARY' in text,'valid synthetic upstream reaches explicit raw boundary')
    finally:
        X.load_table=real


def test_commands_and_models():
    script=REPO/'scripts/care_pd/run_followup.py'
    for args,expected in [(['--help'],0),([],2),(['r1c'],2)]:
        done=subprocess.run([sys.executable,'-I','-B',str(script),*args],capture_output=True,text=True)
        check(done.returncode==expected,'CLI argument contract '+repr(args))
    import torch
    import r8_d_sequence_loco as D
    for name,kind,hidden in D.CONFIGS:
        model=D.SeqNet(75,kind,hidden,D.TRAIN['dropout']).eval()
        logits,embedding=model(torch.zeros(2,4,75),torch.tensor([4,3]))
        check(logits.shape==(2,3) and torch.isfinite(logits).all().item(),name+': frozen sequence configuration executes on fabricated CPU tensors')


def main():
    test_corruption()
    test_commands_and_models()
    print('RELEASE_CONTRACT_TESTS_PASS n='+str(COUNT),flush=True)
    return 0

if __name__=='__main__':
    sys.exit(main())
