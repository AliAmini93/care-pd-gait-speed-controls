"""Exercise every diagnostic condition with the actual six frozen learners on fabricated arrays."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'care_pd'))
import numpy as np
import common as C
import p1_core as Q
import gait_site_r1c as M


def main():
    rng=np.random.default_rng(140)
    cohort=np.repeat(np.arange(4),36)
    label=np.tile(np.tile(np.arange(3),12),4)
    subject=np.repeat(np.arange(24),6)
    speed=1.5-.4*label+rng.normal(0,.05,label.size)
    t={'cohort':cohort,'subject':subject,'label':label,'speed':speed,
       'speed_alt':speed.copy(),'duration':np.full(label.size,3.),
       'pose':rng.normal(size=(label.size,138)),'kin':rng.normal(size=(label.size,21))}
    feats=Q.arm_features(t)
    scorer=Q.Scorer(t,M.N_BOOT)
    baseline={a:Q.outer_run(feats[a],label,cohort,subject,'walk',Q.CONFIGS,[0]) for a in M.ARMS}
    p1={'arms_inductive_strong':{},'selected_configs':{}}
    for a in M.ARMS:
        p1['arms_inductive_strong'][a]=scorer.all_metrics(M.selected_conf(scorer,baseline[a]))
        p1['selected_configs'][a]={s:{'all':Q.CONFIGS[baseline[a][i]['selected']['all']][0]} for i,s in enumerate(C.COHORTS)}
    assert M.check_baseline_parity(t,baseline,p1,scorer)
    print('PASS diagnostic baseline parity with fabricated aggregate',flush=True)
    bad=copy.deepcopy(p1)
    bad['arms_inductive_strong']['S']['macro_f1']['M']['point']+=.001
    try:
        M.check_baseline_parity(t,baseline,bad,scorer)
    except C.SourceError:
        print('PASS corrupt diagnostic baseline refused',flush=True)
    else:
        raise AssertionError('corrupt baseline accepted')
    d1=M.d1_analysis(t,feats,baseline,scorer)
    assert d1['n_boot']==500 and set(d1['arms'])=={'S','PS'}
    assert all({'argmax','source_threshold','target_label_oracle'}<=set(d1['arms'][a]) for a in M.ARMS)
    print('PASS source cutpoints, rank metrics and target-label oracle execute with distinct labels',flush=True)
    d0=M.d0_analysis(t,feats,baseline,scorer)
    assert len(d0['seeds_results'])==3
    assert all(set(r['arms'])=={'S_plus_IID138','S_plus_permutedPose'} for r in d0['seeds_results'])
    assert all(set(v)==set(M.POLICIES) for r in d0['seeds_results'] for v in r['arms'].values())
    text=json.dumps(M.clean({'D0':d0,'D1':d1}),allow_nan=False)
    assert 'prediction_rows' not in text and 'subject_index' not in text
    print('PASS all diagnostic seeds and learner policies execute and serialize aggregate-only output',flush=True)
    print('R1C_INTEGRATION_TESTS_PASS n=4',flush=True)
    return 0

if __name__=='__main__':
    sys.exit(main())
