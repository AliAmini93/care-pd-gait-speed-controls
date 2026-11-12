"""Frozen mathematical and fabricated-world validity tests of all four residual kernels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'care_pd'))
import abcd_stagetests as ST


def main():
    count=0
    for part in ('a','b','c','d'):
        results=ST.validity_checks(part)
        for name,result in results.items():
            if not result['pass']:
                print('FAIL '+part+':'+name,flush=True)
                return 1
            print('PASS '+part+':'+name,flush=True)
            count+=1
    print('ABCD_STAGE_TESTS_PASS n='+str(count),flush=True)
    return 0

if __name__=='__main__':
    sys.exit(main())
