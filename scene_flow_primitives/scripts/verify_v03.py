"""Recompute one saved v0.3 case and compare arrays without comparing wall time."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scene_flow_refinement import Config,refine

KEYS=('gray0','gray1','depth0','depth1','flow_observed','intrinsics')

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',type=Path,required=True,help='Directory containing data.npz and standard.json/npz')
    parser.add_argument('--method',default='standard')
    parser.add_argument('--out',type=Path,default=None)
    args=parser.parse_args()
    directory=args.case
    specification=json.loads((directory/f'{args.method}.json').read_text())
    if specification['method']=='v02':
        raise ValueError('This verifier checks the current v0.3 API. The matched-input benchmark runs the archived baseline.')
    cfg=Config(**specification['config'])
    with np.load(directory/'data.npz') as data:
        result=refine(*(data[k] for k in KEYS),cfg)
    differences={}
    with np.load(directory/f'{args.method}.npz') as stored:
        for name in stored.files:
            if name=='stats':continue
            if name not in result:
                raise AssertionError(f'Missing output field {name}')
            a,b=stored[name],result[name]
            if a.shape!=b.shape:
                raise AssertionError(f'Shape mismatch for {name}: {a.shape} vs {b.shape}')
            if a.dtype==np.bool_ or b.dtype==np.bool_:
                differences[name]=float(np.any(a!=b))
            else:
                if not np.array_equal(np.isfinite(a),np.isfinite(b)):
                    raise AssertionError(f'Finite support changed for {name}')
                valid=np.isfinite(a)
                differences[name]=float(np.max(np.abs(a[valid]-b[valid]),initial=0.))
    check={'case':str(directory),'method':args.method,'max_abs_by_field':differences,
           'max_abs':max(differences.values()),'cpp_seconds':result['stats']['elapsed_seconds']}
    text=json.dumps(check,indent=2,allow_nan=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text+'\n')
    if check['max_abs']>1e-9:
        raise AssertionError('Reproduction differed by more than 1e-9. See per-field differences and compiler/environment.')

if __name__=='__main__':main()
