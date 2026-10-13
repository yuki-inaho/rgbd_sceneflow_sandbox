"""Reproducible primitive benchmark. Numerical core: C++; orchestration: Python."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import numpy as np
from .api import Config,refine
from .primitives import generate_primitives
from .primitive_metrics import region_masks,score_primitive
from .metrics import lift_observed,project_flow


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run_case(out: Path, layout: str, seed: int, source: str, config: Config,
             contrast: float = 1., illumination: bool = False, missing: float = 0.,
             ablations: bool = False, tag: str = '', height: int = 120, width: int = 160
             ) -> tuple[list[dict],list[dict]]:
    name=tag or f'{layout}_{source}_seed{seed}'
    directory=out/name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'metrics.json').exists():
        return json.loads((directory/'metrics.json').read_text()),json.loads((directory/'object_metrics.json').read_text())
    data,meta=generate_primitives(layout,seed,height,width,source,contrast,illumination,missing)
    meta['config']=asdict(config)
    np.savez_compressed(directory/'data.npz',**data)
    save_json(directory/'metadata.json',meta)
    raw=lift_observed(data['depth0'],data['depth1'],data['flow_observed'],data['intrinsics'])
    regions=region_masks(data,raw)
    rows=[];object_rows=[]
    def record(method,r):
        metrics,objects=score_primitive(data,r,regions,meta['objects'],config.max_motion)
        identity={'case':name,'layout':layout,'seed':seed,'source':source,'method':method,
                  'contrast':contrast,'illumination':illumination,'missing_fraction':missing}
        rows.append({**identity,**metrics});object_rows.extend({**identity,**o} for o in objects)
        np.savez_compressed(directory/f'{method}.npz',**{k:v for k,v in r.items() if k!='stats'})
        if 'stats' in r:save_json(directory/f'{method}_stats.json',r['stats'])
        print(name,method,'EPE(mm)=',round(metrics['common_visible_epe_mm'],3),
              'moving=',round(metrics['moving_epe_mm'],3),'object-balanced=',round(metrics['object_balanced_epe_mm'],3),flush=True)
    base={'depth0':data['depth0'],'depth1':data['depth1'],'flow':data['flow_observed']}
    record('raw',{**base,'scene_flow':raw})
    clipped=raw/np.maximum(1,np.linalg.norm(raw,axis=-1,keepdims=True)/config.max_motion)
    record('clipped',{**base,'scene_flow':clipped,'flow':project_flow(data['depth0'],clipped,data['intrinsics'])})
    def solve(c):
        return refine(*(data[k] for k in ('gray0','gray1','depth0','depth1','flow_observed','intrinsics')),c)
    r=solve(config);record('refined',r)
    record('candidate_only',{**base,'scene_flow':np.where((r['support']>.5)[...,None],r['initial_scene_flow'],np.nan),
                             'flow':r['match_flow']})
    if ablations:
        for method,c in [('no_reciprocal',replace(config,reciprocal_check=False)),
                         ('uncentered_photo',replace(config,centered_photo=False)),
                         ('no_diversity',replace(config,diverse_candidates=False)),
                         ('small_patch',replace(config,patch_radius=1)),
                         ('tv',replace(config,use_tgv=False)),
                         ('no_edges',replace(config,edge_aware=False)),
                         ('no_motion_bound',replace(config,enforce_bound=False)),
                         ('no_image',replace(config,image_weight=0,reciprocal_check=False))]:
            save_json(directory/f'{method}_config.json',asdict(c))
            record(method,solve(c))
    save_json(directory/'metrics.json',rows);save_json(directory/'object_metrics.json',object_rows)
    return rows,object_rows


def aggregate(out: Path):
    rows=[];objects=[]
    for p in sorted(out.glob('*/metrics.json')):
        rows.extend(json.loads(p.read_text()))
        objects.extend(json.loads(p.with_name('object_metrics.json').read_text()))
    for name,data in [('metrics',rows),('object_metrics',objects)]:
        save_json(out/f'all_{name}.json',data)
        columns=list(dict.fromkeys(k for row in data for k in row))
        with (out/f'all_{name}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results'))
    parser.add_argument('--layouts',nargs='+',default=['gallery','occlusion','curved'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,23])
    parser.add_argument('--sources',nargs='+',choices=['dis','corrupted_gt'],default=['dis','corrupted_gt'])
    parser.add_argument('--iterations',type=int,default=600)
    parser.add_argument('--outer-iterations',type=int,default=3)
    parser.add_argument('--height',type=int,default=120);parser.add_argument('--width',type=int,default=160)
    parser.add_argument('--ablations',action='store_true');parser.add_argument('--stress',action='store_true')
    parser.add_argument('--concept',action='store_true')
    args=parser.parse_args()
    cfg=Config(iterations=args.iterations,outer_iterations=args.outer_iterations,patch_radius=2)
    args.out.mkdir(parents=True,exist_ok=True)
    for layout in args.layouts:
        for source in args.sources:
            for seed in args.seeds:
                run_case(args.out,layout,seed,source,cfg,height=args.height,width=args.width)
                aggregate(args.out)
    if args.ablations:
        run_case(args.out,'gallery',37,'dis',cfg,ablations=True,tag='ablation_gallery_seed37',height=args.height,width=args.width)
        aggregate(args.out)
    if args.stress:
        run_case(args.out,'occlusion',37,'dis',cfg,tag='stress_reference',height=args.height,width=args.width)
        aggregate(args.out)
        for tag,contrast,illumination,missing in [('weak_texture',.08,False,0.),('illumination',1.,True,0.),('missing_depth',1.,False,.08)]:
            run_case(args.out,'occlusion',37,'dis',cfg,contrast,illumination,missing,tag=tag,height=args.height,width=args.width)
            aggregate(args.out)
    if args.concept:
        for contrast,tag in [(1.,'rotation_textured'),(0.,'rotation_uniform')]:
            run_case(args.out,'rotation',37,'dis',cfg,contrast=contrast,tag=tag,height=args.height,width=args.width)
            aggregate(args.out)
    print('Saved benchmark:',args.out,flush=True)

if __name__=='__main__':main()
