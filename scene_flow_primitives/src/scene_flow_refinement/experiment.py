"""Reproducible synthetic experiments; GT is isolated from estimator arguments."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict,replace
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from .api import Config,refine
from ._ffi import library
from .synthetic import generate
from .metrics import lift_observed,masks,score
from .visualize import make_figures


def estimate(data:dict,config:Config)->dict:
    return refine(data["rgb0"],data["rgb1"],data["depth0"],data["depth1"],
                  data["flow_observed"],data["intrinsics"],config)


def run_one(seed:int,directory:Path,config:Config,*,ablations:bool=False,
            figures:bool=False,weak_texture:bool=False,noise_scale:float=1.0,
            motion_scale:float=1.0,height:int=96,width:int=128,resume:bool=False)->list[dict]:
    directory.mkdir(parents=True,exist_ok=True)
    data=generate(seed=seed,height=height,width=width,sigma_depth=.003*noise_scale,
                  sigma_flow=.7*noise_scale,weak_texture=weak_texture,motion_scale=motion_scale)
    config=replace(config,sigma_depth=.003*noise_scale,sigma_flow=.7*noise_scale)
    cache_data_match=False
    if resume and (directory/"data.npz").exists():
        with np.load(directory/"data.npz",allow_pickle=False) as old:
            cache_data_match=set(old.files)==set(data) and all(np.array_equal(old[key],value,equal_nan=True) for key,value in data.items())
    np.savez_compressed(directory/"data.npz",**data)
    raw=lift_observed(data["depth0"],data["depth1"],data["flow_observed"],data["intrinsics"])
    clipped=raw/np.maximum(1,np.linalg.norm(raw,axis=-1,keepdims=True)/config.max_motion)
    base={"depth0":data["depth0"],"depth1":data["depth1"]}
    outputs={"raw":{**base,"scene_flow":raw},"clipped":{**base,"scene_flow":clipped}}
    native_hash=hashlib.sha256(Path(library()._name).read_bytes()).hexdigest()
    all_cached=True
    variants={"joint_tgv":config}
    if ablations:
        variants.update({"joint_tv":replace(config,use_tgv=False),
                         "no_edge_cut":replace(config,edge_aware=False),
                         "no_ball_projection":replace(config,enforce_bound=False)})
    for name,cfg in variants.items():
        cached=directory/f"{name}.npz"
        diagnostics=directory/f"{name}_diagnostics.json"
        metadata=json.loads(diagnostics.read_text()) if cache_data_match and cached.exists() and diagnostics.exists() else {}
        if metadata.get("config")==asdict(cfg) and metadata.get("native_sha256")==native_hash:
            with np.load(cached,allow_pickle=False) as archive:
                result={key:archive[key] for key in archive.files}
            result["stats"]=metadata["stats"]
        else:
            all_cached=False
            result=estimate(data,cfg)
        print(f"finished {directory.name}/{name}: {result['stats']['elapsed_seconds']:.2f}s",flush=True)
        outputs[name]=result
        np.savez_compressed(directory/f"{name}.npz",**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        (directory/f"{name}_diagnostics.json").write_text(json.dumps({"config":asdict(cfg),"stats":result["stats"],"native_sha256":native_hash},indent=2),encoding="utf-8")
    # This baseline isolates observation-only re-matching from joint regularization.
    # Unsupported initial scene vectors are zero and are NOT removed from evaluation.
    outputs["rematched"]={**base,"scene_flow":outputs["joint_tgv"]["initial_scene_flow"]}
    evaluation=masks(data,raw)
    records=[]
    for method,result in outputs.items():
        row={"seed":seed,"method":method,**score(data,result["scene_flow"],result["depth0"],result["depth1"],evaluation,config.max_motion)}
        if "stats" in result:
            row.update(result["stats"])
            valid=result["support"]>0
            residual=result["geometry_residual"][valid]*1000
            row["geometry_mean_mm"]=float(np.mean(residual))
            row["geometry_p95_mm"]=float(np.quantile(residual,.95))
            row["support_percent"]=float(valid.mean()*100)
            row["visible_support_percent"]=float(valid[data["gt_visible"]].mean()*100)
            invalid_visibility=~data["gt_visible"]
            row["nonvisible_support_percent"]=float(valid[invalid_visibility].mean()*100)
            row["max_depth_adjustment_mm"]=max(float(np.max(np.abs(result[f"depth{t}"]-data[f"depth{t}"]))*1000) for t in (0,1))
        records.append(row)
    summary={"conditions":{"seed":seed,"height":height,"width":width,"weak_texture":weak_texture,
                           "noise_scale":noise_scale,"motion_scale":motion_scale,
                           "actual_outlier_fraction":float(data["injected_outlier"].mean()),
                           "gt_max_motion_mm":float(np.linalg.norm(data["gt_scene_flow"],axis=-1).max()*1000),
                           "gt_visible_fraction":float(data["gt_visible"].mean()),
                           "evaluation_pixels":int(evaluation["common_visible"].sum())},"records":records}
    (directory/"metrics.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    if figures and not (cache_data_match and all_cached and (directory/"figures"/"comparison.png").exists()):
        make_figures(data,outputs,evaluation,directory/"figures",config.max_motion)
    print(json.dumps({"directory":str(directory),"epe_mm":{r["method"]:round(r["common_visible_epe_mm"],4) for r in records}},ensure_ascii=False),flush=True)
    return records


def environment()->dict:
    versions={}
    for package in ("numpy","scipy","matplotlib","pillow","pytest","setuptools"):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    try:compiler=subprocess.check_output(["c++","--version"],text=True).splitlines()[0]
    except (OSError,subprocess.SubprocessError):compiler="not available"
    return {"python":sys.version,"platform":platform.platform(),"packages":versions,"compiler":compiler}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("results"))
    parser.add_argument("--seeds",type=int,nargs="+",default=[0])
    parser.add_argument("--iterations",type=int,default=800)
    parser.add_argument("--outer-iterations",type=int,default=3)
    parser.add_argument("--height",type=int,default=96);parser.add_argument("--width",type=int,default=128)
    parser.add_argument("--ablations",action="store_true")
    parser.add_argument("--stress",action="store_true")
    parser.add_argument("--resume",action="store_true",help="reuse saved solver results when inputs and configuration match")
    args=parser.parse_args()
    config=Config(iterations=args.iterations,outer_iterations=args.outer_iterations)
    args.out.mkdir(parents=True,exist_ok=True)
    all_records=[]
    for index,seed in enumerate(args.seeds):
        rows=run_one(seed,args.out/f"seed_{seed}",config,ablations=args.ablations and index==0,
                     figures=index==0,height=args.height,width=args.width,resume=args.resume)
        all_records.extend({"experiment":"nominal",**row} for row in rows)
    if args.stress:
        stress_cases=[("high_noise",dict(noise_scale=2.5),config),
                      ("weak_texture",dict(weak_texture=True),config),
                      ("wrong_bound",dict(),replace(config,max_motion=.02))]
        for name,kwargs,cfg in stress_cases:
            rows=run_one(101,args.out/name,cfg,figures=True,height=args.height,width=args.width,resume=args.resume,**kwargs)
            all_records.extend({"experiment":name,**row} for row in rows)
    (args.out/"all_metrics.json").write_text(json.dumps(all_records,indent=2,allow_nan=False),encoding="utf-8")
    fields=list(dict.fromkeys(key for row in all_records for key in row))
    with (args.out/"all_metrics.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(all_records)
    (args.out/"environment.json").write_text(json.dumps(environment(),indent=2),encoding="utf-8")

if __name__=="__main__":main()
