"""Metrics with explicit, method-independent evaluation masks."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates


def rays(shape: tuple[int,int], intrinsics: np.ndarray) -> np.ndarray:
    y,x=np.mgrid[:shape[0],:shape[1]]
    fx,fy,cx,cy=intrinsics
    return np.stack(((x-cx)/fx,(y-cy)/fy,np.ones_like(x)),axis=-1)


def lift_observed(depth0: np.ndarray, depth1: np.ndarray, flow: np.ndarray,
                  intrinsics: np.ndarray) -> np.ndarray:
    """Naive bilinear lifting; no clipping or boundary filtering is hidden here."""
    h,w=depth0.shape
    y,x=np.mgrid[:h,:w]
    qx=x+flow[...,0];qy=y+flow[...,1]
    inside=(qx>=0)&(qy>=0)&(qx<w-1)&(qy<h-1)&np.isfinite(qx)&np.isfinite(qy)
    z=map_coordinates(depth1,[qy,qx],order=1,mode="constant",cval=np.nan,prefilter=False)
    fx,fy,cx,cy=intrinsics
    r1=np.stack(((qx-cx)/fx,(qy-cy)/fy,np.ones_like(qx)),axis=-1)
    result=z[...,None]*r1-depth0[...,None]*rays((h,w),intrinsics)
    result[~inside]=np.nan
    return result


def project_flow(depth0: np.ndarray, scene: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    h,w=depth0.shape;y,x=np.mgrid[:h,:w]
    xyz=depth0[...,None]*rays((h,w),intrinsics)+scene
    fx,fy,cx,cy=intrinsics
    return np.stack((fx*xyz[...,0]/xyz[...,2]+cx-x,fy*xyz[...,1]/xyz[...,2]+cy-y),axis=-1)


def masks(data: dict[str,np.ndarray], raw_scene: np.ndarray) -> dict[str,np.ndarray]:
    labels=data["gt_labels0"]
    edge=np.zeros_like(labels,dtype=bool)
    edge[:,:-1]|=labels[:,:-1]!=labels[:,1:];edge[:,1:]|=labels[:,:-1]!=labels[:,1:]
    edge[:-1]|=labels[:-1]!=labels[1:];edge[1:]|=labels[:-1]!=labels[1:]
    edge=binary_dilation(edge,iterations=2)
    common=data["gt_visible"]&np.isfinite(raw_scene).all(axis=-1)
    return {"common_visible":common,"all_visible":data["gt_visible"],
            "foreground":common&(labels>0),"boundary":common&edge,
            "interior":common&~edge,"injected_outlier":common&data["injected_outlier"],
            "coherent_outlier":common&data["coherent_outlier"],
            "occluded_or_outside":~data["gt_visible"]}


def score(data: dict[str,np.ndarray], scene: np.ndarray, depth0: np.ndarray,
          depth1: np.ndarray, evaluation_masks: dict[str,np.ndarray],
          motion_bound: float = 0.05) -> dict[str,float|int|None]:
    error=np.linalg.norm(scene-data["gt_scene_flow"],axis=-1)*1000
    finite=np.isfinite(scene).all(axis=-1)
    result: dict[str,float|int|None]={}
    for name,mask in evaluation_masks.items():
        valid=mask&finite
        result[f"{name}_n"]=int(valid.sum())
        result[f"{name}_coverage"]=float(valid.sum()/max(1,mask.sum()))
        result[f"{name}_epe_mm"]=float(error[valid].mean()) if valid.any() else None
    v=evaluation_masks["common_visible"]&finite
    values=error[v]
    result["median_epe_mm"]=float(np.median(values)) if values.size else None
    result["p95_epe_mm"]=float(np.quantile(values,.95)) if values.size else None
    result["bad_5mm_percent"]=float(np.mean(values>5)*100) if values.size else None
    result["bad_10mm_percent"]=float(np.mean(values>10)*100) if values.size else None
    magnitude=np.linalg.norm(scene,axis=-1)
    result["bound_violation_percent"]=float(np.mean(magnitude[v]>motion_bound+1e-9)*100) if v.any() else None
    result["max_motion_mm"]=float(np.nanmax(magnitude)*1000)
    for t,z in enumerate((depth0,depth1)):
        result[f"depth{t}_rmse_mm"]=float(np.sqrt(np.nanmean((z-data[f"gt_depth{t}"])**2))*1000)
    flow=project_flow(depth0,scene,data["intrinsics"])
    result["flow_epe_px"]=float(np.mean(np.linalg.norm(flow-data["gt_flow"],axis=-1)[v])) if v.any() else None
    return result
