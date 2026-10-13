"""Region-explicit metrics. No method-specific acceptance mask defines the headline set."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation
from .metrics import lift_observed


def region_masks(data: dict[str,np.ndarray], raw: np.ndarray) -> dict[str,np.ndarray]:
    labels=data['gt_labels0'];edge=np.zeros_like(labels,bool)
    for axis in (0,1):
        a=[slice(None)]*2;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
        jump=labels[tuple(a)]!=labels[tuple(b)]
        edge[tuple(a)]|=jump;edge[tuple(b)]|=jump
    edge=binary_dilation(edge,iterations=2)
    common=data['gt_visible']&np.isfinite(raw).all(-1)
    return {'common_visible':common,'all_visible':data['gt_visible'],
            'moving':common&data['gt_moving'],'static':common&~data['gt_moving'],
            'boundary':common&edge,'interior':common&~edge,
            'static_gray_change':common&~data['gt_moving']&~edge&(np.abs(data['gray1']-data['gray0'])>.04),
            'occluded':data['gt_occluded'],'outside':data['gt_outside'],
            'outlier':common&data['injected_outlier'],'coherent_outlier':common&data['coherent_outlier']}


def score_primitive(data: dict[str,np.ndarray], result: dict, regions: dict[str,np.ndarray],
                    objects: list[dict], bound: float = .05) -> tuple[dict,list[dict]]:
    scene=result['scene_flow'];finite=np.isfinite(scene).all(-1)
    error=np.linalg.norm(scene-data['gt_scene_flow'],axis=-1)*1000
    row={}
    for name,mask in regions.items():
        valid=mask&finite;count=int(valid.sum());row[name+'_n']=count
        row[name+'_coverage']=float(count/max(1,int(mask.sum())))
        row[name+'_epe_mm']=float(error[valid].mean()) if count else None
    common=regions['common_visible']&finite
    for pct in (50,95,99):
        row[f'p{pct}_mm']=float(np.percentile(error[common],pct)) if common.any() else None
    for t in (5,10):
        row[f'bad{t}_percent']=float(np.mean(error[common]>t)*100) if common.any() else None
    moving=regions['moving']&finite
    row['moving_p95_mm']=float(np.percentile(error[moving],95)) if moving.any() else None
    mags=np.linalg.norm(scene,axis=-1)
    row['max_estimated_motion_mm']=float(np.nanmax(mags)*1000)
    row['bound_violation_percent']=float(np.mean(mags[finite]>bound+1e-9)*100)
    for t in (0,1):
        z=result[f'depth{t}'];valid=np.isfinite(data[f'depth{t}'])&(data[f'depth{t}']>0)
        dz=z-data[f'gt_depth{t}'];estimate=np.isfinite(z)
        row[f'depth{t}_rmse_mm']=float(np.sqrt(np.mean(dz[estimate]**2))*1000)
        row[f'depth{t}_observed_rmse_mm']=float(np.sqrt(np.mean(dz[valid&estimate]**2))*1000)
        m=data[f'gt_labels{t}']>=2
        row[f'depth{t}_object_rmse_mm']=float(np.sqrt(np.mean(dz[m&estimate]**2))*1000)
        hole=~valid&estimate
        row[f'depth{t}_hole_rmse_mm']=float(np.sqrt(np.mean(dz[hole]**2))*1000) if hole.any() else None
    flow=result.get('flow')
    if flow is not None:
        fe=np.linalg.norm(flow-data['gt_flow'],axis=-1)
        row['optical_flow_epe_px']=float(np.nanmean(fe[common]))
        row['moving_flow_epe_px']=float(np.nanmean(fe[moving])) if moving.any() else None
    if 'support' in result:
        support=result['support']>.5
        row['support_percent']=float(support.mean()*100)
        for label,mask in [('visible',data['gt_visible']),('occluded',data['gt_occluded'])]:
            row[f'{label}_support_percent']=float(support[mask].mean()*100) if mask.any() else None
        residual=result['geometry_residual'];v=support&np.isfinite(residual)
        row['geometry_mean_mm']=float(np.mean(residual[v])*1000) if v.any() else None
        if 'match_flow' in result and 'flow' in result:
            gap=np.linalg.norm(result['match_flow']-result['flow'],axis=-1)
            finite_gap=support&np.isfinite(gap)
            row['match_reprojection_gap_mean_px']=float(gap[finite_gap].mean()) if finite_gap.any() else None
            row['match_reprojection_gap_p95_px']=float(np.percentile(gap[finite_gap],95)) if finite_gap.any() else None
        row['geometry_p95_mm']=float(np.percentile(residual[v],95)*1000) if v.any() else None
        row['supported_epe_mm']=float(error[common&support].mean()) if (common&support).any() else None
        row['unsupported_epe_mm']=float(error[common&~support].mean()) if (common&~support).any() else None
    row.update(result.get('stats',{}))
    per_object=[]
    for i,obj in enumerate(objects):
        mask=common&(data['gt_labels0']==i)
        if mask.any():
            per_object.append({'object':obj['name'],'object_id':i,'moving':obj['moving'],'count':int(mask.sum()),
                               'epe_mm':float(error[mask].mean()),'p95_mm':float(np.percentile(error[mask],95))})
    motion_objects=[r['epe_mm'] for r in per_object if r['moving']]
    row['object_balanced_epe_mm']=float(np.mean(motion_objects)) if motion_objects else None
    return row,per_object
