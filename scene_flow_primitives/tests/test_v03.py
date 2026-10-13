"""Numerical-contract tests for v0.3; test data are not inference inputs."""
from __future__ import annotations
import ctypes as ct
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scene_flow_refinement import Config,refine
from scene_flow_refinement._ffi import library,_DOUBLE
from scene_flow_refinement.primitives import generate_primitives,make_scene,perturb_scene
from scene_flow_refinement.metrics import project_flow

KEYS=('gray0','gray1','depth0','depth1','flow_observed','intrinsics')

def call(d,c):return refine(*(d[k] for k in KEYS),c)

@pytest.fixture(scope='module')
def small():return generate_primitives('gallery',813,32,40)[0]


def alignment(x,y,w):
    lib=library();lib.sfr_weighted_alignment.argtypes=[ct.c_int]+[_DOUBLE]*4;lib.sfr_weighted_alignment.restype=ct.c_int
    x,y,w=[np.ascontiguousarray(a,dtype=np.float64) for a in (x,y,w)]
    out=np.empty(12)
    status=lib.sfr_weighted_alignment(len(w),*[a.ctypes.data_as(_DOUBLE) for a in (x,y,w,out)])
    if status:raise ValueError(lib.sfr_error().decode())
    return out[:9].reshape(3,3),out[9:]

@pytest.mark.parametrize('planar',[False,True])
@pytest.mark.parametrize('angle',[0.,.08,.6,2.7])
def test_horn_weighted_alignment_recovers_proper_pose(planar,angle):
    rng=np.random.default_rng(3);x=rng.normal(size=(80,3))*.13
    if planar:x[:,2]=1.0
    R=Rotation.from_rotvec(np.array([.2,.4,.8])*angle).as_matrix();t=np.array([.02,-.01,.004])
    Q,b=alignment(x,x@R.T+t,rng.uniform(.1,1.,80))
    np.testing.assert_allclose(Q,R,atol=4e-12)
    np.testing.assert_allclose(b,t,atol=4e-12)
    np.testing.assert_allclose(Q.T@Q,np.eye(3),atol=2e-14)
    assert abs(np.linalg.det(Q)-1)<1e-13


def test_alignment_rejects_negative_weights():
    with pytest.raises(ValueError):alignment(np.ones((4,3)),np.ones((4,3)),[1,1,-1,1])


def test_alignment_rejects_zero_total_weight():
    with pytest.raises(ValueError):alignment(np.ones((4,3)),np.ones((4,3)),np.zeros(4))

@pytest.mark.parametrize('kind',[0,1,2])
@pytest.mark.parametrize('scale',[.1,100.])
def test_dual_prox_closed_form(kind,scale):
    lib=library();lib.sfr_probe_dual.argtypes=[ct.c_int,_DOUBLE,ct.c_double,ct.c_double,ct.c_double,_DOUBLE]
    lib.sfr_probe_dual.restype=ct.c_int
    value=np.array([1.,-2.,3.])*scale;output=np.empty(3);weight=3.;sigma=.4;cap=1.7
    assert lib.sfr_probe_dual(kind,value.ctypes.data_as(_DOUBLE),weight,sigma,cap,output.ctypes.data_as(_DOUBLE))==0
    expected=value.copy()
    if kind==0:expected/=max(1.,np.linalg.norm(expected)/weight)
    else:
        expected/=1+sigma/weight
        if kind==2:expected/=max(1.,np.linalg.norm(expected)/cap)
    np.testing.assert_allclose(output,expected,rtol=1e-14,atol=1e-14)


def test_fused_unfused_match_numerically(small):
    cfg=Config(iterations=35,later_iterations=20,outer_iterations=2)
    a=call(small,cfg);b=call(small,replace(cfg,fused_solver=False))
    for field in ('scene_flow','depth0','depth1','flow','geometry_residual'):
        np.testing.assert_allclose(a[field],b[field],rtol=1e-10,atol=2e-11,equal_nan=True)


def test_no_rigid_means_no_hidden_rigid_contribution(small):
    r=call(small,Config(rigid_prior=False,iterations=15,later_iterations=10,outer_iterations=1))
    assert not r['rigid_weight'].any() and not r['rigid_scene'].any()
    assert r['stats']['rigid_accepted']==0


def test_cold_and_warm_respect_declared_iteration_budget(small):
    for warm in (False,True):
        r=call(small,Config(iterations=11,later_iterations=7,outer_iterations=3,warm_start=warm))
        assert r['stats']['inner_iterations']==25
        assert len(r['history'])==6


def test_block_acceptance_is_monotone_with_rigid_and_huber(small):
    r=call(small,Config(iterations=50,later_iterations=25,outer_iterations=3))
    best=r['history'][:,3]
    assert np.all(np.diff(best)<=1e-7*np.maximum(1,np.abs(best[:-1])))
    assert r['stats']['adjoint_error']<1e-10


def test_hard_motion_and_depth_trust_bounds(small):
    cfg=Config(max_motion=.02,iterations=40,later_iterations=20,outer_iterations=2)
    r=call(small,cfg)
    assert np.linalg.norm(r['scene_flow'],axis=-1).max()<=.02+1e-12
    for t in (0,1):assert np.abs(r[f'depth{t}']-small[f'depth{t}']).max()<=cfg.trust_sigma*cfg.sigma_depth+1e-12
    np.testing.assert_allclose(project_flow(r['depth0'],r['scene_flow'],small['intrinsics']),r['flow'],atol=1e-12)


def test_unbounded_ablation_does_not_retain_search_gate(small):
    r=call(small,Config(enforce_bound=False,iterations=5,later_iterations=5,outer_iterations=1))
    assert r['stats']['geometry_rejected_candidates']==0

@pytest.mark.parametrize('field,value',[('iterations',3.5),('warm_start',.5),('geometry_k',0),('rigid_sigma',float('nan')),('coarse_stride',5)])
def test_v03_config_validation(small,field,value):
    with pytest.raises(ValueError):call(small,replace(Config(iterations=1,outer_iterations=1),**{field:value}))


def test_missing_depth_is_not_silently_supported(small):
    d={k:v.copy() for k,v in small.items()};d['depth0'][10:13,10:13]=np.nan
    r=call(d,Config(iterations=10,later_iterations=5,outer_iterations=1))
    assert not r['depth0_supported'][10:13,10:13].any()
    assert not r['rigid_weight'][10:13,10:13].any()

@pytest.mark.parametrize('layout',['gallery','occlusion','curved'])
def test_heldout_shape_pose_material_change_and_generator_bound(layout):
    a=make_scene(layout);b=perturb_scene(a,101)
    assert all(p.displacement_upper_bound<=.05 for p in b)
    assert any(not np.allclose(x.size,y.size) for x,y in zip(a,b))
    assert any(x.material!=y.material for x,y in zip(a,b))
    for p in b:np.testing.assert_allclose(p.orientation.T@p.orientation,np.eye(3),atol=1e-12)


def test_negative_control_has_independent_motion_without_depth_seam():
    objects=make_scene('coplanar_split');a,b=objects[2:]
    assert a.center[2]==b.center[2]
    assert np.isclose(a.center[0]+a.size[0],b.center[0]-b.size[0])
    assert np.linalg.norm(a.translation-b.translation)>.02


def test_maximum_motion_metric_is_in_millimetres(small):
    from scene_flow_refinement.primitive_metrics import score_primitive,region_masks
    from scene_flow_refinement.metrics import lift_observed
    r=call(small,Config(iterations=3,later_iterations=1,outer_iterations=1))
    regions=region_masks(small,lift_observed(small['depth0'],small['depth1'],small['flow_observed'],small['intrinsics']))
    row,_=score_primitive(small,r,regions,[])
    expected=1000*np.linalg.norm(r['scene_flow'],axis=-1).max()
    assert abs(row['max_estimated_motion_mm']-expected)<1e-12
