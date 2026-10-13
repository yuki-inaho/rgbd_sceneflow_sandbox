from __future__ import annotations
import ctypes as ct
from dataclasses import replace
import inspect
import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar
from scene_flow_refinement import Config,refine
from scene_flow_refinement._ffi import library,_DOUBLE
from scene_flow_refinement.synthetic import generate
from scene_flow_refinement.metrics import lift_observed,project_flow

@pytest.fixture(scope="module")
def sample():
    return generate(seed=713,height=40,width=56)

@pytest.fixture(scope="module")
def result(sample):
    return refine(sample["rgb0"],sample["rgb1"],sample["depth0"],sample["depth1"],
                  sample["flow_observed"],sample["intrinsics"],Config(iterations=350,outer_iterations=2))


def call(data,cfg=None):
    return refine(data["rgb0"],data["rgb1"],data["depth0"],data["depth1"],
                  data["flow_observed"],data["intrinsics"],cfg or Config(iterations=40,outer_iterations=1))


def test_native_adjoint(result):
    assert result["stats"]["adjoint_error"]<1e-11


def test_hard_displacement_bound(result):
    assert np.max(np.linalg.norm(result["scene_flow"],axis=-1))<=.05+1e-12


def test_depth_trust_box(sample,result):
    for t in (0,1):
        assert np.max(np.abs(result[f"depth{t}"]-sample[f"depth{t}"]))<=4*.003+1e-12


def test_scene_flow_improves_on_noisy_lifting(sample,result):
    raw=lift_observed(sample["depth0"],sample["depth1"],sample["flow_observed"],sample["intrinsics"])
    mask=sample["gt_visible"]&np.isfinite(raw).all(axis=-1)
    raw_error=np.linalg.norm(raw-sample["gt_scene_flow"],axis=-1)[mask].mean()
    new_error=np.linalg.norm(result["scene_flow"]-sample["gt_scene_flow"],axis=-1)[mask].mean()
    assert new_error<.25*raw_error


def test_depth_noise_reduction(sample,result):
    for t in (0,1):
        old=np.mean((sample[f"depth{t}"]-sample[f"gt_depth{t}"])**2)
        new=np.mean((result[f"depth{t}"]-sample[f"gt_depth{t}"])**2)
        assert new<old


def test_output_flow_is_exact_reprojection(sample,result):
    expected=project_flow(result["depth0"],result["scene_flow"],sample["intrinsics"])
    np.testing.assert_allclose(result["flow"],expected,atol=3e-14,rtol=1e-12)


def test_soft_geometry_residual_is_reported(result):
    values=result["geometry_residual"][result["support"]>0]
    assert 0<float(np.mean(values))<.002


def test_accepted_objective_never_increases(result):
    assert np.max(np.diff(result["history"][:,3]))<1e-7


def test_determinism(sample):
    cfg=Config(iterations=50,outer_iterations=1)
    a=call(sample,cfg);b=call(sample,cfg)
    np.testing.assert_array_equal(a["scene_flow"],b["scene_flow"])


def test_zero_motion_fixed_point():
    h,w=12,16;y,x=np.mgrid[:h,:w]
    rgb=np.stack((x/(w+1),y/(h+1),(x+y)/(h+w+1)),axis=-1)
    depth=np.ones((h,w));flow=np.zeros((h,w,2))
    r=refine(rgb,rgb,depth,depth,flow,np.array([60.,60.,7.5,5.5]),Config(iterations=80,outer_iterations=2))
    assert np.max(np.abs(r["scene_flow"]))<1e-12
    np.testing.assert_allclose(r["depth0"],depth,atol=1e-12)


def test_all_missing_depth_is_unsupported(sample):
    d={**sample,"depth0":np.full_like(sample["depth0"],np.nan),"depth1":np.full_like(sample["depth1"],np.nan)}
    r=call(d)
    assert not np.any(r["support"])
    assert np.all(np.isfinite(r["depth0"]))
    assert np.all(r["depth0"]>0)
    assert np.isnan(r["match_flow"]).all()
    assert np.isnan(r["geometry_residual"]).all()


def test_missing_initialization_obeys_custom_depth_range(sample):
    d={**sample,"depth0":np.full_like(sample["depth0"],np.nan),"depth1":np.full_like(sample["depth1"],np.nan)}
    r=call(d,Config(iterations=20,outer_iterations=1,min_depth=2,max_depth=3))
    assert np.min(r["depth0"])>=2 and np.max(r["depth1"])<=3

@pytest.mark.parametrize("key,value",[("sigma_depth",-1),("max_motion",0),("iterations",0),
                                      ("max_motion",.2),("sigma_geometry",float("nan"))])
def test_invalid_configuration(sample,key,value):
    with pytest.raises(ValueError):call(sample,replace(Config(iterations=10,outer_iterations=1),**{key:value}))


def test_shape_validation(sample):
    with pytest.raises(ValueError):call({**sample,"rgb1":sample["rgb1"][:,:-1]})


def test_rgb_range_validation(sample):
    with pytest.raises(ValueError):call({**sample,"rgb0":sample["rgb0"]*255})


def test_depth_range_validation(sample):
    with pytest.raises(ValueError):call({**sample,"depth0":sample["depth0"]*100})


def test_nonfinite_optical_flow_does_not_crash(sample):
    r=call({**sample,"flow_observed":np.full_like(sample["flow_observed"],np.nan)})
    assert np.isfinite(r["scene_flow"]).all()


def test_no_gt_parameter():
    assert set(inspect.signature(refine).parameters)=={"rgb0","rgb1","depth0","depth1","flow","intrinsics","config"}


def test_ground_truth_projection(sample):
    f=project_flow(sample["gt_depth0"],sample["gt_scene_flow"],sample["intrinsics"])
    np.testing.assert_allclose(f,sample["gt_flow"],atol=2e-14,rtol=1e-10)


def test_huber_prox_against_scalar_optimization():
    rng=np.random.default_rng(57);lib=library()
    for _ in range(30):
        a,y=rng.normal(size=2);eta=float(rng.uniform(.1,2));tau=float(rng.uniform(.05,1));k=2.
        def objective(v):
            residual=abs((v-y)/eta)
            loss=.5*residual**2 if residual<=k else k*(residual-.5*k)
            return .5*(v-a)**2+tau*loss
        reference=minimize_scalar(objective,bounds=(min(a,y)-10,max(a,y)+10),method="bounded",
                                  options={"xatol":1e-12}).x
        actual=lib.sfr_huber_prox(a,y,eta,k,tau)
        assert abs(actual-reference)<2e-7


def test_ray_segment_distance_against_box_optimization():
    rng=np.random.default_rng(61);lib=library()
    for _ in range(30):
        r0=np.array([*rng.uniform(-.5,.5,2),1.]);r1=np.array([*rng.uniform(-.5,.5,2),1.])
        l0,l1=rng.uniform(.5,1.5,2);u0=l0+.03;u1=l1+.03
        actual=lib.sfr_ray_distance(r0.ctypes.data_as(_DOUBLE),r1.ctypes.data_as(_DOUBLE),l0,u0,l1,u1)
        ref=minimize(lambda x:np.sum((x[1]*r1-x[0]*r0)**2),[(l0+u0)/2,(l1+u1)/2],
                     bounds=[(l0,u0),(l1,u1)],method="L-BFGS-B",options={"ftol":1e-15,"gtol":1e-12})
        assert abs(actual-np.sqrt(ref.fun))<1e-7


def regularizer(u,aux,guide,cut):
    h,w,channels=u.shape
    arrays=[np.ascontiguousarray(a,dtype=np.float64) for a in (u,aux,guide)]
    return library().sfr_regularizer_energy(h,w,channels,*(a.ctypes.data_as(_DOUBLE) for a in arrays),.06,1.2,4.,int(cut))


def test_affine_field_has_zero_second_order_regularizer():
    h,w,c=8,11,3;y,x=np.mgrid[:h,:w]
    B=np.array([[.1,.05],[-.02,.04],[.03,-.01]])
    u=np.einsum("...j,cj->...c",np.stack((x,y),axis=-1),B)+np.array([1,2,3])
    aux=np.broadcast_to(B.T,(h,w,2,c)).copy()
    assert regularizer(u,aux,np.ones((h,w)),True)<1e-12


def test_boundary_cuts_both_first_and_second_order_terms():
    h,w,c=8,10,3;y,x=np.mgrid[:h,:w];right=x>=5
    guide=np.where(right,1.4,1.)
    B0=np.array([[.1,.02],[.03,-.04],[-.02,.01]])
    B1=np.array([[-.1,.01],[.07,.03],[.04,-.03]])
    u=np.empty((h,w,c));aux=np.empty((h,w,2,c))
    for mask,B,shift in ((~right,B0,0),(right,B1,3)):
        u[mask]=np.stack((x,y),axis=-1)[mask]@B.T+shift
        aux[mask]=B.T
    assert regularizer(u,aux,guide,True)<1e-12
    assert regularizer(u,aux,guide,False)>1
