from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest
from scene_flow_refinement import Config,refine
from scene_flow_refinement.primitives import Primitive,make_scene,render_queries,render_frame,generate_primitives,image_flow
from scene_flow_refinement.synthetic import rotation
from scene_flow_refinement.metrics import project_flow


def primitive(kind,size=(.2,.2,.2),orientation=None):
    return Primitive('test',kind,size,np.array([0.,0.,1.]),np.eye(3) if orientation is None else orientation,
                     np.zeros(3),np.eye(3),17)


def central_ray(obj):
    return render_queries(np.array([[0.,0.]]),np.array([100.,100.,0.,0.]),[obj],shading=False)[0]


def test_sphere_center_intersection():
    hit=central_ray(primitive(1))
    assert abs(hit[0]-.8)<1e-12
    np.testing.assert_allclose(hit[6:9],[0,0,-1],atol=1e-12)


def test_box_center_intersection():
    hit=central_ray(primitive(2))
    assert abs(hit[0]-.8)<1e-12
    np.testing.assert_allclose(hit[3:6],[0,0,-.2],atol=1e-12)


def test_cylinder_side_intersection():
    assert abs(central_ray(primitive(3))[0]-.8)<1e-12


def test_cylinder_cap_intersection():
    hit=central_ray(primitive(3,(.2,.35,.2),rotation('x',np.pi/2)))
    assert abs(hit[0]-.65)<1e-12
    np.testing.assert_allclose(hit[6:9],[0,0,-1],atol=1e-12)


def test_ellipsoid_intersection_is_on_implicit_surface():
    obj=primitive(1,(.25,.12,.17),rotation('y',.4))
    query=np.array([[0.,0.],[4,5],[-4,3]])
    hit=render_queries(query,np.array([100,100,0,0]),[obj],shading=False)
    np.testing.assert_allclose(np.sum((hit[:,3:6]/obj.size)**2,axis=1),1,atol=2e-14)


def test_nearest_surface_wins():
    front=primitive(1)
    rear=replace(primitive(2),center=np.array([0,0,1.5]))
    a=render_queries(np.zeros((1,2)),np.array([100,100,0,0]),[rear,front],shading=False)
    assert a[0,2]==1 and abs(a[0,0]-.8)<1e-12


@pytest.fixture(scope='module')
def data():
    return generate_primitives('occlusion',123,48,64,'dis')


def test_projection_agrees_with_material_ground_truth(data):
    d,_=data
    np.testing.assert_allclose(project_flow(d['gt_depth0'],d['gt_scene_flow'],d['intrinsics']),d['gt_flow'],atol=3e-12)


def test_visibility_is_a_partition(data):
    d,_=data
    assert np.all(d['gt_visible'].astype(int)+d['gt_occluded'].astype(int)+d['gt_outside'].astype(int)==1)
    assert d['gt_occluded'].any()


def test_renderer_returns_optical_depth_not_range(data):
    d,_=data
    f=render_frame(make_scene('occlusion'),48,64,supersample=1)
    np.testing.assert_allclose(f['xyz'][...,2],d['gt_depth0'],atol=1e-14)


def test_surface_normals_are_unit(data):
    d,_=data
    np.testing.assert_allclose(np.linalg.norm(d['gt_normal0'],axis=-1),1,atol=1e-13)


def test_grayscale_channels_are_identical(data):
    d,_=data
    np.testing.assert_array_equal(d['rgb0'][...,0],d['gray0'])
    np.testing.assert_array_equal(d['rgb0'][...,1],d['gray0'])
    np.testing.assert_array_equal(d['rgb1'][...,2],d['gray1'])


def test_supersampling_does_not_average_depth():
    objects=make_scene('gallery')
    one=render_frame(objects,36,48,supersample=1)
    two=render_frame(objects,36,48,supersample=2)
    np.testing.assert_array_equal(one['depth'],two['depth'])
    np.testing.assert_array_equal(one['labels'],two['labels'])
    assert np.mean(abs(one['gray']-two['gray']))>1e-5


def test_pure_sphere_rotation_does_not_change_depth():
    obj=make_scene('rotation')
    a=render_frame(obj,48,64,0.)
    b=render_frame(obj,48,64,1.)
    np.testing.assert_allclose(a['depth'],b['depth'],atol=1e-12)


def test_texture_makes_pure_sphere_rotation_observable():
    obj=make_scene('rotation')
    a=render_frame(obj,48,64,0.,contrast=1)
    b=render_frame(obj,48,64,1.,contrast=1)
    m=a['labels']==2
    assert np.mean(abs(a['gray'][m]-b['gray'][m]))>.005
    a0=render_frame(obj,48,64,0.,contrast=0)
    b0=render_frame(obj,48,64,1.,contrast=0)
    np.testing.assert_allclose(a0['gray'],b0['gray'],atol=1e-12)


@pytest.mark.parametrize('layout',['gallery','occlusion','curved','rotation'])
def test_entire_primitive_motion_has_safe_conservative_bound(layout):
    objects=make_scene(layout)
    assert all(o.displacement_upper_bound<.05 for o in objects)


def test_dis_recomputed_from_images_only(data):
    d,_=data
    forward,backward=image_flow(d['gray0'],d['gray1'])
    np.testing.assert_array_equal(forward,d['flow_dis'])
    np.testing.assert_array_equal(backward,d['flow_dis_backward'])


def test_gray_interface_equals_replicated_rgb(data):
    d,_=data
    args=(d['depth0'],d['depth1'],d['flow_observed'],d['intrinsics'])
    c=Config(iterations=20,outer_iterations=1,reciprocal_check=False)
    a=refine(d['gray0'],d['gray1'],*args,c)
    b=refine(d['rgb0'],d['rgb1'],*args,c)
    np.testing.assert_array_equal(a['scene_flow'],b['scene_flow'])


def test_no_bound_disables_candidate_gate_and_projection():
    z=np.ones((8,10));gray=np.full_like(z,.5);flow=np.zeros((8,10,2));intr=np.array([40.,40.,4.5,3.5])
    c=Config(iterations=350,outer_iterations=1,patch_radius=0,reciprocal_check=False,
             centered_photo=False,diverse_candidates=False)
    with_bound=refine(gray,gray,z,z+.20,flow,intr,c)
    no_bound=refine(gray,gray,z,z+.20,flow,intr,replace(c,enforce_bound=False))
    assert not np.any(with_bound['support'])
    assert np.all(no_bound['support'])
    assert no_bound['stats']['geometry_rejected_candidates']==0
    assert np.max(np.linalg.norm(no_bound['scene_flow'],axis=-1))>.1


def test_zero_weight_missing_neighbors_do_not_invalidate_integer_sample():
    z=np.full((3,3),np.nan);z[0,0]=1
    gray=np.full((3,3),.5);flow=np.zeros((3,3,2));intr=np.array([30.,30.,1.,1.])
    c=Config(iterations=20,outer_iterations=1,patch_radius=0,reciprocal_check=False)
    r=refine(gray,gray,z,z,flow,intr,c)
    assert r['support'][0,0]==1
    assert np.isfinite(r['geometry_residual'][0,0])


def test_centered_photo_handles_additive_brightness_offset():
    rng=np.random.default_rng(10)
    gray=rng.uniform(.1,.6,(10,12));z=np.ones_like(gray);f=np.zeros((*gray.shape,2));intr=np.array([40.,40.,5.5,4.5])
    cfg=Config(iterations=30,outer_iterations=1,patch_radius=1,reciprocal_check=False,photo_reject=.02)
    centered=refine(gray,gray+.2,z,z,f,intr,cfg)
    uncentered=refine(gray,gray+.2,z,z,f,intr,replace(cfg,centered_photo=False))
    assert centered['support'].mean()>.95
    assert uncentered['support'].mean()<.1


def test_invalid_render_input_is_rejected():
    with pytest.raises(ValueError):render_frame(make_scene(),24,32,supersample=0)
    with pytest.raises(ValueError):render_queries(np.zeros((2,3)),np.array([1,1,0,0]),make_scene())
    with pytest.raises(ValueError):render_queries(np.zeros((2,2)),np.array([0,1,0,0]),make_scene())
    with pytest.raises(ValueError):make_scene('unknown')


@pytest.mark.parametrize('layout',['gallery','occlusion','curved'])
def test_sampled_primitive_interiors_do_not_interpenetrate(layout):
    # A deterministic sampled diagnostic, not a proof of collision freedom.
    rng=np.random.default_rng(499);objects=make_scene(layout)[2:]
    for t in np.linspace(0,1,11):
        for obj in objects:
            local=rng.uniform(-1,1,(2000,3))*obj.size
            if obj.kind==1:local=local[np.sum((local/obj.size)**2,axis=1)<1]
            if obj.kind==3:local=local[local[:,0]**2+local[:,2]**2<obj.size[0]**2]
            c,R=obj.pose(t);world=local@R.T+c
            assert np.max(world[:,1])<.285
            for other in objects:
                if obj is other:continue
                c,R=other.pose(t);p=(world-c)@R
                if other.kind==1:inside=np.sum((p/other.size)**2,axis=1)<1
                elif other.kind==2:inside=np.all(abs(p)<other.size,axis=1)
                else:inside=(p[:,0]**2+p[:,2]**2<other.size[0]**2)&(abs(p[:,1])<other.size[1])
                assert not np.any(inside)


def test_no_image_ablation_does_not_depend_on_image_values():
    rng=np.random.default_rng(188)
    z=np.ones((8,10));f=np.zeros((8,10,2));intr=np.array([40.,40.,4.5,3.5])
    cfg=Config(iterations=20,outer_iterations=1,image_weight=0,reciprocal_check=False,photo_reject=.001)
    a=refine(np.full_like(z,.4),np.full_like(z,.4),z,z,f,intr,cfg)
    b=refine(rng.uniform(size=z.shape),rng.uniform(size=z.shape),z,z,f,intr,cfg)
    np.testing.assert_array_equal(a['scene_flow'],b['scene_flow'])
    np.testing.assert_array_equal(a['support'],b['support'])


def test_flow_source_does_not_change_gray_or_depth_observations():
    a,_=generate_primitives('gallery',85,36,48,'dis')
    b,_=generate_primitives('gallery',85,36,48,'corrupted_gt')
    for key in ('gray0','gray1','depth0','depth1'):
        np.testing.assert_array_equal(a[key],b[key])
    assert np.mean(abs(a['flow_observed']-b['flow_observed']))>.1


def test_missing_mask_does_not_change_surviving_depth_measurements():
    a,_=generate_primitives('gallery',86,36,48,'dis')
    b,_=generate_primitives('gallery',86,36,48,'dis',missing_fraction=.08)
    for t in (0,1):
        valid=np.isfinite(b[f'depth{t}'])
        np.testing.assert_array_equal(a[f'depth{t}'][valid],b[f'depth{t}'][valid])


def test_missing_depth_is_explicitly_marked_unsupported(data):
    d,_=data;z=d['depth0'].copy();z[10,12]=np.nan
    cfg=Config(iterations=20,outer_iterations=1,reciprocal_check=False)
    r=refine(d['gray0'],d['gray1'],z,d['depth1'],d['flow_observed'],d['intrinsics'],cfg)
    assert not r['depth0_supported'][10,12]
    assert r['depth1_supported'].all()
    assert not r['support'][10,12]


def test_ray_miss_has_defined_sentinel_values():
    hit=render_queries(np.array([[1000.,1000.]]),np.array([100.,100.,0.,0.]),[primitive(1)],shading=False)[0]
    assert np.isposinf(hit[0]) and hit[2]==-1 and hit[1]==0


def test_static_border_is_not_mislabeled_outside_by_roundoff(data):
    d,_=data
    assert not np.any(d["gt_outside"] & ~d["gt_moving"])
