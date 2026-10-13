"""Analytic rigid-plane renderer with continuous-coordinate visibility ground truth."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

@dataclass(frozen=True)
class Plane:
    center0: Array
    rotation0: Array
    center1: Array
    rotation1: Array
    half_extent: tuple[float, float]
    texture_id: int


def rotation(axis: str, angle: float) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)
    if axis == "y":
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)
    if axis == "z":
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)
    raise ValueError("axis must be x, y, or z")


def make_planes(motion_scale: float = 1.0) -> list[Plane]:
    identity = np.eye(3)
    background = Plane(np.array([0.,0.,1.45]), identity, np.array([0.,0.,1.45]), identity, (10.,10.), 0)
    c0 = np.array([-0.16,-0.025,0.88])
    r0 = rotation("y",0.17) @ rotation("x",-0.08)
    first = Plane(c0,r0,c0+motion_scale*np.array([0.028,0.006,0.007]),
                  rotation("z",0.045*motion_scale) @ r0,(0.155,0.18),1)
    c1 = np.array([0.185,0.015,1.08])
    r1 = rotation("y",-0.14) @ rotation("x",0.12)
    second = Plane(c1,r1,c1+motion_scale*np.array([-0.018,-0.009,-0.008]),
                   rotation("z",-0.035*motion_scale) @ r1,(0.155,0.165),2)
    return [background, first, second]


def texture(uv: Array, identity: int, weak: bool = False) -> Array:
    """Band-limited material coordinates, unchanged by rigid motion."""
    rng = np.random.default_rng(918+identity)
    xy = uv[..., :2]
    result = np.empty((*xy.shape[:-1], 3))
    for channel in range(3):
        frequencies = rng.uniform(-22,22,(12,2))  # radians/metre after factor below
        phase = rng.uniform(-np.pi,np.pi,12)
        values = np.sin(np.einsum("...j,kj->...k",xy,frequencies)*3.0+phase)
        result[...,channel] = 0.5+0.18*np.sum(values,axis=-1)/np.sqrt(12)
    if weak and identity == 1:
        result = 0.5 + 0.02*(result-0.5)
    return np.clip(result,0.02,0.98)


def render_at(pixels: Array, intrinsics: Array, planes: list[Plane], time: int,
              weak: bool = False) -> tuple[Array, Array, NDArray[np.int64], Array]:
    """Intersect camera rays with bounded planes; select nearest positive intersection."""
    fx,fy,cx,cy = intrinsics
    rays = np.stack(((pixels[...,0]-cx)/fx,(pixels[...,1]-cy)/fy,np.ones(pixels.shape[:-1])),axis=-1)
    depth = np.full(pixels.shape[:-1],np.inf)
    labels = np.full(pixels.shape[:-1],-1,dtype=np.int64)
    local = np.zeros_like(rays)
    for index, plane in enumerate(planes):
        c = plane.center0 if time == 0 else plane.center1
        r = plane.rotation0 if time == 0 else plane.rotation1
        normal = r[:,2]
        denominator = rays @ normal
        z = np.divide(c@normal,denominator,out=np.full_like(depth,np.inf),where=np.abs(denominator)>1e-12)
        xyz = z[...,None]*rays
        uv = (xyz-c) @ r
        valid = ((z>0)&(z<depth)&(np.abs(uv[...,0])<=plane.half_extent[0])&
                 (np.abs(uv[...,1])<=plane.half_extent[1]))
        depth[valid] = z[valid]
        labels[valid] = index
        local[valid] = uv[valid]
    rgb = np.zeros_like(rays)
    for index, plane in enumerate(planes):
        mask = labels == index
        rgb[mask] = texture(local[mask],plane.texture_id,weak)
    return rgb,depth,labels,local


def generate(seed: int = 0, height: int = 96, width: int = 128,
             sigma_depth: float = 0.003, sigma_flow: float = 0.7,
             outlier_fraction: float = 0.15, weak_texture: bool = False,
             motion_scale: float = 1.0) -> dict[str, Array]:
    if min(height,width)<12:
        raise ValueError("synthetic scene needs at least 12x12 pixels")
    if not 0<=outlier_fraction<=1:
        raise ValueError("outlier_fraction must be in [0,1]")
    rng = np.random.default_rng(seed)
    intr = np.array([140.*width/128,140.*width/128,(width-1)/2,(height-1)/2])
    yy,xx = np.mgrid[:height,:width]
    pixels = np.stack((xx,yy),axis=-1).astype(float)
    planes = make_planes(motion_scale)
    rgb0,d0,l0,local0 = render_at(pixels,intr,planes,0,weak_texture)
    rgb1,d1,l1,_ = render_at(pixels,intr,planes,1,weak_texture)
    rays = np.stack(((xx-intr[2])/intr[0],(yy-intr[3])/intr[1],np.ones_like(xx)),axis=-1)
    xyz0 = d0[...,None]*rays
    xyz1 = np.empty_like(xyz0)
    for index,plane in enumerate(planes):
        mask = l0 == index
        xyz1[mask] = local0[mask] @ plane.rotation1.T + plane.center1
    scene = xyz1-xyz0
    q = np.stack((intr[0]*xyz1[...,0]/xyz1[...,2]+intr[2],
                  intr[1]*xyz1[...,1]/xyz1[...,2]+intr[3]),axis=-1)
    flow = q-pixels
    _,target_z,target_label,_ = render_at(q,intr,planes,1,weak_texture)
    visible = ((q[...,0]>=0)&(q[...,0]<width-1)&(q[...,1]>=0)&(q[...,1]<height-1)&
               (target_label==l0)&(np.abs(target_z-xyz1[...,2])<1e-8))
    noisy_flow = flow+rng.normal(0,sigma_flow,flow.shape)
    outlier = rng.random((height,width))<outlier_fraction
    # A coherent wrong-flow region is included; its mask is NEVER passed to refine().
    block = ((xx>width*.18)&(xx<width*.31)&(yy>height*.38)&(yy<height*.56))
    outlier |= block
    noisy_flow[outlier] += rng.uniform(-12,12,(int(outlier.sum()),2))
    noisy_flow[block] = flow[block]+np.array([9.,-6.])
    return {
        "rgb0":np.clip(rgb0+rng.normal(0,0.004,rgb0.shape),0,1),
        "rgb1":np.clip(rgb1+rng.normal(0,0.004,rgb1.shape),0,1),
        "depth0":d0+rng.normal(0,sigma_depth,d0.shape),
        "depth1":d1+rng.normal(0,sigma_depth,d1.shape),
        "flow_observed":noisy_flow,"intrinsics":intr,
        "gt_depth0":d0,"gt_depth1":d1,"gt_flow":flow,"gt_scene_flow":scene,
        "gt_visible":visible,"gt_labels0":l0,"gt_labels1":l1,
        "injected_outlier":outlier,"coherent_outlier":block,
    }
