"""Textured 3-D primitive scenes, analytic C++ ray tracing, and exact material flow.

All distances are metres, all depth values are optical-axis z, all image coordinates
refer to pixel centres. Ground truth is separated from the observation dictionary.
"""
from __future__ import annotations
from dataclasses import dataclass
import ctypes as ct
from typing import Literal
import numpy as np
from ._ffi import library, _DOUBLE
from .synthetic import rotation

@dataclass(frozen=True)
class Primitive:
    name: str
    kind: int
    size: tuple[float, float, float]
    center: np.ndarray
    orientation: np.ndarray
    translation: np.ndarray
    rotation_delta: np.ndarray
    material: int

    def pose(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        from scipy.spatial.transform import Rotation
        delta = Rotation.from_rotvec(Rotation.from_matrix(self.rotation_delta).as_rotvec()*time).as_matrix()
        return self.center+time*self.translation, delta@self.orientation

    def record(self, time: float) -> np.ndarray:
        center, orientation = self.pose(time)
        return np.r_[self.kind, self.size, center, orientation.ravel(), self.material]

    @property
    def moving(self) -> bool:
        return bool(np.linalg.norm(self.translation)>1e-12 or np.linalg.norm(self.rotation_delta-np.eye(3))>1e-12)

    @property
    def displacement_upper_bound(self) -> float:
        from scipy.spatial.transform import Rotation
        radius = max(self.size) if self.kind==1 else float(np.linalg.norm(self.size))
        angle = Rotation.from_matrix(self.rotation_delta).magnitude()
        return float(np.linalg.norm(self.translation)+2*radius*np.sin(angle/2))


def make_scene(layout: str = "gallery") -> list[Primitive]:
    """Three different layouts and motions; not merely different noise realisations."""
    I=np.eye(3)
    def p(name,kind,size,c,t=(0.,0.,0.),angles=(0.,0.,0.),turn=(0.,0.,0.),mat=1):
        R=rotation("z",angles[2])@rotation("y",angles[1])@rotation("x",angles[0])
        D=rotation("z",turn[2])@rotation("y",turn[1])@rotation("x",turn[0])
        return Primitive(name,kind,tuple(size),np.asarray(c,float),R,np.asarray(t,float),D,mat)
    base=[p("wall",0,(5,5,.001),(0,0,2.10),mat=11),
          p("floor",0,(5,5,.001),(0,.285,1.30),angles=(np.pi/2,0,0),mat=27)]
    if layout=="gallery":
        objects=[p("sphere",1,(.16,)*3,(-.27,.075,1.08),(.024,-.008,.005),turn=(0,.055,0),mat=101),
                 p("cube",2,(.15,)*3,(.025,.067,1.55),(-.016,.002,-.008),angles=(.20,.42,-.10),turn=(0,-.06,.025),mat=205),
                 p("cylinder",3,(.10,.17,.10),(.30,.068,1.23),(-.011,-.005,.007),angles=(0,0,-.10),turn=(.04,0,-.035),mat=302),
                 p("ellipsoid",1,(.15,.085,.10),(-.05,-.25,1.64),(.012,.004,-.005),angles=(.15,.25,.3),turn=(.04,0,.035),mat=418)]
    elif layout=="occlusion":
        objects=[p("sphere",1,(.18,)*3,(-.09,.075,1.02),(.031,-.004,.004),turn=(0,.045,0),mat=153),
                 p("cube",2,(.16,.14,.145),(.115,.04,1.60),(-.024,.003,-.006),angles=(.18,-.4,.08),turn=(0,.055,-.015),mat=267),
                 p("cylinder",3,(.092,.175,.092),(-.32,.068,1.48),(.008,-.005,-.006),angles=(.10,0,.06),turn=(0,.04,.025),mat=383),
                 p("small_sphere",1,(.10,)*3,(.335,-.115,1.27),(-.014,.010,-.002),turn=(0,0,-.06),mat=477)]
    elif layout=="curved":
        objects=[p("ellipsoid",1,(.20,.115,.135),(-.19,.07,1.13),(.010,-.007,-.007),angles=(.2,.35,-.35),turn=(.06,-.045,0),mat=517),
                 p("sphere",1,(.145,)*3,(.235,.07,1.20),(-.021,.002,.010),turn=(.035,.045,0),mat=631),
                 p("cube",2,(.10,.12,.095),(.035,-.145,1.54),(.006,.012,-.004),angles=(.3,-.3,.25),turn=(.02,.045,.025),mat=743),
                 p("cylinder",3,(.06,.16,.06),(-.35,-.10,1.47),(.015,.002,-.004),angles=(.6,0,.35),turn=(.025,0,.04),mat=857)]
    elif layout=="coplanar_split":
        # Two contacting coplanar surfaces move independently. Their common
        # depth component violates the added one-rigid-pose assumption.
        objects=[p("left_independent",0,(.17,.14,.001),(-.17,0,1.30),(0,.012,0),mat=1601),
                 p("right_independent",0,(.17,.14,.001),(.17,0,1.30),(0,-.012,0),mat=1711)]
    elif layout=="rotation":
        objects=[p("rotating_sphere",1,(.22,)*3,(0,.045,1.10),turn=(0,.06,.14),mat=997)]
    else:
        raise ValueError(f"unknown layout {layout!r}; expected gallery, occlusion, curved or rotation")
    return base+objects


def perturb_scene(objects: list[Primitive], scene_seed: int) -> list[Primitive]:
    """Observable layout variation, independent of the noise seed and inference.

    Size, pose, texture and motion change. Bounds are enforced analytically
    on the generator parameters, never estimated using a predicted result.
    """
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(scene_seed)
    result=[]
    for obj in objects:
        if not obj.moving:
            result.append(obj)
            continue
        size=tuple(np.asarray(obj.size)*rng.uniform(.88,1.12))
        center=obj.center+rng.uniform([-.025,-.020,-.06],[.025,.020,.06])
        orientation=Rotation.from_rotvec(rng.uniform(-.12,.12,3)).as_matrix()@obj.orientation
        translation=obj.translation+rng.uniform(-.005,.005,3)
        delta=Rotation.from_rotvec(Rotation.from_matrix(obj.rotation_delta).as_rotvec()*rng.uniform(.8,1.1)).as_matrix()
        candidate=replace(obj,size=size,center=center,orientation=orientation,translation=translation,
                          rotation_delta=delta,material=int(rng.integers(2000,10000)))
        factor=min(1.,.048/candidate.displacement_upper_bound)
        if factor<1:
            # Uniformly contract motion; sin(theta/2) <= theta/2 gives safety
            # by an additional small margin on the generated bound.
            candidate=replace(candidate,translation=translation*factor*.98,
                rotation_delta=Rotation.from_rotvec(Rotation.from_matrix(delta).as_rotvec()*factor*.98).as_matrix())
        if candidate.displacement_upper_bound>.05:
            raise ValueError("generated motion exceeds requested bound")
        result.append(candidate)
    return result


def render_queries(pixels: np.ndarray, intrinsics: np.ndarray, objects: list[Primitive],
                   time: float = 0., contrast: float = 1., shadows: bool = True,
                   shading: bool = True) -> np.ndarray:
    """Return [...,12]: z, gray, object id, local xyz, world normal xyz, world xyz.

    A ray miss returns z=+inf, gray=0, id=-1, and zero local/normal/world vectors.
    """
    pixels=np.asarray(pixels,dtype=np.float64)
    if pixels.ndim<2 or pixels.shape[-1]!=2 or not np.isfinite(pixels).all():
        raise ValueError("pixels must be finite [...,2]")
    intr=np.ascontiguousarray(intrinsics,dtype=np.float64)
    if intr.shape!=(4,) or not np.isfinite(intr).all() or np.any(intr[:2]<=0):
        raise ValueError("intrinsics must be finite (fx,fy,cx,cy), fx,fy>0")
    if not objects or not np.isfinite(contrast) or contrast<0:
        raise ValueError("a nonempty scene and a finite nonnegative contrast are required")
    records=np.ascontiguousarray([obj.record(time) for obj in objects],dtype=np.float64)
    if not np.isfinite(records).all() or np.any(records[:,1:4]<=0):
        raise ValueError("primitive parameters must be finite with positive sizes")
    for obj in objects:
        R=obj.orientation
        if R.shape!=(3,3) or not np.allclose(R.T@R,np.eye(3),atol=1e-10) or np.linalg.det(R)<0:
            raise ValueError("orientation must be a proper rotation")
    flat=np.ascontiguousarray(pixels.reshape(-1,2))
    result=np.empty((len(flat),12),dtype=np.float64)
    lib=library()
    lib.sfr_render.argtypes=[ct.c_int,_DOUBLE,_DOUBLE,ct.c_int,_DOUBLE,ct.c_double,ct.c_int,ct.c_int,_DOUBLE]
    lib.sfr_render.restype=ct.c_int
    code=lib.sfr_render(len(flat),flat.ctypes.data_as(_DOUBLE),intr.ctypes.data_as(_DOUBLE),len(records),
                        records.ctypes.data_as(_DOUBLE),contrast,int(shadows),int(shading),result.ctypes.data_as(_DOUBLE))
    if code:
        raise RuntimeError(f"native ray tracer failed ({code})")
    return result.reshape(*pixels.shape[:-1],12)


def camera(height: int, width: int) -> tuple[np.ndarray,np.ndarray]:
    if min(height,width)<12:
        raise ValueError("minimum render resolution is 12 by 12")
    yy,xx=np.mgrid[:height,:width]
    return np.array([1.13*width,1.13*width,(width-1)/2,(height-1)/2]),np.stack((xx,yy),axis=-1).astype(float)


def render_frame(objects: list[Primitive], height: int, width: int, time: float = 0.,
                 contrast: float = 1., supersample: int = 2, shadows: bool = True) -> dict[str,np.ndarray]:
    """Depth/labels stay at the centre ray; only grayscale is supersampled."""
    if supersample<1 or supersample>4:
        raise ValueError("supersample must be between 1 and 4")
    intr,pixels=camera(height,width)
    center=render_queries(pixels,intr,objects,time,contrast,shadows,shading=supersample==1)
    if supersample==1:
        gray=center[...,1]
    else:
        gray=np.zeros((height,width))
        for oy in (np.arange(supersample)+.5)/supersample-.5:
            for ox in (np.arange(supersample)+.5)/supersample-.5:
                gray+=render_queries(pixels+np.array([ox,oy]),intr,objects,time,contrast,shadows)[...,1]/supersample**2
    return {"gray":gray,"depth":center[...,0],"labels":center[...,2].astype(np.int32),
            "local":center[...,3:6],"normal":center[...,6:9],"xyz":center[...,9:12],"intrinsics":intr}


def image_flow(gray0: np.ndarray, gray1: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Image-only OpenCV DIS (native C++), deterministic single-thread execution."""
    import cv2
    cv2.setNumThreads(1)
    a,b=[np.ascontiguousarray(np.clip(np.rint(x*255),0,255),dtype=np.uint8) for x in (gray0,gray1)]
    def estimate(x,y):
        dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(0)
        dis.setPatchSize(8)
        dis.setPatchStride(3)
        return dis.calc(x,y,None).astype(np.float64)
    return estimate(a,b),estimate(b,a)


def generate_primitives(layout: str = "gallery", seed: int = 0, height: int = 120, width: int = 160,
                        flow_source: Literal["corrupted_gt","dis"] = "dis", contrast: float = 1.,
                        illumination: bool = False, missing_fraction: float = 0.,
                        sigma_depth: float = .003, sigma_flow: float = .7,
                        scene_seed: int | None = None) -> tuple[dict[str,np.ndarray],dict]:
    """Generate independently rendered frames and separate analytic evaluation truth.

    The correlated and depth-dependent depth errors are deliberate non-iid stressors.
    The solver still uses one nominal sigma, and this mismatch is reported explicitly.
    """
    from scipy.ndimage import gaussian_filter
    if not 0<=missing_fraction<1 or sigma_depth<=0 or sigma_flow<0:
        raise ValueError("invalid noise or missing-depth settings")
    rng=np.random.default_rng(seed)
    flow_rng=np.random.default_rng(np.random.SeedSequence([seed,701]))
    mask_rng=np.random.default_rng(np.random.SeedSequence([seed,702]))
    objects=make_scene(layout)
    if scene_seed is not None:
        objects=perturb_scene(objects,scene_seed)
    frames=[render_frame(objects,height,width,t,contrast) for t in (0.,1.)]
    intr,pixels=camera(height,width)
    xyz1=np.zeros_like(frames[0]["xyz"])
    moving=np.zeros((height,width),bool)
    for i,obj in enumerate(objects):
        mask=frames[0]["labels"]==i
        c,R=obj.pose(1.)
        xyz1[mask]=frames[0]["local"][mask]@R.T+c
        moving[mask]=obj.moving
    scene=xyz1-frames[0]["xyz"]
    q=np.stack((intr[0]*xyz1[...,0]/xyz1[...,2]+intr[2],intr[1]*xyz1[...,1]/xyz1[...,2]+intr[3]),axis=-1)
    target=render_queries(q,intr,objects,1.,shading=False)
    # Reprojection of an exact border pixel can be -1e-14 from roundoff.
    # This sub-nanopixel tolerance does not replace the physical visibility test.
    visibility_tolerance = 1e-9
    inside=(q[...,0]>=-visibility_tolerance)&(q[...,0]<=width-1+visibility_tolerance)&(q[...,1]>=-visibility_tolerance)&(q[...,1]<=height-1+visibility_tolerance)
    visible=inside&(target[...,2].astype(np.int32)==frames[0]["labels"])&(np.abs(target[...,0]-xyz1[...,2])<1e-8)
    gray0,gray1=[np.clip(f["gray"]+rng.normal(0,.003,(height,width)),0,1) for f in frames]
    if illumination:
        gray1=np.clip(1.14*gray1+.035,0,1)
    flow_gt=q-pixels
    dis_forward,dis_backward=image_flow(gray0,gray1)
    yy,xx=np.mgrid[:height,:width]
    outliers=np.zeros((height,width),bool);block=np.zeros_like(outliers)
    if flow_source=="corrupted_gt":
        flow=flow_gt+flow_rng.normal(0,sigma_flow,flow_gt.shape)
        outliers=flow_rng.random((height,width))<.15
        block=(xx>.24*width)&(xx<.38*width)&(yy>.42*height)&(yy<.60*height)
        outliers|=block
        flow[outliers]+=flow_rng.uniform(-12,12,(int(outliers.sum()),2))
        flow[block]=flow_gt[block]+np.array([8.,-6.])
    elif flow_source=="dis":
        flow=dis_forward
    else:
        raise ValueError("flow_source must be corrupted_gt or dis")
    depths=[]
    for f in frames:
        z=f["depth"]
        sigma=sigma_depth*(.65+.35*(z/1.2)**2)
        correlated=gaussian_filter(rng.normal(size=z.shape),1.5)
        correlated*=.0007/max(1e-12,np.std(correlated))
        obs=z+rng.normal(size=z.shape)*sigma+correlated
        if missing_fraction:
            obs[mask_rng.random(z.shape)<missing_fraction]=np.nan
            obs[(xx>.63*width)&(xx<.68*width)&(yy>.37*height)&(yy<.43*height)]=np.nan
        depths.append(obs)
    data={"gray0":gray0,"gray1":gray1,"rgb0":np.repeat(gray0[...,None],3,axis=-1),
          "rgb1":np.repeat(gray1[...,None],3,axis=-1),"depth0":depths[0],"depth1":depths[1],
          "flow_observed":flow,"flow_dis":dis_forward,"flow_dis_backward":dis_backward,"intrinsics":intr,
          "gt_depth0":frames[0]["depth"],"gt_depth1":frames[1]["depth"],"gt_flow":flow_gt,
          "gt_scene_flow":scene,"gt_visible":visible,"gt_inside":inside,"gt_occluded":inside&~visible,
          "gt_outside":~inside,"gt_moving":moving,"gt_labels0":frames[0]["labels"],"gt_labels1":frames[1]["labels"],
          "gt_normal0":frames[0]["normal"],"gt_normal1":frames[1]["normal"],
          "injected_outlier":outliers,"coherent_outlier":block}
    meta={"layout":layout,"seed":seed,"scene_seed":scene_seed,"height":height,"width":width,"flow_source":flow_source,
          "contrast":contrast,"illumination":illumination,"missing_fraction":missing_fraction,
          "sigma_depth_nominal_m":sigma_depth,"sigma_flow_px":sigma_flow,
          "depth_noise_model":f"sigma={sigma_depth}*(.65+.35*(z/1.2)^2), plus .7mm spatially correlated bias",
          "gray_sigma":.003,"random_streams":"base=seed, flow=[seed,701], missing=[seed,702]","gray_supersample":2,"depth_sampling":"centre ray optical-axis z",
          "shading":"ambient + three point-light samples + shadows, Lambertian",
          "visibility_roundoff_tolerance_px":visibility_tolerance,
          "visible_fraction":float(visible.mean()),"occluded_fraction":float(data["gt_occluded"].mean()),
          "max_true_motion_m":float(np.linalg.norm(scene,axis=-1).max()),
          "objects":[{"name":o.name,"kind":o.kind,"moving":o.moving,"bound_m":o.displacement_upper_bound,
                      "record0":o.record(0.).tolist(),"record1":o.record(1.).tolist()} for o in objects]}
    return data,meta
