"""Validated Python boundary; the numerical algorithm lives entirely in C++."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from . import _ffi

@dataclass(frozen=True)
class Config:
    """Distances are in metres, flow in pixels; see ../docs/theory_ja.tex."""
    scale: float = 0.01
    sigma_depth: float = 0.003
    sigma_flow: float = 0.7
    sigma_image: float = 0.025
    sigma_geometry: float = 0.0008
    max_motion: float = 0.05
    trust_sigma: float = 4.0
    min_depth: float = 0.10
    max_depth: float = 5.0
    edge_threshold: float = 0.06
    depth_alpha1: float = 1.2
    depth_alpha0: float = 4.0
    flow_alpha1: float = 2.0
    flow_alpha0: float = 6.0
    image_weight: float = 1.0
    flow_weight: float = 0.025
    flow_truncation: float = 9.0
    photo_reject: float = 3.0
    huber_k: float = 3.0
    ridge: float = 1e-6
    iterations: int = 400
    outer_iterations: int = 3
    candidate_count: int = 24
    max_search: int = 18
    patch_radius: int = 2
    edge_aware: bool = True
    use_tgv: bool = True
    enforce_bound: bool = True

    centered_photo: bool = True
    reciprocal_check: bool = True
    diverse_candidates: bool = True

    fast_match: bool = True
    affine_photo: bool = False
    rigid_prior: bool = True
    fused_solver: bool = True
    warm_start: bool = True
    keep_identity: bool = True
    later_iterations: int = 150
    coarse_stride: int = 2
    seed_count: int = 3
    rigid_min_points: int = 24
    rigid_sigma: float = 0.0015
    rigid_huber: float = 0.006
    step_balance: float = 1.0
    geometry_huber: bool = True
    geometry_k: float = 3.0

    def native(self) -> np.ndarray:
        values = [float(v) for v in asdict(self).values()]
        return np.asarray(values, dtype=np.float64)


def refine(rgb0: ArrayLike, rgb1: ArrayLike, depth0: ArrayLike, depth1: ArrayLike,
           flow: ArrayLike, intrinsics: ArrayLike, config: Config | None = None) -> dict[str, Any]:
    """Refine a registered, fixed-camera grayscale/RGB and depth pair.

    Gray: HxW or RGB: HxWx3, finite [0,1]; depths: HxW optical-axis depth in metres;
    flow: HxWx2 displacement (dx,dy) from frame 0 to frame 1;
    intrinsics: (fx,fy,cx,cy), integer coordinates at pixel centres.
    Invalid depth may be NaN or non-positive. There is deliberately no GT,
    segmentation label, normal, or visibility-mask parameter.
    """
    rgb0 = np.asarray(rgb0)
    rgb1 = np.asarray(rgb1)
    if rgb0.ndim == 2:
        rgb0 = np.repeat(rgb0[..., None], 3, axis=-1)
    if rgb1.ndim == 2:
        rgb1 = np.repeat(rgb1[..., None], 3, axis=-1)
    arrays = [np.ascontiguousarray(a, dtype=np.float64)
              for a in (rgb0, rgb1, depth0, depth1, flow, intrinsics)]
    result = _ffi.solve(arrays, (config or Config()).native())
    # Missing-depth links are deliberately cut by this model. Values at those
    # sites may retain the numerical initializer and are NOT supported completion.
    result["depth0_supported"] = np.isfinite(arrays[2]) & (arrays[2] > 0)
    result["depth1_supported"] = np.isfinite(arrays[3]) & (arrays[3] > 0)
    return result
