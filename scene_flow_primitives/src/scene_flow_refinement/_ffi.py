"""Small ownership-safe ctypes wrapper around the standalone C++ shared library."""
from __future__ import annotations
import ctypes as ct
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np

_DOUBLE = ct.POINTER(ct.c_double)
STAT_NAMES = ("elapsed_seconds", "matching_seconds", "adjoint_error", "fixed_point_primal",
              "fixed_point_dual", "operator_norm_bound", "candidate_changes", "candidate_total",
              "proposed_candidates", "geometry_rejected_candidates", "prior_seconds", "optimization_seconds",
              "rigid_components", "rigid_accepted", "inner_iterations")

@lru_cache(maxsize=1)
def library() -> ct.CDLL:
    directory = Path(__file__).resolve().parent
    choices = sorted(path for path in directory.glob("_native*") if path.suffix in {".so", ".pyd", ".dylib", ".dll"})
    if not choices:
        raise RuntimeError("C++ library not built. Run `uv sync`, or `python setup.py build_ext --inplace`.")
    lib = ct.CDLL(str(choices[0]))
    lib.sfr_solve.argtypes = [ct.c_int,ct.c_int]+[_DOUBLE]*7+[ct.c_int]
    lib.sfr_solve.restype = ct.c_void_p
    lib.sfr_error.argtypes = []
    lib.sfr_error.restype = ct.c_char_p
    lib.sfr_array.argtypes = [ct.c_void_p,ct.c_int,ct.POINTER(ct.c_int)]
    lib.sfr_array.restype = _DOUBLE
    lib.sfr_free.argtypes = [ct.c_void_p]
    lib.sfr_free.restype = None
    lib.sfr_huber_prox.argtypes = [ct.c_double]*5
    lib.sfr_huber_prox.restype = ct.c_double
    lib.sfr_ray_distance.argtypes = [_DOUBLE,_DOUBLE]+[ct.c_double]*4
    lib.sfr_ray_distance.restype = ct.c_double
    lib.sfr_regularizer_energy.argtypes = [ct.c_int]*3+[_DOUBLE]*3+[ct.c_double]*3+[ct.c_int]
    lib.sfr_regularizer_energy.restype = ct.c_double
    return lib


def solve(arrays: list[np.ndarray], parameters: np.ndarray) -> dict[str,Any]:
    rgb0,rgb1,d0,d1,flow,intr = arrays
    if d0.ndim!=2 or min(d0.shape)<2:
        raise ValueError("depth0 must have shape HxW, H,W>=2")
    h,w = d0.shape
    for name,a,shape in (("rgb0",rgb0,(h,w,3)),("rgb1",rgb1,(h,w,3)),
                         ("depth1",d1,(h,w)),("flow",flow,(h,w,2)),("intrinsics",intr,(4,))):
        if a.shape!=shape:
            raise ValueError(f"{name}: expected {shape}, got {a.shape}")
    lib = library()
    # All returned arrays are copied before the C++ owner is released.
    handle = lib.sfr_solve(h,w,*[a.ctypes.data_as(_DOUBLE) for a in arrays],
                           parameters.ctypes.data_as(_DOUBLE),parameters.size)
    if not handle:
        raise ValueError(lib.sfr_error().decode("utf-8",errors="replace"))
    result: dict[str,Any] = {}
    keys = ("depth0","depth1","scene_flow","flow","match_flow","initial_scene_flow",
            "support","geometry_residual","history","stats","rigid_scene","rigid_weight","components")
    shapes = ((h,w),(h,w),(h,w,3),(h,w,2),(h,w,2),(h,w,3),(h,w),(h,w),(-1,4),(-1,),(h,w,3),(h,w),(h,w))
    try:
        for index,(key,shape) in enumerate(zip(keys,shapes,strict=True)):
            size=ct.c_int()
            ptr=lib.sfr_array(handle,index,ct.byref(size))
            if not ptr and size.value:
                raise RuntimeError(f"native result field {key} is unavailable")
            value=np.ctypeslib.as_array(ptr,shape=(size.value,)).copy().reshape(shape)
            result[key]=dict(zip(STAT_NAMES,value.tolist(),strict=True)) if key=="stats" else value
    finally:
        lib.sfr_free(handle)
    return result
