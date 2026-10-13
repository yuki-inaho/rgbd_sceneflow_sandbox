"""凍結メトリクスの再計算（配布実装と同一の定義）。

推定器の出力（``*.npz``）と入力（``data.npz``）だけから、``*.json`` に記録された
評価量を再計算する。

定義は 2 段階で確定した。

1. アーカイブの数値だけを手がかりに探索し、評価集合 ``V`` などを復元した。
2. 後から届いた配布実装 ``scene_flow_primitives/src/scene_flow_refinement/primitive_metrics.py``
   と突き合わせ、残っていた差（境界の作り方、深度 RMSE の有効域、物体別マスクなど）を解消した。

したがって本モジュールは配布実装の**独立的な再実装**であり、一致すれば
「評価量が保存配列から一意に再現できる」ことの証拠になる。
推定器の内部状態（処理時間・候補数・固定点残差など）は保存配列からは決まらないため対象外。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

#: 移動上限 [m]（凍結プロトコルの標準値）。超過判定には浮動小数の丸め分の許容を置く。
MAX_MOTION_M = 0.05
BOUND_TOLERANCE_M = 1e-9
#: 静止灰度変化域のしきい値（理論書 §6.3）。
GRAY_CHANGE_THRESHOLD = 0.04
#: 支持の判定しきい値（配布実装に合わせる）。
SUPPORT_THRESHOLD = 0.5
#: 物体別 RMSE で「物体」とみなす物体番号の下限（配布実装に合わせる）。
OBJECT_LABEL_MIN = 2

#: 推定器の内部状態に依存し、保存配列からは再計算できないメトリクス。
SOLVER_INTERNAL_KEYS = (
    "elapsed_seconds",
    "matching_seconds",
    "prior_seconds",
    "optimization_seconds",
    "python_wall_seconds",
    "adjoint_error",
    "fixed_point_primal",
    "fixed_point_dual",
    "operator_norm_bound",
    "candidate_changes",
    "candidate_total",
    "proposed_candidates",
    "geometry_rejected_candidates",
    "rigid_components",
    "rigid_accepted",
    "inner_iterations",
)


def rays(shape: tuple[int, int], intrinsics: np.ndarray) -> np.ndarray:
    """画素ごとの正規化レイ（z 成分は 1）。"""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    fx, fy, cx, cy = intrinsics
    return np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1)


def bilinear_sample(image: np.ndarray, qy: np.ndarray, qx: np.ndarray) -> np.ndarray:
    """双線形補間。範囲外や欠測（NaN・非有限）が 1 つでも絡めば NaN を返す。"""
    height, width = image.shape
    inside = np.isfinite(qx) & np.isfinite(qy)
    safe_x = np.where(inside, qx, 0.0)
    safe_y = np.where(inside, qy, 0.0)
    x0 = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    in_bounds = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    wx, wy = safe_x - x0, safe_y - y0
    v00, v01 = image[y0c, x0c], image[y0c, x1c]
    v10, v11 = image[y1c, x0c], image[y1c, x1c]
    top = v00 * (1.0 - wx) + v01 * wx
    bottom = v10 * (1.0 - wx) + v11 * wx
    value = top * (1.0 - wy) + bottom * wy
    # 重みが 0 でも欠測は伝播させる（配布実装の map_coordinates と同じ挙動）。
    bad = ~inside | ~in_bounds | ~np.isfinite(v00) | ~np.isfinite(v01) | ~np.isfinite(v10) | ~np.isfinite(v11)
    return np.where(bad, np.nan, value)


def lift_observed(inputs: dict[str, np.ndarray]) -> np.ndarray:
    """観測フローと観測深度の素朴な持ち上げ（3 次元変位）。

    配布実装 ``metrics.lift_observed`` と同じ条件で、クリップも境界除去も隠さない。
    """
    depth0 = inputs["depth0"]
    depth1 = inputs["depth1"]
    flow = inputs["flow_observed"]
    intrinsics = inputs["intrinsics"]
    height, width = depth0.shape
    yy, xx = np.mgrid[0:height, 0:width]
    qx = xx + flow[..., 0]
    qy = yy + flow[..., 1]
    inside = (qx >= 0) & (qy >= 0) & (qx < width - 1) & (qy < height - 1) & np.isfinite(qx) & np.isfinite(qy)
    z = bilinear_sample(depth1, qy, qx)
    fx, fy, cx, cy = intrinsics
    direction = np.stack(((qx - cx) / fx, (qy - cy) / fy, np.ones_like(qx)), axis=-1)
    raw = z[..., None] * direction - depth0[..., None] * rays((height, width), intrinsics)
    raw[~inside] = np.nan
    return raw


def boundary_mask(labels: np.ndarray, iterations: int = 2) -> np.ndarray:
    """物体境界の帯。隣接画素の**両側**を立ててから膨張する（配布実装と同じ）。"""
    edge = np.zeros(labels.shape, dtype=bool)
    for axis in (0, 1):
        left = [slice(None)] * 2
        right = [slice(None)] * 2
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        jump = labels[tuple(left)] != labels[tuple(right)]
        edge[tuple(left)] |= jump
        edge[tuple(right)] |= jump
    return ndimage.binary_dilation(edge, iterations=iterations)


def common_visible_set(inputs: dict[str, np.ndarray]) -> np.ndarray:
    """評価集合 ``V``（主指標を平均する集合）。"""
    raw = lift_observed(inputs)
    return inputs["gt_visible"] & np.isfinite(raw).all(axis=-1)


def region_masks(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """評価集合の一覧（配布実装 ``primitive_metrics.region_masks`` と同じ）。"""
    labels = inputs["gt_labels0"]
    common = common_visible_set(inputs)
    edge = boundary_mask(labels)
    return {
        "common_visible": common,
        "all_visible": inputs["gt_visible"],
        "moving": common & inputs["gt_moving"],
        "static": common & ~inputs["gt_moving"],
        "boundary": common & edge,
        "interior": common & ~edge,
        "static_gray_change": common
        & ~inputs["gt_moving"]
        & ~edge
        & (np.abs(inputs["gray1"] - inputs["gray0"]) > GRAY_CHANGE_THRESHOLD),
        "occluded": inputs["gt_occluded"],
        "outside": inputs["gt_outside"],
        "outlier": common & inputs["injected_outlier"],
        "coherent_outlier": common & inputs["coherent_outlier"],
    }


def epe_mm(scene_flow: np.ndarray, gt_scene_flow: np.ndarray) -> np.ndarray:
    """画素ごとの 3 次元端点誤差 [mm]。"""
    return np.linalg.norm(scene_flow - gt_scene_flow, axis=2) * 1000.0


def evaluate(
    inputs: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
    metadata: dict | None = None,
) -> dict[str, Any]:
    """保存配列から凍結メトリクスを再計算する。"""
    scene = result["scene_flow"]
    finite = np.isfinite(scene).all(axis=-1)
    error = epe_mm(scene, inputs["gt_scene_flow"])
    regions = region_masks(inputs)

    metrics: dict[str, Any] = {}
    for name, mask in regions.items():
        valid = mask & finite
        metrics[f"{name}_n"] = int(valid.sum())
        metrics[f"{name}_coverage"] = float(valid.sum() / max(1, int(mask.sum())))
        metrics[f"{name}_epe_mm"] = float(error[valid].mean()) if valid.any() else None

    common = regions["common_visible"] & finite
    for percentile in (50, 95, 99):
        metrics[f"p{percentile}_mm"] = (
            float(np.percentile(error[common], percentile)) if common.any() else None
        )
    for threshold in (5, 10):
        metrics[f"bad{threshold}_percent"] = (
            float(np.mean(error[common] > threshold) * 100) if common.any() else None
        )
    moving = regions["moving"] & finite
    metrics["moving_p95_mm"] = float(np.percentile(error[moving], 95)) if moving.any() else None

    magnitude = np.linalg.norm(scene, axis=-1)
    metrics["max_estimated_motion_mm"] = float(np.nanmax(magnitude) * 1000.0)
    metrics["bound_violation_percent"] = float(
        np.mean(magnitude[finite] > MAX_MOTION_M + BOUND_TOLERANCE_M) * 100.0
    )

    for frame in (0, 1):
        estimate = result[f"depth{frame}"]
        observed = inputs[f"depth{frame}"]
        valid = np.isfinite(observed) & (observed > 0)
        delta = estimate - inputs[f"gt_depth{frame}"]
        available = np.isfinite(estimate)
        metrics[f"depth{frame}_rmse_mm"] = float(np.sqrt(np.mean(delta[available] ** 2)) * 1000.0)
        metrics[f"depth{frame}_observed_rmse_mm"] = float(
            np.sqrt(np.mean(delta[valid & available] ** 2)) * 1000.0
        )
        objects_mask = inputs[f"gt_labels{frame}"] >= OBJECT_LABEL_MIN
        object_pixels = objects_mask & available
        metrics[f"depth{frame}_object_rmse_mm"] = (
            float(np.sqrt(np.mean(delta[object_pixels] ** 2)) * 1000.0)
            if object_pixels.any()
            else float("nan")
        )
        hole = ~valid & available
        metrics[f"depth{frame}_hole_rmse_mm"] = (
            float(np.sqrt(np.mean(delta[hole] ** 2)) * 1000.0) if hole.any() else None
        )

    flow = result.get("flow")
    if flow is not None:
        flow_error = np.linalg.norm(flow - inputs["gt_flow"], axis=-1)
        metrics["optical_flow_epe_px"] = float(np.nanmean(flow_error[common]))
        metrics["moving_flow_epe_px"] = float(np.nanmean(flow_error[moving])) if moving.any() else None

    support = result["support"] > SUPPORT_THRESHOLD
    metrics["support_percent"] = float(support.mean() * 100.0)
    for label, mask in (("visible", inputs["gt_visible"]), ("occluded", inputs["gt_occluded"])):
        metrics[f"{label}_support_percent"] = float(support[mask].mean() * 100.0) if mask.any() else None

    residual = result["geometry_residual"]
    covered = support & np.isfinite(residual)
    metrics["geometry_mean_mm"] = float(np.mean(residual[covered]) * 1000.0) if covered.any() else None
    metrics["geometry_p95_mm"] = (
        float(np.percentile(residual[covered], 95) * 1000.0) if covered.any() else None
    )
    if flow is not None:
        gap = np.linalg.norm(result["match_flow"] - result["flow"], axis=-1)
        finite_gap = support & np.isfinite(gap)
        metrics["match_reprojection_gap_mean_px"] = (
            float(gap[finite_gap].mean()) if finite_gap.any() else None
        )
        metrics["match_reprojection_gap_p95_px"] = (
            float(np.percentile(gap[finite_gap], 95)) if finite_gap.any() else None
        )
    metrics["supported_epe_mm"] = float(error[common & support].mean()) if (common & support).any() else None
    metrics["unsupported_epe_mm"] = (
        float(error[common & ~support].mean()) if (common & ~support).any() else None
    )

    objects = _object_metrics(inputs, error, common, metadata)
    moving_epe = [obj["epe_mm"] for obj in objects if obj["moving"]]
    metrics["object_balanced_epe_mm"] = float(np.mean(moving_epe)) if moving_epe else None
    return {"metrics": metrics, "objects": objects}


def _object_metrics(
    inputs: dict[str, np.ndarray],
    error: np.ndarray,
    common: np.ndarray,
    metadata: dict | None,
) -> list[dict[str, Any]]:
    """物体番号ごとの件数と端点誤差（動体かどうかは metadata の記載に従う）。"""
    specification = (metadata or {}).get("objects", [])
    labels = inputs["gt_labels0"]
    objects = []
    for index, obj in enumerate(specification):
        mask = common & (labels == index)
        if not mask.any():
            continue
        objects.append(
            {
                "object": obj.get("name"),
                "object_id": index,
                "moving": bool(obj.get("moving")),
                "count": int(mask.sum()),
                "epe_mm": float(error[mask].mean()),
                "p95_mm": float(np.percentile(error[mask], 95)),
            }
        )
    return objects


@dataclass(frozen=True)
class MetricStatus:
    """1 メトリクスの突合結果。"""

    name: str
    recomputed: float | None
    documented: float | None
    difference: float | None
    status: str  # exact / approx / mismatch / not_recomputable / undefined


def compare(
    recomputed: dict[str, Any],
    documented: dict[str, Any],
    *,
    approx_keys: frozenset[str] = frozenset(),
) -> list[MetricStatus]:
    """再計算値と凍結値を突合する。

    ``exact``    : 浮動小数の丸め内で一致（相対 1e-12）
    ``approx``   : 定義の細部が未特定で、近いが一致しない
    ``mismatch`` : 一致せず、近似扱いにもしていない
    """
    statuses: list[MetricStatus] = []
    for name in sorted(documented):
        if name in SOLVER_INTERNAL_KEYS:
            statuses.append(MetricStatus(name, None, documented[name], None, "not_recomputable"))
            continue
        if name not in recomputed:
            statuses.append(MetricStatus(name, None, documented[name], None, "undefined"))
            continue
        new = recomputed[name]
        old = documented[name]
        if new is None and old is None:
            statuses.append(MetricStatus(name, None, None, None, "exact"))
            continue
        if new is None or old is None:
            statuses.append(MetricStatus(name, new, old, None, "mismatch"))
            continue
        difference = abs(float(new) - float(old))
        if isinstance(old, int) and isinstance(new, int):
            status = "exact" if difference == 0 else ("approx" if name in approx_keys else "mismatch")
        else:
            scale = max(abs(float(old)), 1e-12)
            status = (
                "exact" if difference <= 1e-12 * scale else ("approx" if name in approx_keys else "mismatch")
            )
        statuses.append(MetricStatus(name, float(new), float(old), difference, status))
    return statuses


#: 定義の細部を特定できず近似として扱うメトリクス。
#: 配布実装との突き合わせで境界の作り方を確定したため、現在は空。
APPROX_KEYS: frozenset[str] = frozenset()
