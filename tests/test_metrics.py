"""メトリクス再計算の単体検査（合成配列）。"""

from __future__ import annotations

import numpy as np
import pytest

from rgbd_sceneflow.metrics import (
    bilinear_sample,
    boundary_mask,
    common_visible_set,
    compare,
    epe_mm,
    evaluate,
    lift_observed,
)


def test_bilinear_sample_interpolates_and_propagates_missing():
    image = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=float)
    value = bilinear_sample(image, np.array([[0.5]]), np.array([[0.5]]))
    assert value[0, 0] == pytest.approx(3.0)
    broken = image.copy()
    broken[0, 1] = np.nan
    assert np.isnan(bilinear_sample(broken, np.array([[0.5]]), np.array([[0.5]]))[0, 0])
    assert np.isnan(bilinear_sample(image, np.array([[-1.0]]), np.array([[0.0]]))[0, 0])


def test_boundary_mask_marks_both_sides_then_dilates():
    from scipy import ndimage

    labels = np.zeros((9, 9), dtype=np.int32)
    labels[:, 4:] = 1  # 列4で物体番号が変わる
    both_sides = np.zeros_like(labels, dtype=bool)
    both_sides[:, 3:5] = True  # 変わり目の左右を両方立てる
    one = boundary_mask(labels, iterations=1)
    two = boundary_mask(labels, iterations=2)
    assert np.array_equal(one, ndimage.binary_dilation(both_sides, iterations=1))
    assert two.sum() > one.sum()
    assert two[:, 6].any() and not two[:, 0].any()  # 半径2の広がり
    assert not boundary_mask(np.zeros_like(labels), iterations=2).any()


def test_lift_observed_is_finite_only_for_valid_correspondence():
    height, width = 4, 5
    shape = (height, width)
    inputs = {
        "depth0": np.ones(shape),
        "depth1": np.ones(shape),
        "flow_observed": np.zeros((height, width, 2)),
        "intrinsics": np.array([100.0, 100.0, 2.0, 2.0]),
    }
    raw = lift_observed(inputs)
    # 最右列・最下行は x0+1 が画像外になるため有限にならない。
    assert np.isfinite(raw[:-1, :-1]).all()
    assert not np.isfinite(raw[:, -1]).any()
    assert common_visible_set({**inputs, "gt_visible": np.ones(shape, dtype=bool)}).sum() == (height - 1) * (
        width - 1
    )


def test_epe_is_zero_for_perfect_estimate_and_uses_millimetres():
    gt = np.zeros((2, 3, 3), dtype=float)
    assert epe_mm(gt.copy(), gt).max() == 0.0
    estimate = gt.copy()
    estimate[0, 0, 0] = 0.002
    assert epe_mm(estimate, gt)[0, 0] == pytest.approx(2.0)


def _synthetic_inputs(height: int = 6, width: int = 8) -> dict[str, np.ndarray]:
    shape = (height, width)
    visible = np.ones(shape, dtype=bool)
    visible[0, 0] = False
    return {
        "gray0": np.zeros(shape),
        "gray1": np.zeros(shape),
        "depth0": np.ones(shape),
        "depth1": np.ones(shape),
        "flow_observed": np.zeros((height, width, 2)),
        "intrinsics": np.array([100.0, 100.0, width / 2, height / 2]),
        "gt_depth0": np.ones(shape),
        "gt_depth1": np.ones(shape),
        "gt_flow": np.zeros((height, width, 2)),
        "gt_scene_flow": np.zeros((height, width, 3)),
        "gt_visible": visible,
        "gt_moving": np.zeros(shape, dtype=bool),
        "gt_occluded": np.zeros(shape, dtype=bool),
        "gt_outside": np.zeros(shape, dtype=bool),
        "gt_labels0": np.zeros(shape, dtype=np.int32),
        "gt_labels1": np.zeros(shape, dtype=np.int32),
        "injected_outlier": np.zeros(shape, dtype=bool),
        "coherent_outlier": np.zeros(shape, dtype=bool),
    }


def _synthetic_result(height: int = 6, width: int = 8) -> dict[str, np.ndarray]:
    shape = (height, width)
    return {
        "scene_flow": np.zeros((height, width, 3)),
        "flow": np.zeros((height, width, 2)),
        "match_flow": np.zeros((height, width, 2)),
        "depth0": np.ones(shape),
        "depth1": np.ones(shape),
        "support": np.ones(shape),
        "geometry_residual": np.zeros(shape),
    }


def test_evaluate_returns_documented_metric_names():
    inputs = _synthetic_inputs()
    metadata = {"objects": [{"name": "wall", "moving": False}, {"name": "ball", "moving": True}]}
    out = evaluate(inputs, _synthetic_result(), metadata)["metrics"]
    for key in ("common_visible_n", "moving_epe_mm", "depth0_rmse_mm", "support_percent", "p95_mm"):
        assert key in out
    assert out["support_percent"] == pytest.approx(100.0)
    assert out["depth0_rmse_mm"] == pytest.approx(0.0)
    assert out["object_balanced_epe_mm"] is None  # 動体は物体番号 1 だが画素が無い


def test_compare_classifies_exact_approx_and_internal():
    recomputed = {"a": 1.0, "b": 2.0}
    documented = {"a": 1.0, "b": 2.5, "elapsed_seconds": 9.0}
    statuses = {s.name: s.status for s in compare(recomputed, documented, approx_keys=frozenset({"b"}))}
    assert statuses["a"] == "exact"
    assert statuses["b"] == "approx"
    assert statuses["elapsed_seconds"] == "not_recomputable"
