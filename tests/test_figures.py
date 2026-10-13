"""図の再構成が保存配列だけで動くことの検査。"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rgbd_sceneflow import figures
from rgbd_sceneflow.archive import DATA_ROOT, iter_cases

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason="data が無い（`just extract` で復元してください）"
)


@pytest.fixture(scope="module")
def heldout_cases():
    return [case for case in iter_cases() if case.split == "heldout"][:2]


def test_input_and_error_figures_are_written(tmp_path, heldout_cases):
    inputs_path = tmp_path / "inputs.png"
    errors_path = tmp_path / "errors.png"
    plt.close(figures.figure_inputs(heldout_cases, inputs_path))
    plt.close(figures.figure_error_maps(heldout_cases, errors_path))
    assert inputs_path.stat().st_size > 0
    assert errors_path.stat().st_size > 0


def test_speed_accuracy_figure_uses_stored_timings(tmp_path):
    case = figures.ablation_case()
    assert case is not None
    path = tmp_path / "speed.png"
    plt.close(figures.figure_speed_accuracy(case, path))
    assert path.stat().st_size > 0


def test_scene_flow_3d_figure_is_written(tmp_path):
    case = next(c for c in iter_cases() if c.key == "heldout/gallery_scene101")
    path = tmp_path / "scene_flow_3d.png"
    figure = figures.figure_scene_flow_3d(case, path=path)
    assert len(figure.axes) == len(figures.DEFAULT_3D_VARIANTS) + 1  # パネル + 色バー
    plt.close(figure)
    assert path.stat().st_size > 0


def _case(key: str = "heldout/gallery_scene101"):
    return next(c for c in iter_cases() if c.key == key)


def test_ground_truth_figure_is_written(tmp_path):
    path = tmp_path / "ground_truth.png"
    plt.close(figures.figure_ground_truth(_case(), path))
    assert path.stat().st_size > 0


def test_style_figures_are_written(tmp_path):
    case = _case()
    for name, function in (
        ("optical_flow_styles.png", figures.figure_optical_flow_styles),
        ("scene_flow_styles.png", figures.figure_scene_flow_styles),
        ("scene_flow_3d_views.png", figures.figure_scene_flow_3d_views),
    ):
        path = tmp_path / name
        plt.close(function(case, path=path))
        assert path.stat().st_size > 0, name


def test_flow_color_wheel_matches_the_reference_layout():
    wheel = figures.flow_color_wheel()
    assert wheel.shape == (sum(figures.FLOW_WHEEL_SEGMENTS), 3)
    assert wheel.min() >= 0.0 and wheel.max() <= 1.0
    assert np.allclose(wheel[0], [1.0, 0.0, 0.0])  # 赤から始まる


def test_optical_flow_coding_maps_direction_and_magnitude():
    flow = np.zeros((1, 3, 2))
    flow[0, 1] = [1.0, 0.0]  # 右向き・最大値
    flow[0, 2] = [0.0, 1.0]  # 下向き・最大値
    rgb = figures.optical_flow_to_rgb(flow, max_magnitude=1.0)
    assert np.allclose(rgb[0, 0], 1.0, atol=0.2)  # ゼロは白
    assert not np.allclose(rgb[0, 1], rgb[0, 2], atol=0.05)  # 向きで色が変わる
    assert rgb[0, 1].max() > 0.9  # 最大値は彩度が高い

    weaker = np.zeros((1, 1, 2))
    weaker[0, 0] = [0.5, 0.0]  # 同じ向きで半分の大きさ
    assert figures.optical_flow_to_rgb(weaker, max_magnitude=1.0).mean() > rgb[0, 1].mean()


def test_scene_flow_component_coding_is_grey_at_zero():
    scene_flow = np.zeros((1, 3, 3))
    scene_flow[0, 1, 0] = 0.01  # x 正
    scene_flow[0, 2, 0] = -0.01  # x 負
    rgb = figures.scene_flow_to_rgb(scene_flow, scales=(0.01, 0.01, 0.01))
    assert np.allclose(rgb[0, 0], 0.5)
    assert rgb[0, 1, 0] > 0.5 > rgb[0, 2, 0]
    assert np.allclose(rgb[0, 1, 1:], 0.5)
