"""図の再構成が保存配列だけで動くことの検査。"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from rgbd_sceneflow import figures
from rgbd_sceneflow.archive import DATA_ROOT, iter_cases

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason="data/v03 が無い（`just extract` で復元してください）"
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
