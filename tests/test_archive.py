"""配布アーカイブの索引と整合性。"""

from __future__ import annotations

import numpy as np
import pytest

from rgbd_sceneflow.archive import (
    DATA_ROOT,
    SPLITS,
    case_count,
    dependency_versions,
    iter_cases,
    result_count,
)

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason="data/v03 が無い（`just extract` で復元してください）"
)

EXPECTED_INPUT_KEYS = {
    "gray0",
    "gray1",
    "depth0",
    "depth1",
    "flow_observed",
    "intrinsics",
    "gt_depth0",
    "gt_depth1",
    "gt_flow",
    "gt_scene_flow",
    "gt_visible",
    "gt_occluded",
    "gt_outside",
    "gt_moving",
    "gt_labels0",
    "gt_labels1",
}


def test_splits_are_present():
    counts = case_count()
    assert set(counts) == set(SPLITS), counts
    assert all(value > 0 for value in counts.values())


def test_result_json_count_is_70():
    assert result_count() == 70


def test_input_arrays_have_expected_keys_and_shapes():
    for case in iter_cases():
        inputs = case.inputs()
        assert EXPECTED_INPUT_KEYS <= set(inputs), case.key
        height, width = case.shape()
        assert inputs["gray0"].shape == (height, width)
        assert inputs["flow_observed"].shape == (height, width, 2)
        assert inputs["gt_scene_flow"].shape == (height, width, 3)


def test_result_arrays_have_expected_keys():
    expected = {
        "depth0",
        "depth1",
        "scene_flow",
        "flow",
        "match_flow",
        "initial_scene_flow",
        "support",
        "geometry_residual",
    }
    for case in iter_cases():
        for method in case.methods():
            assert expected <= set(case.result(method)), f"{case.key}/{method}"


def test_input_hash_records_are_present():
    for case in iter_cases():
        hashes = case.metadata()["input_sha256"]
        assert set(hashes) == {"gray0", "gray1", "depth0", "depth1", "flow_observed", "intrinsics"}


def test_recorded_dependency_versions_match_archive():
    versions = dependency_versions()["packages"]
    assert versions["numpy"] == "2.3.5"
    assert versions["scipy"] == "1.17.0"


def test_max_motion_matches_the_documented_value():
    """移動上限は標準版で課されるが、アブレーションの「上限なし」では課されない。"""
    excess = []
    for case in iter_cases():
        for method in case.methods():
            result = case.result(method)
            motion = np.linalg.norm(result["scene_flow"], axis=2).max() * 1000.0
            documented = case.documented(method)["metrics"]["max_estimated_motion_mm"]
            if abs(motion - documented) > 1e-9 * max(abs(documented), 1.0):
                excess.append((case.key, method, motion, documented))
    assert not excess, excess[:5]
