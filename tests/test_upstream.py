"""配布実装（scene_flow_primitives）との突き合わせ。

ラボ側の再実装が、配布された評価コードと同じ値を出すことを全 70 組で確認する。
配布実装のネイティブライブラリは遅延ロードなので、評価関数の読み込みにビルドは要らない
（推定そのものを走らせるには `just build` が必要）。
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from rgbd_sceneflow import metrics
from rgbd_sceneflow.archive import PROJECT_ROOT, iter_cases

UPSTREAM = PROJECT_ROOT / "scene_flow_primitives"

pytestmark = pytest.mark.skipif(
    not (UPSTREAM / "src").is_dir(),
    reason="配布実装 scene_flow_primitives/ が無い（コードの zip を展開してください）",
)


@pytest.fixture(scope="module")
def upstream_modules():
    sys.path.insert(0, str(UPSTREAM / "src"))
    try:
        from scene_flow_refinement.metrics import lift_observed
        from scene_flow_refinement.primitive_metrics import region_masks, score_primitive
    except ImportError as exc:  # pragma: no cover - 環境依存
        pytest.skip(f"配布実装を読み込めない: {exc}")
    return lift_observed, region_masks, score_primitive


def test_common_visible_set_matches_upstream(upstream_modules):
    lift_observed, region_masks, _ = upstream_modules
    for case in iter_cases():
        data = case.inputs()
        raw = lift_observed(data["depth0"], data["depth1"], data["flow_observed"], data["intrinsics"])
        theirs = region_masks(data, raw)["common_visible"]
        assert np.array_equal(theirs, metrics.common_visible_set(data)), case.key


def test_all_documented_metrics_match_upstream(upstream_modules):
    lift_observed, region_masks, score_primitive = upstream_modules
    checked = 0
    for case in iter_cases():
        data = case.inputs()
        metadata = case.metadata()
        raw = lift_observed(data["depth0"], data["depth1"], data["flow_observed"], data["intrinsics"])
        regions = region_masks(data, raw)
        for method in case.methods():
            theirs, _ = score_primitive(data, case.result(method), regions, metadata["objects"])
            ours = metrics.evaluate(data, case.result(method), metadata)["metrics"]
            for key, value in theirs.items():
                if value is None or not isinstance(value, (int, float)):
                    continue
                assert key in ours, f"{case.key}/{method}: {key} がラボ側に無い"
                tolerance = 1e-12 * max(abs(float(value)), 1.0)
                assert abs(float(value) - float(ours[key])) <= tolerance, (
                    f"{case.key}/{method}: {key} 配布実装={value} ラボ={ours[key]}"
                )
            checked += 1
    assert checked == 70


def test_vendored_tree_is_complete():
    for relative in (
        "cpp/solver.cpp",
        "cpp/matcher.cpp",
        "cpp/rigid.cpp",
        "cpp/renderer.cpp",
        "cpp/bindings_c.cpp",
        "src/scene_flow_refinement/api.py",
        "src/scene_flow_refinement/primitive_metrics.py",
        "scripts/verify_v03.py",
        "tests/test_v03.py",
    ):
        assert (UPSTREAM / relative).is_file(), relative
