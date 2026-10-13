"""70 組の凍結メトリクスを再計算し、突合する統合テスト。

配布実装のソースが届いた後も、ラボ側の再実装が**独立に**同じ値を出すことを確認する。
推定器の再実行そのものは `just build` と `just verify-upstream` が担う。
"""

from __future__ import annotations

from collections import Counter

import pytest

from rgbd_sceneflow.archive import DATA_ROOT, iter_cases
from rgbd_sceneflow.metrics import APPROX_KEYS, compare, evaluate

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason="data が無い（`just extract` で復元してください）"
)

CORE_METRICS = {
    "common_visible_n",
    "common_visible_epe_mm",
    "all_visible_epe_mm",
    "moving_n",
    "moving_epe_mm",
    "static_n",
    "static_epe_mm",
    "boundary_n",
    "boundary_epe_mm",
    "interior_n",
    "interior_epe_mm",
    "static_gray_change_n",
    "static_gray_change_epe_mm",
    "occluded_n",
    "occluded_epe_mm",
    "p50_mm",
    "p95_mm",
    "p99_mm",
    "bad5_percent",
    "bad10_percent",
    "moving_p95_mm",
    "optical_flow_epe_px",
    "moving_flow_epe_px",
    "depth0_rmse_mm",
    "depth1_rmse_mm",
    "depth0_hole_rmse_mm",
    "depth1_hole_rmse_mm",
    "support_percent",
    "object_balanced_epe_mm",
    "match_reprojection_gap_mean_px",
}


@pytest.fixture(scope="module")
def statuses_by_record():
    records = {}
    for case in iter_cases():
        for method in case.methods():
            recomputed = evaluate(case.inputs(), case.result(method), case.metadata())["metrics"]
            documented = case.documented(method)["metrics"]
            records[(case.key, method)] = compare(recomputed, documented, approx_keys=APPROX_KEYS)
    return records


def test_all_70_records_are_compared(statuses_by_record):
    assert len(statuses_by_record) == 70


def test_no_mismatch_and_no_approximation(statuses_by_record):
    counts = Counter(status.status for statuses in statuses_by_record.values() for status in statuses)
    assert counts["mismatch"] == 0
    assert counts["approx"] == 0
    assert counts["exact"] > counts["not_recomputable"] > 0


def test_every_core_metric_is_exact_in_every_record(statuses_by_record):
    missing = [
        (key, status.name, status.recomputed, status.documented)
        for key, statuses in statuses_by_record.items()
        for status in statuses
        if status.name in CORE_METRICS and status.status != "exact"
    ]
    assert not missing, f"厳密一致しなかった指標 {len(missing)} 件: {missing[:5]}"


def test_solver_internal_metrics_are_marked_not_recomputable(statuses_by_record):
    names = {
        status.name
        for statuses in statuses_by_record.values()
        for status in statuses
        if status.status == "not_recomputable"
    }
    assert {"elapsed_seconds", "adjoint_error", "candidate_total"} <= names
