"""集計表の再計算が凍結集計（summary.json）と一致することの検査。"""

from __future__ import annotations

import pytest

from rgbd_sceneflow import tables
from rgbd_sceneflow.archive import DATA_ROOT, read_json

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason="data/v03 が無い（`just extract` で復元してください）"
)


def test_main_table_recomputes_from_per_case_metrics():
    stats = tables.aggregate(tables.collect())
    retest = stats[("retest", "standard")]
    heldout = stats[("heldout", "standard")]
    assert retest["n"] == 6 and heldout["n"] == 6
    # 理論書 table_main.tex の値との照合（標準版・共通可視）。
    assert heldout["common_visible_epe_mm"]["mean"] == pytest.approx(0.991, abs=5e-4)
    assert retest["common_visible_epe_mm"]["mean"] == pytest.approx(0.852, abs=5e-4)


def test_aggregate_matches_frozen_summary_exactly():
    stats = tables.aggregate(tables.collect())
    summary = read_json(DATA_ROOT / "summary.json")
    rows = tables.compare_with_summary(stats, summary)
    assert rows, "突合項目が空"
    worst = max(row["difference"] for row in rows)
    assert worst <= 1e-9, f"凍結集計との最大差 {worst:.3e}"
