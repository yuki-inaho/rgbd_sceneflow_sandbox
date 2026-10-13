"""保存メトリクスからの集計表の再構成と、凍結集計（summary.json）との突合。

uv run python -m rgbd_sceneflow.tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .archive import DATA_ROOT, OUTPUTS_DIR, iter_cases, read_json

#: 集計表に出す指標（列名, json キー）。
COLUMNS = (
    ("共通可視 [mm]", "common_visible_epe_mm"),
    ("動体 [mm]", "moving_epe_mm"),
    ("C++時間 [s]", "elapsed_seconds"),
    ("深度0 RMSE [mm]", "depth0_rmse_mm"),
    ("深度1 RMSE [mm]", "depth1_rmse_mm"),
    ("幾何平均 [mm]", "geometry_mean_mm"),
)

#: 凍結集計 summary.json のキー（split, method）。
SUMMARY_KEYS = {
    ("retest", "v02"): "retest_v02",
    ("retest", "standard"): "retest_standard",
    ("heldout", "v02"): "heldout_v02",
    ("heldout", "standard"): "heldout_standard",
}


def collect(data_root: Path = DATA_ROOT) -> dict[tuple[str, str], list[dict]]:
    """split と手法ごとに、ケース単位の凍結メトリクスを集める。"""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for case in iter_cases(data_root):
        for method in case.methods():
            key = (case.split, method)
            grouped.setdefault(key, []).append(case.documented(method)["metrics"])
    return grouped


def aggregate(grouped: dict[tuple[str, str], list[dict]]) -> dict[tuple[str, str], dict]:
    """平均と標本標準偏差を計算する（理論書 §6.3 の集計規則）。"""
    result: dict[tuple[str, str], dict] = {}
    for key, records in grouped.items():
        stats: dict[str, dict[str, float | int]] = {"n": len(records)}
        for _, json_key in COLUMNS:
            values = np.array([r[json_key] for r in records], dtype=float)
            stats[json_key] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        result[key] = stats
    return result


def compare_with_summary(stats: dict[tuple[str, str], dict], summary: dict) -> list[dict]:
    """凍結集計と突合する。"""
    rows = []
    for key, summary_key in SUMMARY_KEYS.items():
        if key not in stats or summary_key not in summary:
            continue
        for _, json_key in COLUMNS:
            if json_key not in summary[summary_key]:
                continue
            documented = summary[summary_key][json_key]
            recomputed = stats[key][json_key]
            for field in ("mean", "sd"):
                rows.append(
                    {
                        "split": key[0],
                        "method": key[1],
                        "metric": json_key,
                        "field": field,
                        "recomputed": recomputed[field],
                        "documented": documented[field],
                        "difference": abs(recomputed[field] - documented[field]),
                    }
                )
    return rows


def write_main_table(stats: dict[tuple[str, str], dict], path: Path) -> None:
    """主比較表（理論書 table_main.tex と同じ構成）を Markdown で書く。"""
    lines = [
        "| 区分 | 手法 | n | 共通可視 [mm] | 動体 [mm] | C++時間 [s] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split, label in (("retest", "再試験"), ("heldout", "新規条件")):
        for method, method_label in (("v02", "第0.2版"), ("standard", "標準版")):
            stats_ = stats.get((split, method))
            if stats_ is None:
                continue
            cells = []
            for _, json_key in COLUMNS[:3]:
                mean = stats_[json_key]["mean"]
                sd = stats_[json_key]["sd"]
                cells.append(f"{mean:.3f} ± {sd:.3f}")
            lines.append(f"| {label} | {method_label} | {stats_['n']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_check(rows: list[dict], path: Path) -> None:
    worst = max((r["difference"] for r in rows), default=0.0)
    lines = [
        "# 集計の再計算と凍結集計の突合",
        "",
        f"- 突合項目: {len(rows)}",
        f"- 最大差: {worst:.3e}",
        "" if worst <= 1e-9 else f"- **最大差 {worst:.3e}（要確認）**",
        "",
        "| 区分 | 手法 | 指標 | 量 | 再計算 | 凍結値 | 差 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {} | {} | `{}` | {} | {:.9f} | {:.9f} | {:.3e} |".format(
                row["split"],
                row["method"],
                row["metric"],
                row["field"],
                row["recomputed"],
                row["documented"],
                row["difference"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grouped = collect(args.data_root)
    stats = aggregate(grouped)
    summary = read_json(args.data_root / "summary.json")
    rows = compare_with_summary(stats, summary)

    write_main_table(stats, args.out / "main_table.md")
    write_check(rows, args.out / "aggregate_check.md")
    (args.out / "aggregate_stats.json").write_text(
        json.dumps({f"{k[0]}/{k[1]}": v for k, v in stats.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    worst = max((r["difference"] for r in rows), default=0.0)
    print(f"突合項目: {len(rows)}  最大差: {worst:.3e}")
    print(f"wrote {args.out / 'main_table.md'}")
    print(f"wrote {args.out / 'aggregate_check.md'}")
    return 0 if worst <= 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
