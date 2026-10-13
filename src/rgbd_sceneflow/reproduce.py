"""70 組の凍結メトリクスを再計算し、突合報告書を出力する。

uv run python -m rgbd_sceneflow.reproduce
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import metrics as metrics_mod
from .archive import OUTPUTS_DIR, Case, iter_cases, result_count

#: 前版との比較に使う手法名（v02 は配布時の前版）。
BASELINE = "v02"


def compare_case(case: Case, method: str) -> list[metrics_mod.MetricStatus]:
    inputs = case.inputs()
    result = case.result(method)
    recomputed = metrics_mod.evaluate(inputs, result, case.metadata())["metrics"]
    documented = case.documented(method)["metrics"]
    return metrics_mod.compare(recomputed, documented, approx_keys=metrics_mod.APPROX_KEYS)


def build_report() -> dict:
    per_metric: dict[str, Counter] = defaultdict(Counter)
    records: list[dict] = []
    mismatches: list[dict] = []
    for case in iter_cases():
        for method in case.methods():
            statuses = compare_case(case, method)
            counts = Counter(s.status for s in statuses)
            records.append({"case": case.key, "method": method, "status_counts": dict(counts)})
            for status in statuses:
                per_metric[status.name][status.status] += 1
                if status.status == "mismatch":
                    mismatches.append(
                        {
                            "case": case.key,
                            "method": method,
                            "metric": status.name,
                            "recomputed": status.recomputed,
                            "documented": status.documented,
                            "difference": status.difference,
                        }
                    )
    summary = {name: dict(sorted(counter.items())) for name, counter in sorted(per_metric.items())}
    return {
        "records": len(records),
        "per_metric": summary,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:200],
        "detail": records,
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# 再現チェック報告（第0.3版アーカイブ）",
        "",
        "配布物には推定器のソースが含まれないため、再計算の対象は**保存済み推定出力から",
        "決まる評価量**である（解そのものは再実行できない）。",
        "",
        f"- 突合した記録: **{report['records']} 組**",
        f"- 不一致（mismatch）: **{report['mismatch_count']} 件**",
        "",
        "## メトリクス別の状態",
        "",
        "| メトリクス | exact | approx | not_recomputable | mismatch |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, counts in report["per_metric"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                name,
                counts.get("exact", 0),
                counts.get("approx", 0),
                counts.get("not_recomputable", 0),
                counts.get("mismatch", 0),
            )
        )
    if report["mismatches"]:
        lines += [
            "",
            "## 不一致の内訳（先頭のみ）",
            "",
            "| ケース | 手法 | メトリクス | 再計算 | 凍結値 | 差 |",
            "|---|---|---|---:|---:|---:|",
        ]
        for item in report["mismatches"]:
            lines.append(
                "| {} | {} | `{}` | {} | {} | {} |".format(
                    item["case"],
                    item["method"],
                    item["metric"],
                    _fmt(item["recomputed"]),
                    _fmt(item["documented"]),
                    _fmt(item["difference"]),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1e-3:
        return f"{value:.6g}"
    return f"{value:.3e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=OUTPUTS_DIR / "reproduction_report.json")
    parser.add_argument("--markdown", type=Path, default=OUTPUTS_DIR / "reproduction_report.md")
    args = parser.parse_args()

    args.json.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.markdown)

    print(f"records            : {report['records']}（アーカイブの手法 json 総数 {result_count()}）")
    print(f"mismatches         : {report['mismatch_count']}")
    print(f"report (json)      : {args.json}")
    print(f"report (markdown)  : {args.markdown}")
    exact = sum(c.get("exact", 0) for c in report["per_metric"].values())
    approx = sum(c.get("approx", 0) for c in report["per_metric"].values())
    internal = sum(c.get("not_recomputable", 0) for c in report["per_metric"].values())
    print(f"exact={exact}  approx={approx}  not_recomputable={internal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
