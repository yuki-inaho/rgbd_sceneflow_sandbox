"""保存配列からの図の再構成。

理論書の図は生成コードが配布されていないため、ここでは**保存配列だけ**から
同等の情報を持つ図を作り直す（数値の再現ではなく、内容の再構成）。

各 ``figure_*`` は ``Figure`` を返す。``path`` を渡すと保存して閉じる
（Notebook からは ``path`` を渡さずに呼び、セル内にインライン表示する）。

    uv run python -m rgbd_sceneflow.figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .archive import DATA_ROOT, FIGURES_DIR, Case, iter_cases
from .metrics import OBJECT_LABEL_MIN, common_visible_set, epe_mm

# 日本語ラベルを描画できるフォントを優先する（無い環境では DejaVu にフォールバック）。
matplotlib.rcParams["font.family"] = [
    "Noto Sans CJK JP",
    "IPAexGothic",
    "IPAGothic",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

#: 誤差表示の飽和値 [mm]（理論書の図と同じ）。
ERROR_SATURATION_MM = 20.0


def _finish(fig: Figure, path: Path | None, dpi: int = 140) -> Figure:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return fig


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """深度を 0.5--2.0 m の turbo 色で RGB 化する（欠測は暗色）。"""
    valid = np.isfinite(depth) & (depth > 0)
    norm = matplotlib.colors.Normalize(vmin=0.5, vmax=2.0, clip=True)
    rgb = plt.get_cmap("turbo")(norm(np.where(valid, depth, 0.5)))[..., :3]
    return np.where(valid[..., None], rgb, 0.15)


def figure_inputs(cases: list[Case], path: Path | None = None) -> Figure:
    """入力例（灰度2枚 + 深度）。"""
    fig, axes = plt.subplots(len(cases), 3, figsize=(9.5, 3.1 * len(cases)), constrained_layout=True)
    for row, case in enumerate(cases):
        inputs = case.inputs()
        panels = [
            (inputs["gray0"], "gray 0", plt.get_cmap("gray")),
            (inputs["gray1"], "gray 1", plt.get_cmap("gray")),
            (depth_to_rgb(inputs["depth0"]), "depth 0 [m]", None),
        ]
        for col, (image, title, cmap) in enumerate(panels):
            ax = axes[row, col]
            if cmap is None:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_title(f"{case.key} / {title}", fontsize=9)
            ax.axis("off")
    fig.suptitle("保存配列から再構成した入力（160x120、拡大は推定解像度の増加ではない）", fontsize=10)
    return _finish(fig, path)


def figure_error_maps(cases: list[Case], path: Path | None = None) -> Figure:
    """前版（v02）と標準版の端点誤差マップ。"""
    methods = ["v02", "standard"]
    fig, axes = plt.subplots(
        len(methods), len(cases), figsize=(2.6 * len(cases), 5.4), constrained_layout=True
    )
    for row, method in enumerate(methods):
        for col, case in enumerate(cases):
            inputs = case.inputs()
            errors = epe_mm(case.result(method)["scene_flow"], inputs["gt_scene_flow"])
            ax = axes[row, col]
            ax.imshow(
                np.clip(errors, 0.0, ERROR_SATURATION_MM),
                cmap="magma",
                vmin=0.0,
                vmax=ERROR_SATURATION_MM,
            )
            if row == 0:
                ax.set_title(case.name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(method, fontsize=10)
    fig.suptitle(f"3次元端点誤差 [mm]（{ERROR_SATURATION_MM:.0f} mm で飽和、遮蔽を含む全画素）", fontsize=10)
    return _finish(fig, path)


def figure_paired_errors(cases: list[Case], path: Path | None = None) -> Figure:
    """入力ごとの前版・標準版の端点誤差。"""
    labels, old, new = [], [], []
    for case in cases:
        inputs = case.inputs()
        common = common_visible_set(inputs)
        for method, bucket in (("v02", old), ("standard", new)):
            if not case.has(method):
                bucket.append(float("nan"))
                continue
            errors = epe_mm(case.result(method)["scene_flow"], inputs["gt_scene_flow"])
            bucket.append(float(errors[common].mean()))
        labels.append(f"{case.split[:3]}/{case.name}\n(seed {case.metadata()['seed']})")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(labels)), 4.2), constrained_layout=True)
    ax.bar(x - 0.2, old, width=0.4, label="v0.2 (前版)", color="#7f8fa6")
    ax.bar(x + 0.2, new, width=0.4, label="standard (第0.3版)", color="#c0392b")
    ax.set_ylabel("共通可視集合の端点誤差 [mm]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("保存配列から再計算した端点誤差", fontsize=10)
    return _finish(fig, path)


def figure_speed_accuracy(case: Case, path: Path | None = None) -> Figure:
    """アブレーション入力の速度と精度（時間は保存値、誤差は再計算値）。"""
    inputs = case.inputs()
    common = common_visible_set(inputs)
    xs, ys, names = [], [], []
    for method in case.methods():
        errors = epe_mm(case.result(method)["scene_flow"], inputs["gt_scene_flow"])
        xs.append(case.documented(method)["metrics"]["elapsed_seconds"])
        ys.append(float(errors[common].mean()))
        names.append(method)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.scatter(xs, ys, s=28, color="#2c3e50")
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("C++ 処理時間 [s]（保存値）")
    ax.set_ylabel("共通可視領域の端点誤差 [mm]（再計算）")
    ax.grid(alpha=0.3)
    fig.suptitle(f"アブレーション入力 {case.key} の速度と精度", fontsize=10)
    return _finish(fig, path)


#: 3D ビューで矢印の長さに掛ける倍率（見やすさのため）。
ARROW_SCALE = 5.0
#: 3D ビューの点群・矢印の間引き間隔 [画素]。
CLOUD_STRIDE = 2
ARROW_STRIDE = 8
#: 3D ビューの比較対象（ラベル, 手法名）。手法名 None は正解。
DEFAULT_3D_VARIANTS = (("正解", None), ("前版", "v02"), ("補正版", "standard"))


def figure_scene_flow_3d(
    case: Case,
    variants: tuple[tuple[str, str | None], ...] = DEFAULT_3D_VARIANTS,
    path: Path | None = None,
) -> Figure:
    """点群とシーンフローを 3D で並べる。

    視点・表示範囲・間引き・矢印倍率を全パネルで揃え、動体だけを描く。
    座標は (x, z, -y) で、奥行き +z が画面の奥、-y が画面の上になる。
    """
    inputs = case.inputs()
    depth = inputs["depth0"]
    height, width = depth.shape
    yy, xx = np.mgrid[0:height, 0:width]
    fx, fy, cx, cy = inputs["intrinsics"]
    rays = np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1)
    xyz = depth[..., None] * rays

    moving = inputs["gt_labels0"] >= OBJECT_LABEL_MIN
    finite = np.isfinite(depth)
    cloud = moving & finite & (xx % CLOUD_STRIDE == 0) & (yy % CLOUD_STRIDE == 0)
    arrows = moving & finite & (xx % ARROW_STRIDE == 0) & (yy % ARROW_STRIDE == 0)
    points = xyz[cloud]

    norm = matplotlib.colors.Normalize(vmin=0.5, vmax=2.0, clip=True)
    cmap = plt.get_cmap("turbo")
    gt = inputs["gt_scene_flow"]

    fig = plt.figure(figsize=(4.9 * len(variants), 4.9), constrained_layout=True)
    for index, (label, method) in enumerate(variants, start=1):
        scene = gt if method is None else case.result(method)["scene_flow"]
        ax = fig.add_subplot(1, len(variants), index, projection="3d")
        ax.scatter(
            points[:, 0],
            points[:, 2],
            -points[:, 1],
            c=depth[cloud],
            cmap=cmap,
            norm=norm,
            s=4.5,
            linewidths=0,
        )
        ax.quiver(
            xyz[arrows, 0],
            xyz[arrows, 2],
            -xyz[arrows, 1],
            scene[arrows, 0] * ARROW_SCALE,
            scene[arrows, 2] * ARROW_SCALE,
            -scene[arrows, 1] * ARROW_SCALE,
            color="#c0392b" if method else "#111111",
            linewidth=0.8,
            alpha=0.9,
            arrow_length_ratio=0.3,
        )
        error = float(np.linalg.norm(scene - gt, axis=-1)[moving & finite].mean() * 1000.0)
        ax.set_title(f"{label}：動体 EPE {error:.2f} mm", fontsize=10)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("z [m]（奥行）", fontsize=8)
        ax.set_zlabel("-y [m]（上）", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=18, azim=-72)
        ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        ax.set_ylim(points[:, 2].min(), points[:, 2].max())
        ax.set_zlim(-points[:, 1].max(), -points[:, 1].min())

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=fig.axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.04,
        label="観測深度 [m]（点群の色）",
    )
    fig.suptitle(
        f"{case.key}：動体の点群と 3 次元変位（矢印は ×{ARROW_SCALE:.0f}、{ARROW_STRIDE} 画素ごと）",
        fontsize=11,
    )
    return _finish(fig, path)


def ablation_case(data_root: Path = DATA_ROOT) -> Case | None:
    for case in iter_cases(data_root):
        if case.split == "ablation":
            return case
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()

    heldout = [c for c in iter_cases(args.data_root) if c.split == "heldout"]
    written = [
        args.out / "inputs.png",
        args.out / "error_maps.png",
        args.out / "paired_errors.png",
    ]
    figure_inputs(heldout[:2], written[0])
    figure_error_maps(heldout, written[1])
    figure_paired_errors(heldout, written[2])
    case = ablation_case(args.data_root)
    if case is not None:
        written.append(args.out / "speed_accuracy.png")
        figure_speed_accuracy(case, written[-1])

    view3d = next((c for c in heldout if c.key.endswith("gallery_scene101")), None)
    if view3d is not None:
        written.append(args.out / "scene_flow_3d.png")
        figure_scene_flow_3d(view3d, DEFAULT_3D_VARIANTS, written[-1])

    for path in written:
        print(f"wrote {path} ({path.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
