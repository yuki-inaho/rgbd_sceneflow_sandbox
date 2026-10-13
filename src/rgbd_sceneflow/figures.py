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


# CLI（`just figures`）では画面が無いので Agg を使う。Notebook では既存の
# インラインバックエンドを壊さないよう、切り替えを行わない。
def _running_in_ipython() -> bool:
    """IPython（Jupyter）上で動いているかを返す。"""
    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover - IPython 無しの環境
        return False
    return get_ipython() is not None


if not _running_in_ipython():
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

#: Middlebury の色相環を構成する区分数（flow-code と同じ配分）。
FLOW_WHEEL_SEGMENTS = (15, 6, 4, 11, 13, 6)
#: 光流の色相環で使う既定の最大値 [px]。
FLOW_CODING_MAX_PX = 1.0
#: XYZ→RGB 合成で小さい変位を持ち上げる指数。
COMPONENT_GAMMA = 0.6
#: 矢印オーバーレイの間引き間隔 [画素]。
QUIVER_STRIDE = 6


def _finish(fig: Figure, path: Path | None, dpi: int = 140) -> Figure:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return fig


def show(fig: Figure) -> None:
    """Notebook で図を 1 枚だけ表示して閉じる（CLI では何もしない）。

    セルの最後の式に ``Figure`` を置くと formatter とインラインのフックで
    二重に埋め込まれるため、Notebook からはこの関数を使う。
    """
    try:
        from IPython.display import display
    except ImportError:  # pragma: no cover - IPython 無しの環境
        return
    display(fig)
    plt.close(fig)


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


def figure_ground_truth(
    case: Case,
    path: Path | None = None,
    max_magnitude: float = FLOW_CODING_MAX_PX,
) -> Figure:
    """真値（正解）の一式：画像・深度・光流・シーンフロー・可視性・動体。

    入力の確認に使う。推定結果と混ぜず、正解だけを並べる。
    """
    inputs = case.inputs()
    gt_flow = inputs["gt_flow"]
    gt_scene = inputs["gt_scene_flow"]
    magnitude = np.linalg.norm(gt_scene, axis=-1) * 1000.0
    limit = float(np.nanpercentile(magnitude, 99))
    visibility = np.where(inputs["gt_outside"], 0.0, np.where(inputs["gt_occluded"], 0.45, 1.0))
    moving = np.where(inputs["gt_moving"], 1.0, 0.35)

    panels = [
        (inputs["gray0"], "gray", 0.0, 1.0),
        (inputs["gray1"], "gray", 0.0, 1.0),
        (optical_flow_to_rgb(gt_flow, max_magnitude), None, None, None),
        (scene_flow_to_rgb(gt_scene), None, None, None),
        (magnitude, "turbo", 0.0, limit),
        (visibility, "gray", 0.0, 1.0),
        (moving, "gray", 0.0, 1.0),
        (inputs["depth0"], "turbo", 0.5, 2.0),
    ]
    titles = [
        "入力画像（時刻 0）",
        "入力画像（時刻 1）",
        f"真値の光流（Middlebury 色相環、最大 {max_magnitude:.1f} px）",
        "真値の変位（XYZ→RGB）",
        f"真値の変位の大きさ [mm]（0—{limit:.1f} mm）",
        "可視性（白=可視、灰=遮蔽、黒=画面外）",
        "動体（白=動体、灰=静止）",
        "観測深度（時刻 0）[m]",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.4), constrained_layout=True)
    for ax, (image, cmap, vmin, vmax), title in zip(axes.ravel(), panels, titles):
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{case.key}：正解（真値）の保存配列", fontsize=11)
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


def flow_color_wheel(segments: tuple[int, ...] = FLOW_WHEEL_SEGMENTS) -> np.ndarray:
    """Middlebury の色相環（flow-code と同じ 55 段）を [0, 1] の RGB で返す。"""
    red_yellow, yellow_green, green_cyan, cyan_blue, blue_magenta, magenta_red = segments
    wheel = np.zeros((sum(segments), 3))
    index = 0
    wheel[index : index + red_yellow, 0] = 1.0
    wheel[index : index + red_yellow, 1] = np.arange(red_yellow) / red_yellow
    index += red_yellow
    wheel[index : index + yellow_green, 0] = 1 - np.arange(yellow_green) / yellow_green
    wheel[index : index + yellow_green, 1] = 1.0
    index += yellow_green
    wheel[index : index + green_cyan, 1] = 1.0
    wheel[index : index + green_cyan, 2] = np.arange(green_cyan) / green_cyan
    index += green_cyan
    wheel[index : index + cyan_blue, 1] = 1 - np.arange(cyan_blue) / cyan_blue
    wheel[index : index + cyan_blue, 2] = 1.0
    index += cyan_blue
    wheel[index : index + blue_magenta, 2] = 1.0
    wheel[index : index + blue_magenta, 0] = np.arange(blue_magenta) / blue_magenta
    index += blue_magenta
    wheel[index : index + magenta_red, 2] = 1 - np.arange(magenta_red) / magenta_red
    wheel[index : index + magenta_red, 0] = 1.0
    return wheel


def optical_flow_to_rgb(
    flow: np.ndarray,
    max_magnitude: float = FLOW_CODING_MAX_PX,
    wheel: np.ndarray | None = None,
) -> np.ndarray:
    """光流を Middlebury 流の色相環で RGB 化する（Baker らの flow-code と同じ手順）。

    向きを色相、大きさを明度（大きいほど白に寄る）で表す。欠測は黒。
    """
    wheel = flow_color_wheel() if wheel is None else wheel
    size = wheel.shape[0]
    finite = np.isfinite(flow).all(axis=-1)
    u = np.where(finite, flow[..., 0], 0.0)
    v = np.where(finite, flow[..., 1], 0.0)
    magnitude = np.hypot(u, v)
    position = (np.arctan2(v, u) / np.pi + 1.0) / 2.0 * (size - 1)
    lower = np.clip(np.floor(position).astype(np.int64), 0, size - 1)
    upper = (lower + 1) % size
    weight = position - lower
    rgb = np.empty((*flow.shape[:2], 3))
    for channel in range(3):
        rgb[..., channel] = (1 - weight) * wheel[lower, channel] + weight * wheel[upper, channel]
    normalized = np.clip(magnitude / max(max_magnitude, 1e-9), 0.0, 1.0)
    rgb = 1.0 - normalized[..., None] * (1.0 - rgb)
    return np.where(finite[..., None], rgb, 0.0)


def scene_flow_to_rgb(
    scene_flow: np.ndarray,
    scales: tuple[float, float, float] | None = None,
    percentiles: tuple[float, float] = (1.0, 99.0),
    gamma: float = COMPONENT_GAMMA,
) -> np.ndarray:
    """3 次元変位を成分ごとに RGB へ写す（x→R、y→G、z→B。0 は中間色）。

    各軸を 99 パーセンタイルで正規化し、小さい変位を持ち上げてから符号付きで写す。
    正の変位は明るく、負の変位は暗く、変位なしは 0.5 の中間色になる。
    """
    scales = scales or tuple(
        float(np.nanpercentile(np.abs(scene_flow[..., axis]), percentiles[1])) for axis in range(3)
    )
    rgb = np.empty((*scene_flow.shape[:2], 3))
    for axis, scale in enumerate(scales):
        ratio = np.clip(scene_flow[..., axis] / max(scale, 1e-9), -1.0, 1.0)
        boosted = np.sign(ratio) * np.abs(ratio) ** gamma
        rgb[..., axis] = 0.5 + 0.5 * boosted
    return np.where(np.isfinite(scene_flow).all(axis=-1)[..., None], rgb, 0.0)


def _point_cloud_geometry(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """点群と矢印の座標・マスクを共通形式で作る。"""
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
    return {"xyz": xyz, "depth": depth, "cloud": cloud, "arrows": arrows, "points": xyz[cloud]}


def _draw_point_cloud(
    ax,
    geometry: dict[str, np.ndarray],
    scene: np.ndarray,
    *,
    arrow_color: str,
    arrow_scale: float = ARROW_SCALE,
) -> None:
    """1 つの 3D 軸へ点群と変位ベクトルを描く（座標は x, z, -y）。"""
    xyz, points, arrows, depth, cloud = (
        geometry["xyz"],
        geometry["points"],
        geometry["arrows"],
        geometry["depth"],
        geometry["cloud"],
    )
    norm = matplotlib.colors.Normalize(vmin=0.5, vmax=2.0, clip=True)
    ax.scatter(
        points[:, 0],
        points[:, 2],
        -points[:, 1],
        c=depth[cloud],
        cmap=plt.get_cmap("turbo"),
        norm=norm,
        s=4.5,
        linewidths=0,
    )
    ax.quiver(
        xyz[arrows, 0],
        xyz[arrows, 2],
        -xyz[arrows, 1],
        scene[arrows, 0] * arrow_scale,
        scene[arrows, 2] * arrow_scale,
        -scene[arrows, 1] * arrow_scale,
        color=arrow_color,
        linewidth=0.8,
        alpha=0.9,
        arrow_length_ratio=0.3,
    )
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("z [m]（奥行）", fontsize=8)
    ax.set_zlabel("-y [m]（上）", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 2].min(), points[:, 2].max())
    ax.set_zlim(-points[:, 1].max(), -points[:, 1].min())


def figure_scene_flow_3d(
    case: Case,
    path: Path | None = None,
    variants: tuple[tuple[str, str | None], ...] = DEFAULT_3D_VARIANTS,
) -> Figure:
    """点群とシーンフローを 3D で並べる。

    視点・表示範囲・間引き・矢印倍率を全パネルで揃え、動体だけを描く。
    座標は (x, z, -y) で、奥行き +z が画面の奥、-y が画面の上になる。
    """
    inputs = case.inputs()
    geometry = _point_cloud_geometry(inputs)
    gt = inputs["gt_scene_flow"]
    moving = inputs["gt_labels0"] >= OBJECT_LABEL_MIN
    finite = np.isfinite(inputs["depth0"])

    fig = plt.figure(figsize=(4.9 * len(variants), 4.9), constrained_layout=True)
    for index, (label, method) in enumerate(variants, start=1):
        scene = gt if method is None else case.result(method)["scene_flow"]
        ax = fig.add_subplot(1, len(variants), index, projection="3d")
        _draw_point_cloud(ax, geometry, scene, arrow_color="#c0392b" if method else "#111111")
        error = float(np.linalg.norm(scene - gt, axis=-1)[moving & finite].mean() * 1000.0)
        ax.set_title(f"{label}：動体 EPE {error:.2f} mm", fontsize=10)
        ax.view_init(elev=18, azim=-72)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0.5, vmax=2.0), cmap="turbo"),
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


def figure_scene_flow_3d_views(
    case: Case,
    path: Path | None = None,
    method: str = "standard",
    views: tuple[tuple[float, float, str], ...] = (
        (18, -72, "斜め（既定）"),
        (0, -90, "正面"),
        (72, -90, "俯瞰"),
        (18, 18, "斜め裏"),
    ),
) -> Figure:
    """同じ推定を複数の視点から描く（ビューアのように眺めるための並べ方）。"""
    inputs = case.inputs()
    geometry = _point_cloud_geometry(inputs)
    scene = case.result(method)["scene_flow"]
    gt = inputs["gt_scene_flow"]
    moving = inputs["gt_labels0"] >= OBJECT_LABEL_MIN
    finite = np.isfinite(inputs["depth0"])
    error = float(np.linalg.norm(scene - gt, axis=-1)[moving & finite].mean() * 1000.0)

    fig = plt.figure(figsize=(4.6 * len(views), 4.6), constrained_layout=True)
    for index, (elevation, azimuth, label) in enumerate(views, start=1):
        ax = fig.add_subplot(1, len(views), index, projection="3d")
        _draw_point_cloud(ax, geometry, scene, arrow_color="#c0392b")
        ax.set_title(f"{label}（elev={elevation:.0f}°, azim={azimuth:.0f}°）", fontsize=9)
        ax.view_init(elev=elevation, azim=azimuth)

    fig.suptitle(
        f"{case.key} / {method}：動体 EPE {error:.2f} mm を 4 視点で（矢印は ×{ARROW_SCALE:.0f}）",
        fontsize=11,
    )
    return _finish(fig, path)


def figure_optical_flow_styles(
    case: Case,
    path: Path | None = None,
    method: str = "standard",
    max_magnitude: float = FLOW_CODING_MAX_PX,
) -> Figure:
    """光流の定番表示（Middlebury 色相環・矢印オーバーレイ・誤差）。"""
    inputs = case.inputs()
    observed = inputs["flow_observed"]
    estimated = case.result(method)["flow"]
    gt = inputs["gt_flow"]
    image = inputs["gray0"]
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    moving = inputs["gt_labels0"] >= OBJECT_LABEL_MIN
    finite = np.isfinite(observed).all(axis=-1) & np.isfinite(estimated).all(axis=-1)
    error = np.linalg.norm(estimated - gt, axis=-1)

    fig = plt.figure(figsize=(11.5, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.0))

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title("入力画像（時刻 0）", fontsize=10)
    ax.axis("off")

    for column, (label, flow) in enumerate(
        (("観測光流", observed), (f"推定光流（{method}）", estimated)), start=1
    ):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(optical_flow_to_rgb(flow, max_magnitude))
        ax.set_title(f"{label}：Middlebury 色相環（最大 {max_magnitude:.1f} px）", fontsize=10)
        ax.axis("off")

    ax = fig.add_subplot(grid[1, 0])
    step = QUIVER_STRIDE
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.quiver(
        xx[::step, ::step][moving[::step, ::step]],
        yy[::step, ::step][moving[::step, ::step]],
        observed[::step, ::step, 0][moving[::step, ::step]],
        observed[::step, ::step, 1][moving[::step, ::step]],
        color="#00d0ff",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
    )
    ax.quiver(
        xx[::step, ::step][moving[::step, ::step]],
        yy[::step, ::step][moving[::step, ::step]],
        estimated[::step, ::step, 0][moving[::step, ::step]],
        estimated[::step, ::step, 1][moving[::step, ::step]],
        color="#ff3b30",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
    )
    ax.set_title(f"矢印（動体・{step} 画素ごと）：水色=観測、赤=推定", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(grid[1, 1])
    color = ax.imshow(np.where(finite, error, np.nan), cmap="turbo", vmin=0.0, vmax=1.0)
    ax.set_title("光流の誤差 [px]（真値との差）", fontsize=10)
    ax.axis("off")
    fig.colorbar(color, ax=ax, fraction=0.046, pad=0.02, label="画素")

    ax = fig.add_subplot(grid[1, 2])
    size = 160
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    radius = np.hypot(grid_x - size / 2, grid_y - size / 2) / (size / 2)
    wheel_flow = np.stack(
        ((grid_x - size / 2) / (size / 2) * max_magnitude, (grid_y - size / 2) / (size / 2) * max_magnitude),
        axis=-1,
    )
    wheel_flow[radius > 1.0] = 0.0
    ax.imshow(optical_flow_to_rgb(wheel_flow, max_magnitude))
    ax.set_title("色相環の読み方（向き＝色、大きさ＝明度）", fontsize=10)
    ax.axis("off")

    fig.suptitle(f"{case.key}：光流の可視化スタイル", fontsize=11)
    return _finish(fig, path)


def figure_scene_flow_styles(
    case: Case,
    path: Path | None = None,
    variants: tuple[tuple[str, str | None], ...] = DEFAULT_3D_VARIANTS,
) -> Figure:
    """3 次元変位の定番表示（XYZ→RGB 合成と大きさ）。"""
    inputs = case.inputs()
    gt = inputs["gt_scene_flow"]
    finite = np.isfinite(inputs["depth0"]) & np.isfinite(gt).all(axis=-1)
    moving = inputs["gt_labels0"] >= OBJECT_LABEL_MIN

    scales = tuple(float(np.nanpercentile(np.abs(gt[..., axis]), 99)) for axis in range(3))
    magnitudes = {
        label: np.linalg.norm(gt if method is None else case.result(method)["scene_flow"], axis=-1)
        for label, method in variants
    }
    limit = float(np.nanpercentile(magnitudes["正解"][finite], 99)) * 1000.0

    fig = plt.figure(figsize=(4.2 * len(variants), 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, len(variants))
    for column, (label, method) in enumerate(variants):
        scene = gt if method is None else case.result(method)["scene_flow"]
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(scene_flow_to_rgb(scene, scales=scales))
        ax.set_title(f"{label}：変位の XYZ→RGB", fontsize=10)
        ax.axis("off")

        ax = fig.add_subplot(grid[1, column])
        magnitude = np.linalg.norm(scene, axis=-1) * 1000.0
        image = ax.imshow(np.where(finite, magnitude, np.nan), cmap="turbo", vmin=0.0, vmax=limit)
        error = float(np.linalg.norm(scene - gt, axis=-1)[moving & finite].mean() * 1000.0)
        ax.set_title(f"{label}：変位の大きさ（動体 EPE {error:.2f} mm）", fontsize=10)
        ax.axis("off")

    fig.colorbar(
        image,
        ax=fig.axes,
        orientation="horizontal",
        fraction=0.04,
        pad=0.03,
        label=f"変位の大きさ [mm]（0—{limit:.1f} mm）",
    )
    fig.suptitle(
        f"{case.key}：3 次元変位の XYZ→RGB（x=赤、y=緑、z=青。各軸 "
        f"{scales[0] * 1000:.1f} / {scales[1] * 1000:.1f} / {scales[2] * 1000:.1f} mm で飽和）",
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
        args.out / "ground_truth.png",
        args.out / "error_maps.png",
        args.out / "paired_errors.png",
    ]
    figure_inputs(heldout[:2], written[0])
    if heldout:
        figure_ground_truth(heldout[0], written[1])
    figure_error_maps(heldout, written[2])
    figure_paired_errors(heldout, written[3])
    case = ablation_case(args.data_root)
    if case is not None:
        written.append(args.out / "speed_accuracy.png")
        figure_speed_accuracy(case, written[-1])

    view3d = next((c for c in heldout if c.key.endswith("gallery_scene101")), None)
    if view3d is not None:
        written.append(args.out / "scene_flow_3d.png")
        figure_scene_flow_3d(view3d, written[-1])
        written.append(args.out / "scene_flow_3d_views.png")
        figure_scene_flow_3d_views(view3d, path=written[-1])
        written.append(args.out / "optical_flow_styles.png")
        figure_optical_flow_styles(view3d, path=written[-1])
        written.append(args.out / "scene_flow_styles.png")
        figure_scene_flow_styles(view3d, path=written[-1])

    for path in written:
        print(f"wrote {path} ({path.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
