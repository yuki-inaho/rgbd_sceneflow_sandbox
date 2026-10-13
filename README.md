# rgbd_sceneflow — 境界を保つ RGB-D シーンフロー補正（第0.3版）再現・検証ラボ

配布された一式を展開し、**環境を組み直して「どこまで実行再現できるか」を検査する**ためのラボです。

| 入力（`~/Downloads/`） | 中身 | 展開先 |
|---|---|---|
| `scene_flow_v03_results.zip` (168 MB) | 結果配列 `*.npz`・凍結メトリクス `*.json`・集計 CSV | `data/v03/` |
| `scene_flow_v03_theory_sources.zip` (1.0 MB) | 理論書 `theory_ja.tex`・表・図 | `docs/` |
| `scene_flow_v03_code (1).zip` (15 MB) | 実装（C++17 コア・Python ラッパ・試験・v0.2/v0.1 比較用） | `scene_flow_primitives/` |
| `ChatGPT-シーンフロー定式化検討.md` | 定式化の検討ログ（関連資料） | `docs/` |

## 構成

```
rgbd_sceneflow/
├── data/v03/               配布結果の展開先（`just extract` で復元、Git 追跡外）
├── docs/                   理論書（LaTeX と PDF）と関連資料
│   ├── theory_ja.tex       第0.3版 理論仕様（正本）
│   ├── theory_ja.pdf       上記のコンパイル結果（26 ページ）
│   ├── table_*.tex, numbers.tex, figures/, CHANGELOG_ja.md
│   └── formulation_discussion_ja.md
├── scene_flow_primitives/  配布された実装（**現行版のみ**、ビルド・実行用）
│   ├── cpp/                C++17 コア（solver / matcher / rigid / renderer / bindings）
│   ├── src/scene_flow_refinement/  Python ラッパ（ctypes 経由）
│   ├── scripts/           集計と再計算の検証
│   └── tests/              自動試験 93 件
├── src/rgbd_sceneflow/     検証ラボ（保存配列からの再計算・突合・作図）
├── tests/                  24 件（うち 3 件は配布実装との突き合わせ）
├── notebooks/              実行済み Notebook（画像を埋め込み）
├── figures/ outputs/       生成物（Git 追跡外）
└── temp/                   作業書
```

## 使い方

```bash
just extract        # ~/Downloads の zip から data/ と docs/ を復元
uv sync             # 検証ラボの環境（Python 3.13）
uv run pytest       # 24 passed
just reproduce      # 70 組の突合報告書（outputs/reproduction_report.md）
just tables         # 集計表の再構成（outputs/main_table.md）
just figures        # 図の再構成（figures/）
just notebook       # Notebook を実行（図がセル内に埋め込まれる）
just docs           # 理論書を LuaLaTeX でビルド（docs/theory_ja.pdf）

just build          # 配布実装をビルド（C++17 コア）
just upstream-test  # 配布実装の自動試験（93 件）
just verify-upstream  # 保存済み推定 1 件を再計算し、配列が完全一致するか確認
```

### 旧版の扱い

v0.1 / v0.2 の**実装ツリーは削除**しました（`baseline_v02/` `legacy_v01/`
`data/reference_v02/`、およびそれ専用のビルド・比較スクリプトと v0.1 互換試験）。
旧版の*測定値*は検証対象なので、保存結果 `data/v03/` に含まれる `v02` の JSON/NPZ は残しています。
したがって主比較（第0.2版 vs 標準版）の検証はそのまま可能で、
旧版を**ビルドし直して**比較を再生成することだけができません。

## 再現の結果（二段階）

### 1. 実行再現 — 配布実装をビルドして走らせる

| 検査 | 結果 | 記録との比較 |
|---|---|---|
| `just upstream-test` | **93 passed** | 配布時の記録は 94 件（旧版 v0.1 の ABI 互換試験 1 件は、旧版コードの削除に伴い対象外） |
| `just verify-upstream`（`retest/gallery_dis_seed11`, standard） | **14 フィールドすべて `max_abs = 0.0`** | `clean_reproduction.json` と同じ（時間のみ 11.68s vs 記録 10.70〜10.75s） |

### 2. 評価量の再現 — 保存配列だけから独立に再計算

70 組（21 ケース × 手法）の凍結メトリクスと突合した結果です。

| 状態 | 件数 | 内容 |
|---|---:|---|
| exact | 4,270 | 丸め誤差の範囲で一致（相対 1e-12） |
| **mismatch** | **0** | 説明できない不一致なし |
| not_recomputable | 1,055 | 推定器内部の量（処理時間・候補数・固定点残差など） |

評価集合 `V`、端点誤差と集合別内訳（動体・静止域・境界・内部・遮蔽・物体別）、分位点と bad 率、
深度 RMSE（全体・観測有効域・欠測域・物体）、支持率、幾何残差、光学フロー誤差、
マッチ再投影ギャップ、移動上限の検査まで含みます。

定義はアーカイブの数値だけから探索して復元したあと、配布実装
`src/scene_flow_refinement/primitive_metrics.py` と突き合わせて確定しました。
両者の一致は `tests/test_upstream.py` が全 70 組で検査します（最大差 **0.0**）。

### 再現できないもの

推定器の内部状態（`elapsed_seconds`、`candidate_*`、`fixed_point_*`、`adjoint_error`、
`rigid_components`、`inner_iterations`）は保存配列からは決まりません。時間は実行環境依存で、
本機の再実行では 11.7 秒（記録 10.7 秒）でした。

## 出所

- アーカイブの記録では、環境は Python 3.13.5 / numpy 2.3.5 / scipy 1.17.0 /
  matplotlib 3.10.8 / OpenCV 4.13.0.92、CPU 単一スレッド（GPU 不使用）。
- 本ラボの `pyproject.toml` はこの版に合わせています（`uv sync` の解決では scipy 1.18.1）。
- 配布物の数値はすべて人工データに対する測定結果で、実センサ・実データへの一般化を
  主張するものではありません（理論書 結論節）。本ラボもそれを主張しません。
