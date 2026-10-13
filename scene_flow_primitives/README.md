# 灰度・深度シーンフロー補正 v0.3

C++17による計算・解析的描画と、Pythonによる実験・可視化を組み合わせた研究用実装です。
uvで環境を管理します。学習済みモデル、GPU、正解の物体分割は推定器に必要ありません。
**固定カメラ・位置合わせ済みRGB-D画像・推定済みフロー**が入力です。

理論、単位、全目的関数、近接写像、存在・凸性の仮定、有限反復の限界は
リポジトリ直下の `../docs/theory_ja.pdf` と `../docs/theory_ja.tex` に記載しています。
成功条件だけでなく、剛体仮定が有害になる同深度の独立運動、観測不能な球の回転も評価しています。

## v0.2からの変更

対応探索を階層化し、一時配列の動的確保と灰度の三重計算を除去しました。
疎作用素の計算を融合し、双対初期値の再利用と反復数変更を切り替えられるようにしました。
幾何残差にはベクトルHuber損失を、観測深度の連結領域には頑健に当てはめた剛体運動の**柔らかい補助項**を追加しています。

補助項の重みは観測からの便宜的な信頼度であり、校正済みの正答確率ではありません。
元の画像・深度の情報を再利用するため、独立の観測を追加したベイズ推定でもありません。
境界付き二階正則化は、標準的な対称勾配TGVと区別した全勾配型です。
対角前処理そのものは既にv0.2に存在し、本版で初めて導入したものではありません。

## 導入と試験

C++17コンパイラを用意し、プロジェクト直下で実行します。

```bash
uv sync
uv run python -m pytest -q
```

実行環境では外部パッケージのDNS解決に失敗しました。uvの環境内で既存の依存を利用し、
`uv pip install --no-build-isolation --no-deps -e .` によるビルド・実行を確認しています。
新規ダウンロードを伴う `uv sync` の成功や、解決済みの `uv.lock` の存在は主張しません。
使用した版は `requirements-verified.txt` と結果版の `environment.json` にあります。

## 完全な最小例

```python
from pathlib import Path
import json
import numpy as np
from scene_flow_refinement import Config, refine
from scene_flow_refinement.primitives import generate_primitives

output = Path("outputs/example_v03")
output.mkdir(parents=True, exist_ok=True)
data, metadata = generate_primitives(
    layout="gallery", seed=701, height=120, width=160,
    flow_source="dis", scene_seed=303,
)
config = Config()
result = refine(
    data["gray0"], data["gray1"], data["depth0"], data["depth1"],
    data["flow_observed"], data["intrinsics"], config,
)
np.savez_compressed(output / "data.npz", **data)
np.savez_compressed(
    output / "refined.npz",
    **{key: value for key, value in result.items() if key != "stats"},
)
(output / "metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8",
)
print("C++ inference seconds:", result["stats"]["elapsed_seconds"])
print("Observation-derived rigid components:", result["stats"]["rigid_accepted"])
```

上の推定器の引数に `gt_` 配列、正解可視性、正解物体番号は含まれません。
別々に動く物体が同じ深度領域につながっている場合は、`Config(rigid_prior=False)` の結果も比較してください。
この設定を切り替えても、画像照合・深度補正・移動上限・局所正則化は残ります。

## 固定条件の比較実験

```bash
uv run python scripts/report_v03.py --results ../data
uv run python scripts/verify_v03.py --case ../data/retest/gallery_dis_seed11 --method standard
```

保存済みの比較結果はリポジトリ直下の `../data/` にあります
（再試験6入力、新規条件6入力、19構成のアブレーション、8種類のストレス条件）。
`verify_v03.py` は保存済み入力で推定をやり直し、保存配列と突き合わせます。
全ての推定を直列・単一論理CPUへ固定して測定します。GPUや並列処理による高速化ではありません。
記録したC++時間には、描画・初期DISフローの生成・Pythonの評価・可視化を含めません。

新規条件では物体寸法、位置、姿勢、材料模様、運動を変更します。
基本の3配置からの変形なので、実データへの一般化を検証したものではありません。
新規入力の生成器は `src/scene_flow_refinement/primitives.py` にあります。
ソースを変更した再実験では、**必ず新しい `--out` を指定**してください。
結果版を展開した既存ディレクトリの再実行は、元の環境記録と混在させず別ディレクトリで行ってください。

## アブレーションの意味

標準版から一項ずつ外すほか、幾何Huberの有無と剛体補助項の有無を2×2で交差させます。
探索全列挙、融合なし、双対を毎回ゼロ初期化、長反復、短反復、一次TV、輝度利得補正、
移動上限なし、正則化の境界停止なし、往復照合なし、弱い剛体補助項、ステップ比4も比較します。
前版と同じ候補・反復条件にした「同値な計算核変更のみ」の経路もあります。

「境界停止なし」は正則化だけで、候補の深度段差検査と剛体領域の分割は残ります。
「上限なし」は候補の物理検査と出力変位球を両方外しますが、計算上の探索半径は残ります。
標準版での双対再利用は、この実験の全条件で精度を上げたわけではありません。
長く反復すれば正解誤差も必ず下がる、という関係もありません。

## 入出力契約と限界

灰度 `H×W` またはRGB `H×W×3` は有限の `[0,1]`、深度 `H×W` は光軸方向のメートル値です。
フロー `H×W×2` は時刻0から1への `(dx,dy)`、内部パラメータは `(fx,fy,cx,cy)` です。
整数座標を画素中心とします。深度のNaN・非正値は欠測です。カメラ運動の推定は含みません。

`scene_flow` は時刻間の3次元変位で、速度ではありません。`flow` は深度と変位の再投影、
`match_flow` は離散候補の対応です。柔らかい幾何項を使うため、両者には差が残ります。
`geometry_residual`、`support`、`rigid_weight`、`components` も保存します。
`support` は観測からの対応支持、`components` は観測深度の連結番号であり、正解の可視性や物体番号ではありません。

欠測をまたぐ結合を保守的に切るため、**深度の穴を完全に補完する方式ではありません。**
`depth0_supported` / `depth1_supported` が偽の数値には初期値が残り得ます。
このマスクは有効な元観測があることを示し、深度が正しいことを保証しません。

同じ深度で接する独立運動では、剛体補助項が動体精度を悪化させる反例が得られています。
無模様の球の回転は、この観測だけでは静止と区別できません。
移動上限違反がゼロでも、誤った小さな上限を与えれば真の運動を復元できません。
この実装は研究用CPU実装で、実時間動作やRAFTに対する優位性は主張しません。

## 理論文書と保存結果

```bash
uv run python scripts/report_v03.py --results ../data
cd ../docs
lualatex -interaction=nonstopmode -halt-on-error theory_ja.tex
lualatex -interaction=nonstopmode -halt-on-error theory_ja.tex
```

LuaLaTeX、日本語用のNoto CJKフォント、DejaVuフォントが必要です。フォントは同梱しません。
LaTeX本文、数値表、参照した図はリポジトリ直下の `../docs/` にあります。
保存結果 `../data/` には入力・正解・推定NPZ、設定・入力SHA-256・時間・領域別JSON、
CSV、数値検査を含めています。

旧評価関数で `max_estimated_motion_mm` にメートル値が保存されていた不備も修正し、
保存された変位からmmへ再集計しました。端点誤差と処理時間の値は変更していません。
修正履歴は結果版の `unit_corrections.json` です。

## 構成

```text
cpp/                       C++17：照合・剛体推定・最適化・解析的描画
src/scene_flow_refinement/  Python：API・データ生成・評価・可視化
scripts/                   集計と再計算の検証
tests/                     数式・核・境界・単位・生成器の試験
```

理論書（`../docs/`）と保存結果（`../data/`）はリポジトリ直下に置いています。

MIT License。外部ライブラリと参照論文には、それぞれの利用条件が適用されます。
