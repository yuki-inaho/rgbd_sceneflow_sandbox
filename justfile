# 配布アーカイブから data/ と docs/ を復元する（~/Downloads の zip が入力）
extract:
    #!/usr/bin/env bash
    set -eu
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    unzip -q -o "$HOME/Downloads/scene_flow_v03_results.zip" -d "$tmp"
    unzip -q -o "$HOME/Downloads/scene_flow_v03_theory_sources.zip" -d "$tmp"
    rm -rf data/v03 && mkdir -p data/v03
    cp -a "$tmp/scene_flow_primitives/results/v03/." data/v03/
    cp -a "$tmp/scene_flow_v03_theory/theory_ja.tex" docs/theory_ja.tex
    cp -a "$tmp/scene_flow_v03_theory/numbers.tex" docs/numbers.tex
    cp -a "$tmp"/scene_flow_v03_theory/table_*.tex docs/
    mkdir -p docs/figures && cp -a "$tmp/scene_flow_v03_theory/figures/." docs/figures/
    cp -a "$tmp/scene_flow_v03_theory/CHANGELOG_ja.md" docs/CHANGELOG_ja.md
    echo "restored: data/v03 ($(find data/v03 -name '*.npz' | wc -l) npz) and docs/"

sync:
    uv sync

# 配布実装をビルドする（C++17 の共有ライブラリ）
build:
    #!/usr/bin/env bash
    set -eu
    cd scene_flow_primitives
    uv sync
    uv pip install setuptools wheel
    uv pip install --no-build-isolation --no-deps -e .
    ls -l src/scene_flow_refinement/_native*.so

# 配布実装の自動試験（93 件）
upstream-test:
    cd scene_flow_primitives && uv run python -m pytest -q

# 保存済み推定 1 件を再計算し、配列が完全一致することを確認する
verify-upstream:
    cd scene_flow_primitives && uv run python scripts/verify_v03.py \
        --case ../data/v03/retest/gallery_dis_seed11 --method standard \
        --out ../outputs/upstream_verify_retest_gallery_dis_seed11.json

test:
    uv run pytest

# 70 組の保存結果を再計算し、凍結メトリクスと突合した報告書を出力する
reproduce:
    uv run python -m rgbd_sceneflow.reproduce

figures:
    uv run python -m rgbd_sceneflow.figures

tables:
    uv run python -m rgbd_sceneflow.tables

notebook:
    uv run jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 notebooks/01_scene_flow_v03_reproduction.ipynb

docs:
    cd docs && lualatex -interaction=nonstopmode -halt-on-error theory_ja.tex
    cd docs && lualatex -interaction=nonstopmode -halt-on-error theory_ja.tex

lint:
    uv run ruff check .
    uv run ruff format --check .
