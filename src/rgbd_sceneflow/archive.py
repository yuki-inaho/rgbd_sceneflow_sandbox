"""配布アーカイブ（data/）の読み込みと索引。

アーカイブ内のディレクトリ構造をそのままの意味で扱う。

    data/
      <split>/<case>/data.npz         入力（灰度・深度・観測フロー・正解）
      data/<split>/<case>/<method>.npz   手法ごとの推定出力
      data/<split>/<case>/<method>.json  手法ごとの凍結メトリクス
      data/<split>/<case>/metadata.json  生成条件（真値・入力ハッシュ）
      data/*.csv, *.json                 集計と監査記録
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: リポジトリ直下（`src/rgbd_sceneflow/archive.py` から2つ上）。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: 配布アーカイブを展開した場所。
DATA_ROOT = PROJECT_ROOT / "data"
#: 図の出力先。
FIGURES_DIR = PROJECT_ROOT / "figures"
#: 報告書の出力先。
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

#: アーカイブが持つ split 名。
SPLITS = ("ablation", "heldout", "retest", "stress")

#: 結果 json ではない補助ファイル。
AUXILIARY_JSON = {"metadata.json"}


@dataclass(frozen=True)
class Case:
    """1 つの入力ケース（split 名 + ケース名）。"""

    split: str
    name: str
    directory: Path

    @property
    def key(self) -> str:
        return f"{self.split}/{self.name}"

    def methods(self) -> list[str]:
        """このケースに保存されている手法名（`v02` は前版）。"""
        names = sorted(p.stem for p in self.directory.glob("*.json") if p.name not in AUXILIARY_JSON)
        return names

    def has(self, method: str) -> bool:
        return (self.directory / f"{method}.json").is_file()

    def inputs(self) -> dict[str, np.ndarray]:
        """`data.npz`（観測入力・正解・マスク）を辞書で返す。"""
        with np.load(self.directory / "data.npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def result(self, method: str) -> dict[str, np.ndarray]:
        """手法の推定出力（`*.npz`）を辞書で返す。"""
        with np.load(self.directory / f"{method}.npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def documented(self, method: str) -> dict:
        """手法の凍結 json（config / metrics / objects / 入力ハッシュ）を返す。"""
        return json.loads((self.directory / f"{method}.json").read_text(encoding="utf-8"))

    def metadata(self) -> dict:
        """生成条件（真値・可視率・入力 sha256）を返す。"""
        return json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))

    def shape(self) -> tuple[int, int]:
        meta = self.metadata()
        return int(meta["height"]), int(meta["width"])


def iter_cases(data_root: Path = DATA_ROOT) -> Iterator[Case]:
    """アーカイブ内の全ケースを安定した順序で列挙する。"""
    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        for directory in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            yield Case(split=split, name=directory.name, directory=directory)


def list_cases(data_root: Path = DATA_ROOT) -> list[Case]:
    return list(iter_cases(data_root))


def case_count(data_root: Path = DATA_ROOT) -> dict[str, int]:
    """split ごとのケース数を返す。"""
    counts: dict[str, int] = {}
    for case in iter_cases(data_root):
        counts[case.split] = counts.get(case.split, 0) + 1
    return counts


def result_count(data_root: Path = DATA_ROOT) -> int:
    """`<method>.json` の総数（アーカイブの「70組」に対応）。"""
    return sum(len(case.methods()) for case in iter_cases(data_root))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_input_hashes(data_root: Path = DATA_ROOT) -> dict[str, dict[str, str]]:
    """ケースごとに記録された入力配列の sha256 を返す。"""
    return {
        case.key: case.metadata()["input_sha256"]
        for case in iter_cases(data_root)
        if (case.directory / "metadata.json").is_file()
    }


def environment(data_root: Path = DATA_ROOT) -> dict:
    """実行環境の記録（Python 版・CPU・numpy 版）。"""
    return read_json(data_root / "environment.json")


def dependency_versions(data_root: Path = DATA_ROOT) -> dict:
    """当時インストールされていたパッケージ版。"""
    return read_json(data_root / "dependency_versions.json")


def sha256_array(array: np.ndarray) -> str:
    """配列の C 連続バイト列の sha256（入力ハッシュ照合に使う）。"""
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
