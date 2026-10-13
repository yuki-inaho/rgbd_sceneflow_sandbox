"""Refine an existing NPZ of registered RGB-D frames and an external optical flow."""
from __future__ import annotations
import argparse
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
from .api import Config,refine


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--output",type=Path,default=Path("refined.npz"))
    parser.add_argument("--config",type=Path,help="JSON object with Config fields")
    args=parser.parse_args()
    settings=json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    unknown=set(settings)-{field.name for field in fields(Config)}
    if unknown:parser.error(f"unknown configuration keys: {sorted(unknown)}")
    config=Config(**settings)
    with np.load(args.input,allow_pickle=False) as data:
        required=("rgb0","rgb1","depth0","depth1","flow_observed","intrinsics")
        missing=set(required)-set(data.files)
        if missing:parser.error(f"missing arrays: {sorted(missing)}")
        result=refine(*(data[key] for key in required),config=config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**{key:value for key,value in result.items() if isinstance(value,np.ndarray)})
    args.output.with_suffix(".json").write_text(json.dumps(result["stats"],indent=2),encoding="utf-8")
    print(args.output)

if __name__=="__main__":main()
