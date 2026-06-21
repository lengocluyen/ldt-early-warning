from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ldt_pipeline import train_and_evaluate, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one model on feature CSVs.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--method", choices=["LR", "RF", "ET", "HGB", "XGB", "LGBM", "CAT", "TABTX"], default="XGB")
    args = parser.parse_args()
    cfg = load_config(args.config)
    metrics = train_and_evaluate(pd.read_csv(args.train), pd.read_csv(args.test), args.method, cfg)
    clean = {k: metrics[k] for k in ["precision", "recall", "f1", "auc"]}
    print(json.dumps(clean, indent=2))


if __name__ == "__main__":
    main()
