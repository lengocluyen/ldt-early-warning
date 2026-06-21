from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ldt_pipeline import load_config, load_oulad, preprocess, raw_features, split_presentations, train_and_evaluate, shap_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SHAP values for one XGBoost checkpoint model.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--module", default="DDD")
    parser.add_argument("--checkpoint", type=int, default=28)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tables = preprocess(load_oulad(cfg["data_dir"]))
    train_pres, test_pres = split_presentations(tables, args.module)
    train = raw_features(tables, args.module, train_pres, args.checkpoint)
    test = raw_features(tables, args.module, test_pres, args.checkpoint)
    metrics = train_and_evaluate(train, test, "XGB", cfg)
    values = shap_values(metrics["model"], test)
    print(json.dumps({"rows": int(values.shape[0]), "features": int(values.shape[1])}, indent=2))


if __name__ == "__main__":
    main()
