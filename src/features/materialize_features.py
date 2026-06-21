from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ldt_pipeline import build_observation_graph, kg_features, load_config, load_oulad, preprocess, raw_features, split_presentations


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize PredictionFeature nodes for one checkpoint.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--module", default="DDD")
    parser.add_argument("--checkpoint", type=int, default=28)
    parser.add_argument("--partition", choices=["train", "test"], default="test")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tables = preprocess(load_oulad(cfg["data_dir"]))
    train, test = split_presentations(tables, args.module)
    presentations = train if args.partition == "train" else test
    raw = raw_features(tables, args.module, presentations, args.checkpoint)
    graph, _ = build_observation_graph(tables, args.module, presentations, args.checkpoint, cfg["namespace"])
    frame = kg_features(graph, raw, args.checkpoint, cfg["namespace"])
    # Full analytic feature materialization is performed during alert materialization in run_checkpoint/run_all.
    output = Path(args.output or f"results/features/{args.module}_{args.partition}_kg_day{args.checkpoint}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(json.dumps({"features": str(output), "rows": len(frame)}, indent=2))


if __name__ == "__main__":
    main()
