from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ldt_pipeline import build_observation_graph, load_config, load_oulad, preprocess, split_presentations


def main() -> None:
    parser = argparse.ArgumentParser(description="Map OULAD records to an RDF observation graph.")
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
    graph, stats = build_observation_graph(tables, args.module, presentations, args.checkpoint, cfg["namespace"])
    output = Path(args.output or f"results/graphs/{args.module}_{args.partition}_day{args.checkpoint}.ttl")
    output.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(output), format="turtle")
    print(json.dumps({"graph": str(output), **stats}, indent=2))


if __name__ == "__main__":
    main()
