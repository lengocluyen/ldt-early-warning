from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ldt_pipeline import (
    Checkpoint,
    initialize_observation_graph,
    load_config,
    load_oulad,
    preprocess,
    split_presentations,
    update_observation_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally replay OULAD records across configured checkpoints.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--module", default="DDD")
    parser.add_argument("--partition", choices=["train", "test"], default="test")
    parser.add_argument("--output-dir", default="results/graphs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables = preprocess(load_oulad(cfg["data_dir"]))
    train, test = split_presentations(tables, args.module)
    presentations = train if args.partition == "train" else test
    checkpoints = [Checkpoint(**cp) for cp in cfg["checkpoints"]]
    namespace = cfg.get("namespace", "http://example.org/ldt#")

    graph, learner_keys, seen = initialize_observation_graph(tables, args.module, presentations, namespace)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cp in checkpoints:
        stats = update_observation_graph(graph, tables, args.module, presentations, cp.day, learner_keys, seen, namespace)
        path = output_dir / f"{args.module}_{args.partition}_day{cp.day}.ttl"
        graph.serialize(destination=str(path), format="turtle")
        rows.append({"checkpoint": cp.name, "graph": str(path), **stats})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
