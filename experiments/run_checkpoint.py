from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ldt_pipeline import (
    build_observation_graph,
    compare_features,
    kg_features,
    load_config,
    load_oulad,
    materialize_alerts,
    model_specs,
    preprocess,
    raw_features,
    split_presentations,
    train_and_evaluate,
    validate_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one module/checkpoint experiment.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--module", default="DDD")
    parser.add_argument("--checkpoint", type=int, default=28)
    parser.add_argument("--feature-source", choices=["raw", "kg", "both"], default="both")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg.get("results_dir", "results"))
    namespace = cfg.get("namespace", "http://example.org/ldt#")
    tables = preprocess(load_oulad(cfg["data_dir"]))
    train_pres, test_pres = split_presentations(tables, args.module)

    train_raw = raw_features(tables, args.module, train_pres, args.checkpoint)
    test_raw = raw_features(tables, args.module, test_pres, args.checkpoint)
    train_graph, train_stats = build_observation_graph(tables, args.module, train_pres, args.checkpoint, namespace)
    test_graph, test_stats = build_observation_graph(tables, args.module, test_pres, args.checkpoint, namespace)
    train_kg = kg_features(train_graph, train_raw, args.checkpoint, namespace)
    test_kg = kg_features(test_graph, test_raw, args.checkpoint, namespace)

    features_dir = results_dir / "features"
    graphs_dir = results_dir / "graphs"
    tables_dir = results_dir / "tables"
    for directory in [features_dir, graphs_dir, tables_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_raw.to_csv(features_dir / f"{args.module}_train_raw_day{args.checkpoint}.csv", index=False)
    test_raw.to_csv(features_dir / f"{args.module}_test_raw_day{args.checkpoint}.csv", index=False)
    train_kg.to_csv(features_dir / f"{args.module}_train_kg_day{args.checkpoint}.csv", index=False)
    test_kg.to_csv(features_dir / f"{args.module}_test_kg_day{args.checkpoint}.csv", index=False)

    rows = []
    feature_frames = {
        "raw": (train_raw, test_raw),
        "kg": (train_kg, test_kg),
    }
    requested_sources = {"raw", "kg"} if args.feature_source == "both" else {args.feature_source}
    kg_xgb_metrics = None
    for spec in model_specs(cfg):
        method = spec["method"]
        source = spec["feature_source"]
        if source not in requested_sources:
            continue
        train_frame, test_frame = feature_frames[source]
        metrics = train_and_evaluate(train_frame, test_frame, method, cfg)
        rows.append({
            "module": args.module,
            "checkpoint_day": args.checkpoint,
            "method": method,
            "feature_source": source,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
        })
        if method == "XGB" and source == "kg":
            kg_xgb_metrics = metrics
    if "kg" in requested_sources and kg_xgb_metrics is not None:
        grounding = materialize_alerts(
            test_graph,
            test_kg,
            kg_xgb_metrics["model"],
            kg_xgb_metrics["prob"],
            args.checkpoint,
            namespace,
            float(cfg.get("alert_threshold", 0.5)),
            cfg.get("top_k", [3, 5]),
        )
        conforms, violations = validate_graph(test_graph)
        grounding.update({"shacl_conforms_after_analytics": conforms, "shacl_violations_after_analytics": violations})
        (tables_dir / f"grounding_{args.module}_day{args.checkpoint}.json").write_text(json.dumps(grounding, indent=2), encoding="utf-8")

    import pandas as pd

    pd.DataFrame(rows).to_csv(tables_dir / f"performance_{args.module}_day{args.checkpoint}.csv", index=False)
    test_graph.serialize(destination=str(graphs_dir / f"{args.module}_day{args.checkpoint}.ttl"), format="turtle")

    summary = {
        "module": args.module,
        "checkpoint_day": args.checkpoint,
        "train_graph": train_stats,
        "test_graph": test_stats,
        "feature_parity": compare_features(test_raw, test_kg),
        "performance_rows": len(rows),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
