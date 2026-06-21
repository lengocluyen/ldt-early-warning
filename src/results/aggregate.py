from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def mean_std(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)[value_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    count = frame.groupby(group_cols, dropna=False).size().rename("n")
    return pd.concat([count, mean, std], axis=1).reset_index()


def _all_true(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns or frame.empty:
        return False
    values = frame[column].astype(str).str.lower()
    return bool(values.isin(["true", "1"]).all())


def _publication_readiness(results: Path) -> dict[str, Any]:
    tables = results / "tables"
    checks: list[dict[str, Any]] = []

    dataset = tables / "dataset_distribution.csv"
    if dataset.exists():
        frame = pd.read_csv(dataset)
        n_modules = int(frame["module"].nunique()) if "module" in frame.columns else 0
        checks.append({
            "check": "multiple_modules",
            "passed": n_modules > 1,
            "detail": f"{n_modules} module(s) represented",
        })

    performance_agg = tables / "predictive_performance_aggregated.csv"
    if performance_agg.exists():
        frame = pd.read_csv(performance_agg)
        min_n = int(frame["n"].min()) if "n" in frame.columns and not frame.empty else 0
        checks.append({
            "check": "aggregated_performance_has_replicates",
            "passed": min_n > 1,
            "detail": f"minimum aggregated n={min_n}",
        })

    construction = tables / "incremental_construction.csv"
    if construction.exists():
        frame = pd.read_csv(construction)
        checks.append({
            "check": "observation_graph_shacl_conforms",
            "passed": _all_true(frame, "shacl_conforms"),
            "detail": "all observation/checkpoint rows must conform",
        })

    grounding = tables / "explanation_grounding.csv"
    if grounding.exists():
        frame = pd.read_csv(grounding)
        direct_cols = [col for col in frame.columns if col.startswith("coverage_direct_")]
        checks.append({
            "check": "analytics_graph_shacl_conforms",
            "passed": _all_true(frame, "shacl_conforms_after_analytics"),
            "detail": "all analytic write-back rows must conform",
        })
        checks.append({
            "check": "direct_evidence_coverage_reported",
            "passed": bool(direct_cols),
            "detail": ", ".join(direct_cols) if direct_cols else "no direct coverage columns found",
        })

    parity = tables / "feature_parity.csv"
    if parity.exists():
        frame = pd.read_csv(parity)
        numeric = frame.select_dtypes(include="number")
        max_diff = float(numeric.max().max()) if not numeric.empty else float("nan")
        checks.append({
            "check": "raw_kg_feature_parity",
            "passed": bool(pd.notna(max_diff) and max_diff <= 1e-9),
            "detail": f"maximum absolute difference={max_diff}",
        })

    ready = bool(checks) and all(check["passed"] for check in checks)
    return {"ready_for_publication_tables": ready, "checks": checks}


def aggregate_result_tables(results: str | Path = "results", output: str | Path | None = None) -> dict[str, Any]:
    results = Path(results)
    output = Path(output or results / "tables")
    output.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    performance = results / "tables" / "predictive_performance.csv"
    if performance.exists():
        frame = pd.read_csv(performance)
        agg = mean_std(frame, ["checkpoint", "method", "feature_source"], ["precision", "recall", "f1", "auc"])
        path = output / "predictive_performance_aggregated.csv"
        agg.to_csv(path, index=False)
        generated.append(str(path))

    grounding = results / "tables" / "explanation_grounding.csv"
    if grounding.exists():
        frame = pd.read_csv(grounding)
        coverage_cols = [col for col in frame.columns if col.startswith("coverage_")]
        value_cols = coverage_cols + ["fidelity_3", "traceability", "avg_evidence_per_alert"]
        value_cols = [col for col in value_cols if col in frame.columns]
        agg = mean_std(frame, ["checkpoint"], value_cols)
        path = output / "explanation_grounding_aggregated.csv"
        agg.to_csv(path, index=False)
        generated.append(str(path))

    construction = results / "tables" / "incremental_construction.csv"
    if construction.exists():
        frame = pd.read_csv(construction)
        value_cols = [
            "learners",
            "new_traces",
            "new_assessment_results",
            "new_triples",
            "cumulative_traces",
            "cumulative_assessment_results",
            "cumulative_triples",
            "traces",
            "assessment_results",
            "triples",
            "update_time",
            "shacl_violations",
        ]
        value_cols = [col for col in value_cols if col in frame.columns]
        agg = mean_std(frame, ["checkpoint"], value_cols)
        path = output / "incremental_construction_aggregated.csv"
        agg.to_csv(path, index=False)
        generated.append(str(path))

    readiness = _publication_readiness(results)
    readiness_path = output / "publication_readiness.json"
    readiness_path.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    generated.append(str(readiness_path))

    return {"generated": generated, "publication_readiness": readiness}
