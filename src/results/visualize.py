from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHECKPOINT_ORDER = {
    "Week 4": 4,
    "Week 8": 8,
    "Week 12": 12,
}

PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#ff7f0e",
]


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return frame if not frame.empty else None


def _checkpoint_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    return CHECKPOINT_ORDER.get(text, 10_000), text


def _model_label(frame: pd.DataFrame) -> pd.Series:
    return frame["method"].astype(str) + "/" + frame["feature_source"].astype(str)


def _finish(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _line_metric(frame: pd.DataFrame, metric: str, output: Path) -> str | None:
    required = {"checkpoint", "method", "feature_source", metric}
    if not required <= set(frame.columns):
        return None
    data = frame.copy()
    data["model"] = _model_label(data)
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    grouped = (
        data.groupby(["checkpoint", "model"], dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    checkpoints = sorted(grouped["checkpoint"].dropna().unique(), key=_checkpoint_sort_key)
    models = sorted(grouped["model"].dropna().unique())
    if not checkpoints or not models:
        return None

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(checkpoints))
    for idx, model in enumerate(models):
        sub = grouped[grouped["model"] == model].set_index("checkpoint").reindex(checkpoints)
        y = sub["mean"].to_numpy(dtype=float)
        yerr = sub["std"].to_numpy(dtype=float)
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(x, y, marker="o", linewidth=2, color=color, label=model)
        if np.isfinite(yerr).any() and (sub["count"].fillna(0) > 1).any():
            lower = y - np.nan_to_num(yerr)
            upper = y + np.nan_to_num(yerr)
            ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)

    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Checkpoint")
    ax.set_title(f"Predictive {metric.upper()} by checkpoint")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return _finish(fig, output / f"predictive_{metric}_by_checkpoint.png")


def _line_auc_compact(frame: pd.DataFrame, output: Path) -> str | None:
    required = {"checkpoint", "method", "feature_source", "auc"}
    if not required <= set(frame.columns):
        return None

    data = frame.copy()
    data["auc"] = pd.to_numeric(data["auc"], errors="coerce")
    data["method"] = data["method"].astype(str)
    data["feature_source"] = data["feature_source"].astype(str)
    data = data[data["feature_source"] == "kg"].copy()
    if data.empty:
        return None

    data["model"] = data["method"] + " KG"
    grouped = (
        data.groupby(["checkpoint", "model"], dropna=False)["auc"]
        .mean()
        .reset_index()
    )
    checkpoints = sorted(grouped["checkpoint"].dropna().unique(), key=_checkpoint_sort_key)
    model_order = [
        "XGB KG",
        "RF KG",
        "ET KG",
        "HGB KG",
        "LGBM KG",
        "CAT KG",
        "TABTX KG",
    ]
    models = [model for model in model_order if model in set(grouped["model"])]
    if not checkpoints or not models:
        return None

    colors = {
        "XGB KG": "#0072B2",
        "RF KG": "#009E73",
        "ET KG": "#D55E00",
        "HGB KG": "#CC79A7",
        "LGBM KG": "#E69F00",
        "CAT KG": "#56B4E9",
        "TABTX KG": "#882255",
    }
    markers = {
        "XGB KG": "s",
        "RF KG": "^",
        "ET KG": "D",
        "HGB KG": "v",
        "LGBM KG": "P",
        "CAT KG": "X",
        "TABTX KG": "*",
    }
    line_styles = {
        "XGB KG": "-",
        "RF KG": "--",
        "ET KG": "-.",
        "HGB KG": ":",
        "LGBM KG": (0, (5, 1, 1, 1)),
        "CAT KG": (0, (3, 1, 1, 1, 1, 1)),
        "TABTX KG": (0, (7, 2)),
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.25))
    x = np.arange(len(checkpoints))
    y_values: list[float] = []
    endpoints: list[tuple[str, float, float, str]] = []
    for model in models:
        sub = grouped[grouped["model"] == model].set_index("checkpoint").reindex(checkpoints)
        y = sub["auc"].to_numpy(dtype=float)
        y_values.extend([float(value) for value in y if np.isfinite(value)])
        finite_idx = np.flatnonzero(np.isfinite(y))
        if len(finite_idx):
            last_idx = int(finite_idx[-1])
            endpoints.append((model, float(x[last_idx]), float(y[last_idx]), colors.get(model, "#333333")))
        ax.plot(
            x,
            y,
            marker=markers.get(model, "o"),
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.2,
            linestyle=line_styles.get(model, "-"),
            color=colors.get(model, "#333333"),
            label=model,
        )

    if y_values:
        ymin = max(0.0, np.floor((min(y_values) - 0.025) * 20) / 20)
        ymax = min(1.0, np.ceil((max(y_values) + 0.025) * 20) / 20)
        if ymax - ymin < 0.15:
            ymax = min(1.0, ymin + 0.15)
        ax.set_ylim(ymin, ymax)

    if endpoints:
        ymin, ymax = ax.get_ylim()
        min_gap = (ymax - ymin) * 0.045
        label_items = sorted(endpoints, key=lambda item: item[2])
        adjusted: list[tuple[str, float, float, float, str]] = []
        previous_y = ymin
        for model, x_end, y_end, color in label_items:
            label_y = max(y_end, previous_y + min_gap)
            adjusted.append((model, x_end, y_end, label_y, color))
            previous_y = label_y
        overflow = adjusted[-1][3] - (ymax - min_gap) if adjusted else 0
        if overflow > 0:
            adjusted = [
                (model, x_end, y_end, label_y - overflow, color)
                for model, x_end, y_end, label_y, color in adjusted
            ]
        label_x = float(x[-1]) + 0.10
        for model, x_end, y_end, label_y, color in adjusted:
            ax.plot([x_end + 0.03, label_x - 0.02], [y_end, label_y], color=color, linewidth=0.8, alpha=0.55)
            ax.text(
                label_x,
                label_y,
                f"{model.replace(' KG', '')} {y_end:.3f}",
                color=color,
                fontsize=8,
                va="center",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints)
    ax.set_xlim(-0.1, float(x[-1]) + 0.62)
    ax.set_ylabel("Mean AUC across modules")
    ax.set_xlabel("Checkpoint")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _finish(fig, output / "predictive_auc_by_checkpoint.png")


def _plot_predictive(results: Path, output: Path) -> list[str]:
    frame = _read_csv(results / "tables" / "predictive_performance.csv")
    if frame is None:
        return []
    generated = []
    for metric in ["auc", "f1", "precision", "recall"]:
        path = _line_auc_compact(frame, output) if metric == "auc" else _line_metric(frame, metric, output)
        if path:
            generated.append(path)
    return generated


def _plot_grounding(results: Path, output: Path) -> list[str]:
    frame = _read_csv(results / "tables" / "explanation_grounding.csv")
    if frame is None or "checkpoint" not in frame.columns:
        return []
    metric_labels = {
        "coverage_direct_3": "Direct Cov@3",
        "coverage_direct_5": "Direct Cov@5",
        "fidelity_3": "Presented Fid@3",
    }
    metric_cols = [col for col in metric_labels if col in frame.columns]
    if not metric_cols:
        return []
    data = frame.copy()
    for col in metric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    grouped_mean = data.groupby("checkpoint", dropna=False)[metric_cols].mean().reset_index()
    grouped_std = data.groupby("checkpoint", dropna=False)[metric_cols].std().reset_index()
    checkpoints = sorted(grouped_mean["checkpoint"].dropna().unique(), key=_checkpoint_sort_key)
    grouped = grouped_mean.set_index("checkpoint").reindex(checkpoints)
    grouped_err = grouped_std.set_index("checkpoint").reindex(checkpoints)

    fig, ax = plt.subplots(figsize=(7.8, 4.25))
    x = np.arange(len(checkpoints))
    styles = {
        "coverage_direct_3": ("#0072B2", "o", "-"),
        "coverage_direct_5": ("#D55E00", "s", "--"),
        "fidelity_3": ("#009E73", "D", "-."),
    }
    x_offsets = {
        "coverage_direct_3": -0.12,
        "coverage_direct_5": 0.00,
        "fidelity_3": 0.12,
    }
    endpoints: list[tuple[str, float, float, str]] = []
    y_values: list[float] = []
    y_low_values: list[float] = []
    for col in metric_cols:
        label = metric_labels[col]
        color, marker, linestyle = styles.get(col, ("#333333", "o", "-"))
        x_metric = x + x_offsets.get(col, 0.0)
        y = grouped[col].to_numpy(dtype=float)
        yerr = grouped_err[col].to_numpy(dtype=float) if col in grouped_err else None
        y_values.extend([float(value) for value in y if np.isfinite(value)])
        if yerr is not None:
            y_low_values.extend(
                [
                    float(value - err)
                    for value, err in zip(y, yerr)
                    if np.isfinite(value) and np.isfinite(err)
                ]
            )
        finite_idx = np.flatnonzero(np.isfinite(y))
        if len(finite_idx):
            last_idx = int(finite_idx[-1])
            endpoints.append((label, float(x_metric[last_idx]), float(y[last_idx]), color))
        ax.errorbar(
            x_metric,
            y,
            yerr=yerr,
            marker=marker,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.2,
            linestyle=linestyle,
            color=color,
            capsize=3,
            label=label,
            zorder=3,
        )

    if y_values:
        ymin = max(0.0, (min(y_low_values) if y_low_values else min(y_values)) - 0.03)
        ax.set_ylim(ymin, 1.035)
    ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1.2, alpha=0.75)
    ax.text(
        x[0] - 0.05,
        1.006,
        "Contextual coverage and traceability = 1.000",
        color="#555555",
        fontsize=8,
        va="bottom",
    )

    if endpoints:
        label_x = max(item[1] for item in endpoints) + 0.10
        for idx, (label, x_end, y_end, color) in enumerate(sorted(endpoints, key=lambda item: item[2], reverse=True)):
            label_y = y_end - idx * 0.008
            ax.plot([x_end + 0.03, label_x - 0.02], [y_end, label_y], color=color, linewidth=0.8, alpha=0.6)
            ax.text(label_x, label_y, f"{label} {y_end:.3f}", color=color, fontsize=8, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints)
    left_limit = min(float(np.min(x + offset)) for offset in x_offsets.values()) - 0.08
    right_limit = max(float(np.max(x + offset)) for offset in x_offsets.values()) + 0.76
    ax.set_xlim(left_limit, right_limit)
    ax.set_ylabel("Mean across modules")
    ax.set_xlabel("Checkpoint")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return [_finish(fig, output / "evidence_grounding_metrics.png")]


def _plot_construction(results: Path, output: Path) -> list[str]:
    frame = _read_csv(results / "tables" / "incremental_construction.csv")
    if frame is None or "checkpoint" not in frame.columns:
        return []
    generated = []
    data = frame.copy()
    checkpoints = sorted(data["checkpoint"].dropna().unique(), key=_checkpoint_sort_key)

    if "update_time" in data.columns:
        data["update_time"] = pd.to_numeric(data["update_time"], errors="coerce")
        grouped = data.groupby("checkpoint", dropna=False)["update_time"].mean().reindex(checkpoints)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(np.arange(len(checkpoints)), grouped.to_numpy(dtype=float), color="#1f77b4")
        ax.set_xticks(np.arange(len(checkpoints)))
        ax.set_xticklabels(checkpoints)
        ax.set_ylabel("Seconds")
        ax.set_xlabel("Checkpoint")
        ax.set_title("Mean graph update time")
        ax.grid(axis="y", alpha=0.25)
        generated.append(_finish(fig, output / "construction_update_time.png"))

    cumulative_cols = [
        col
        for col in ["cumulative_traces", "cumulative_assessment_results", "cumulative_triples"]
        if col in data.columns
    ]
    if cumulative_cols:
        for col in cumulative_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        grouped = data.groupby("checkpoint", dropna=False)[cumulative_cols].mean().reindex(checkpoints)
        fig, axes = plt.subplots(len(cumulative_cols), 1, figsize=(8.5, 3.0 * len(cumulative_cols)), sharex=True)
        axes_list = np.atleast_1d(axes)
        x = np.arange(len(checkpoints))
        for idx, col in enumerate(cumulative_cols):
            axes_list[idx].plot(x, grouped[col].to_numpy(dtype=float), marker="o", linewidth=2, color=PALETTE[idx])
            axes_list[idx].set_ylabel(col.replace("_", " "))
            axes_list[idx].grid(axis="y", alpha=0.25)
        axes_list[-1].set_xticks(x)
        axes_list[-1].set_xticklabels(checkpoints)
        axes_list[0].set_title("Cumulative LDT graph growth")
        generated.append(_finish(fig, output / "construction_cumulative_growth.png"))

    violation_cols = [col for col in ["shacl_violations"] if col in data.columns]
    grounding = _read_csv(results / "tables" / "explanation_grounding.csv")
    if grounding is not None and "shacl_violations_after_analytics" in grounding.columns:
        tmp = grounding[["checkpoint", "shacl_violations_after_analytics"]].copy()
        tmp["source"] = "analytics"
        tmp = tmp.rename(columns={"shacl_violations_after_analytics": "violations"})
        obs = None
        if violation_cols:
            obs = data[["checkpoint", "shacl_violations"]].copy()
            obs["source"] = "observation"
            obs = obs.rename(columns={"shacl_violations": "violations"})
        shacl_frame = pd.concat([obs, tmp], ignore_index=True) if obs is not None else tmp
    elif violation_cols:
        shacl_frame = data[["checkpoint", "shacl_violations"]].copy()
        shacl_frame["source"] = "observation"
        shacl_frame = shacl_frame.rename(columns={"shacl_violations": "violations"})
    else:
        shacl_frame = None
    if shacl_frame is not None and not shacl_frame.empty:
        shacl_frame["violations"] = pd.to_numeric(shacl_frame["violations"], errors="coerce").fillna(0)
        grouped = shacl_frame.groupby(["checkpoint", "source"], dropna=False)["violations"].mean().reset_index()
        sources = sorted(grouped["source"].unique())
        x = np.arange(len(checkpoints))
        width = 0.75 / max(len(sources), 1)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for idx, source in enumerate(sources):
            sub = grouped[grouped["source"] == source].set_index("checkpoint").reindex(checkpoints)
            ax.bar(x + (idx - (len(sources) - 1) / 2) * width, sub["violations"].fillna(0), width, label=source)
        ax.set_xticks(x)
        ax.set_xticklabels(checkpoints)
        ax.set_ylabel("Mean SHACL violations")
        ax.set_xlabel("Checkpoint")
        ax.set_title("SHACL validation results")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        generated.append(_finish(fig, output / "shacl_violations.png"))

    return generated


def _plot_feature_parity(results: Path, output: Path) -> list[str]:
    frame = _read_csv(results / "tables" / "feature_parity.csv")
    if frame is None or "checkpoint" not in frame.columns:
        return []
    data = frame.copy()
    feature_cols = []
    for col in data.columns:
        if col in {"module", "checkpoint"}:
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].notna().any():
            feature_cols.append(col)
    if not feature_cols:
        return []
    for col in feature_cols:
        data[col] = data[col].abs()
    checkpoints = sorted(data["checkpoint"].dropna().unique(), key=_checkpoint_sort_key)
    heat = data.groupby("checkpoint", dropna=False)[feature_cols].max().reindex(checkpoints).T

    fig, ax = plt.subplots(figsize=(8.0, max(4.5, 0.4 * len(feature_cols))))
    values = heat.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(checkpoints)))
    ax.set_xticklabels(checkpoints)
    ax.set_yticks(np.arange(len(feature_cols)))
    ax.set_yticklabels(feature_cols)
    ax.set_title("Raw/KG feature parity max absolute difference")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    return [_finish(fig, output / "feature_parity_heatmap.png")]


def _plot_experimental_process(results: Path, output: Path) -> list[str]:
    frame = _read_csv(results / "tables" / "experimental_process.csv")
    if frame is None or "checkpoint" not in frame.columns:
        return []
    time_cols = [
        col
        for col in [
            "raw_feature_time",
            "graph_update_time",
            "kg_feature_time",
            "validation_time",
            "model_train_eval_time",
            "evidence_materialization_time",
            "analytics_validation_time",
        ]
        if col in frame.columns
    ]
    if not time_cols:
        return []
    data = frame.copy()
    for col in time_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    checkpoints = sorted(data["checkpoint"].dropna().unique(), key=_checkpoint_sort_key)
    grouped = data.groupby("checkpoint", dropna=False)[time_cols].mean().reindex(checkpoints)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(checkpoints))
    bottom = np.zeros(len(checkpoints))
    for idx, col in enumerate(time_cols):
        values = grouped[col].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=PALETTE[idx % len(PALETTE)], label=col.replace("_", " "))
        bottom += np.nan_to_num(values)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints)
    ax.set_ylabel("Mean seconds")
    ax.set_xlabel("Checkpoint")
    ax.set_title("Experimental process time by stage")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return [_finish(fig, output / "experimental_process_timing.png")]


def _plot_model_timing(results: Path, output: Path) -> list[str]:
    frame = _read_csv(results / "tables" / "model_timing.csv")
    if frame is None or not {"checkpoint", "method", "feature_source", "train_eval_time"} <= set(frame.columns):
        return []
    data = frame.copy()
    data["model"] = _model_label(data)
    data["train_eval_time"] = pd.to_numeric(data["train_eval_time"], errors="coerce")
    grouped = data.groupby(["checkpoint", "model"], dropna=False)["train_eval_time"].mean().reset_index()
    checkpoints = sorted(grouped["checkpoint"].dropna().unique(), key=_checkpoint_sort_key)
    models = sorted(grouped["model"].dropna().unique())
    if not checkpoints or not models:
        return []

    x = np.arange(len(checkpoints))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for idx, model in enumerate(models):
        sub = grouped[grouped["model"] == model].set_index("checkpoint").reindex(checkpoints)
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(x + offset, sub["train_eval_time"].fillna(0), width, label=model, color=PALETTE[idx % len(PALETTE)])
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints)
    ax.set_ylabel("Mean seconds")
    ax.set_xlabel("Checkpoint")
    ax.set_title("Model training and evaluation time")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return [_finish(fig, output / "model_training_time.png")]


def _plot_training_history(results: Path, output: Path) -> list[str]:
    frame = _read_csv(results / "tables" / "training_history.csv")
    if frame is None or not {"epoch", "train_loss", "method", "feature_source"} <= set(frame.columns):
        return []
    data = frame.copy()
    data["epoch"] = pd.to_numeric(data["epoch"], errors="coerce")
    data["train_loss"] = pd.to_numeric(data["train_loss"], errors="coerce")
    data["val_loss"] = pd.to_numeric(data["val_loss"], errors="coerce") if "val_loss" in data.columns else np.nan
    data["series"] = _model_label(data)
    if "checkpoint" in data.columns:
        data["series"] = data["checkpoint"].astype(str) + " " + data["series"]
    grouped = data.groupby(["series", "epoch"], dropna=False)[["train_loss", "val_loss"]].mean().reset_index()
    series_values = sorted(grouped["series"].dropna().unique())
    if not series_values:
        return []

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for idx, series in enumerate(series_values):
        sub = grouped[grouped["series"] == series].sort_values("epoch")
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(sub["epoch"], sub["train_loss"], linewidth=2, color=color, label=f"{series} train")
        if sub["val_loss"].notna().any():
            ax.plot(sub["epoch"], sub["val_loss"], linewidth=1.8, linestyle="--", color=color, label=f"{series} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("TabTransformer training history")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return [_finish(fig, output / "tabtransformer_training_history.png")]


def _plot_dataset(results: Path, output: Path) -> list[str]:
    frame = _read_csv(results / "tables" / "dataset_distribution.csv")
    if frame is None or "module" not in frame.columns:
        return []
    data = frame.copy().sort_values("module")
    generated = []
    x = np.arange(len(data))

    learner_cols = [col for col in ["train_learners", "test_learners"] if col in data.columns]
    if learner_cols:
        fig, ax = plt.subplots(figsize=(max(8.5, 0.6 * len(data)), 4.8))
        width = 0.75 / len(learner_cols)
        for idx, col in enumerate(learner_cols):
            ax.bar(x + (idx - (len(learner_cols) - 1) / 2) * width, pd.to_numeric(data[col], errors="coerce"), width, label=col)
        ax.set_xticks(x)
        ax.set_xticklabels(data["module"], rotation=30, ha="right")
        ax.set_ylabel("Learners")
        ax.set_title("Dataset distribution by module")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        generated.append(_finish(fig, output / "dataset_learners_by_module.png"))

    risk_cols = [col for col in ["at_risk_pct_train", "at_risk_pct_test"] if col in data.columns]
    if risk_cols:
        fig, ax = plt.subplots(figsize=(max(8.5, 0.6 * len(data)), 4.8))
        for idx, col in enumerate(risk_cols):
            ax.plot(x, pd.to_numeric(data[col], errors="coerce"), marker="o", linewidth=2, label=col, color=PALETTE[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(data["module"], rotation=30, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("At-risk learners (%)")
        ax.set_title("At-risk rate by module")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        generated.append(_finish(fig, output / "dataset_at_risk_by_module.png"))

    return generated


def visualize_result_tables(results: str | Path = "results", output: str | Path | None = None) -> dict[str, Any]:
    results = Path(results)
    output = Path(output or results / "figures")
    output.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    generated.extend(_plot_dataset(results, output))
    generated.extend(_plot_predictive(results, output))
    generated.extend(_plot_grounding(results, output))
    generated.extend(_plot_construction(results, output))
    generated.extend(_plot_feature_parity(results, output))
    generated.extend(_plot_experimental_process(results, output))
    generated.extend(_plot_model_timing(results, output))
    generated.extend(_plot_training_history(results, output))

    manifest = {
        "generated": generated,
        "output_dir": str(output),
    }
    manifest_path = output / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    generated.append(str(manifest_path))
    manifest["generated"] = generated
    return manifest
