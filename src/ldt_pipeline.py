from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdflib import Graph, Literal, Namespace, RDF, URIRef, Variable, XSD
from rdflib.plugins.sparql import prepareQuery
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.models.tab_transformer import TabTransformerClassifier
from src.results.aggregate import aggregate_result_tables
from src.evidence.policy import should_present_evidence

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional until dependencies are installed
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional until dependencies are installed
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional until dependencies are installed
    CatBoostClassifier = None

try:
    import shap
except Exception:  # pragma: no cover - optional until dependencies are installed
    shap = None

try:
    from pyshacl import validate as shacl_validate
except Exception:  # pragma: no cover - optional until dependencies are installed
    shacl_validate = None


FEATURES = [
    "total_clicks",
    "active_days",
    "distinct_resources",
    "activity_trend",
    "days_since_last",
    "n_submitted",
    "avg_score",
    "missing_assessment",
    "late_submission",
]

AT_RISK = {"Fail", "Withdrawn"}

DEFAULT_MODEL_SPECS = [
    {"method": "LR", "feature_source": "raw"},
    {"method": "XGB", "feature_source": "raw"},
    {"method": "XGB", "feature_source": "kg"},
]

EXTENDED_MODEL_SPECS = [
    *DEFAULT_MODEL_SPECS,
    {"method": "RF", "feature_source": "raw"},
    {"method": "RF", "feature_source": "kg"},
    {"method": "ET", "feature_source": "raw"},
    {"method": "ET", "feature_source": "kg"},
    {"method": "HGB", "feature_source": "raw"},
    {"method": "HGB", "feature_source": "kg"},
    {"method": "LGBM", "feature_source": "raw"},
    {"method": "LGBM", "feature_source": "kg"},
    {"method": "CAT", "feature_source": "raw"},
    {"method": "CAT", "feature_source": "kg"},
    {"method": "TABTX", "feature_source": "raw"},
    {"method": "TABTX", "feature_source": "kg"},
]


@dataclass(frozen=True)
class Checkpoint:
    day: int
    name: str


@dataclass
class ModelEvaluation:
    precision: float
    recall: float
    f1: float
    auc: float
    model: Any
    prob: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "model": self.model,
            "prob": self.prob,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def items(self):
        return self.as_dict().items()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=None)
def prepare_feature_query(name: str) -> Any:
    query_dir = Path(__file__).resolve().parents[1] / "sparql" / "features"
    return prepareQuery((query_dir / f"{name}.rq").read_text(encoding="utf-8"))


def load_oulad(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    base = Path(data_dir)
    tables = {}
    for name in [
        "studentInfo",
        "courses",
        "vle",
        "studentVle",
        "assessments",
        "studentAssessment",
        "studentRegistration",
    ]:
        tables[name] = pd.read_csv(base / f"{name}.csv")
    return tables


def preprocess(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {k: v.copy() for k, v in tables.items()}
    for table in out.values():
        for col in ["id_student", "id_site", "id_assessment"]:
            if col in table.columns:
                table[col] = pd.to_numeric(table[col], errors="coerce").astype("Int64")
    for col in ["date", "sum_click", "date_submitted", "score", "date_registration", "date_unregistration"]:
        for table in out.values():
            if col in table.columns:
                table[col] = pd.to_numeric(table[col], errors="coerce")

    reg = out["studentRegistration"]
    excluded = reg.loc[reg["date_unregistration"].notna() & (reg["date_unregistration"] <= 0),
                       ["code_module", "code_presentation", "id_student"]]
    info = out["studentInfo"].merge(
        excluded.assign(_exclude=True),
        how="left",
        on=["code_module", "code_presentation", "id_student"],
    )
    out["studentInfo"] = info[info["_exclude"].isna()].drop(columns=["_exclude"])
    return out


def eligible_modules(tables: dict[str, pd.DataFrame]) -> list[str]:
    info = tables["studentInfo"].copy()
    info["year"] = info["code_presentation"].astype(str).str[:4]
    years = info.groupby("code_module")["year"].agg(set)
    return sorted([module for module, ys in years.items() if {"2013", "2014"} <= ys])


def split_presentations(tables: dict[str, pd.DataFrame], module: str) -> tuple[list[str], list[str]]:
    pres = sorted(tables["studentInfo"].loc[tables["studentInfo"]["code_module"] == module, "code_presentation"].unique())
    train = [p for p in pres if str(p).startswith("2013")]
    test = [p for p in pres if str(p).startswith("2014")]
    return train, test


def cohort(tables: dict[str, pd.DataFrame], module: str, presentations: list[str]) -> pd.DataFrame:
    info = tables["studentInfo"]
    df = info[(info["code_module"] == module) & (info["code_presentation"].isin(presentations))].copy()
    df["learner_key"] = make_learner_key(df)
    df["label"] = df["final_result"].isin(AT_RISK).astype(int)
    return df


def make_learner_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["id_student"].astype(str)
        + "_"
        + df["code_module"].astype(str)
        + "_"
        + df["code_presentation"].astype(str)
    )


def _assessment_join(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return tables["studentAssessment"].merge(
        tables["assessments"],
        on="id_assessment",
        how="left",
        suffixes=("", "_assessment"),
    )


def raw_features(
    tables: dict[str, pd.DataFrame],
    module: str,
    presentations: list[str],
    checkpoint: int,
) -> pd.DataFrame:
    learners = cohort(tables, module, presentations)[["code_module", "code_presentation", "id_student", "learner_key", "label"]]
    features = learners[["learner_key", "label"]].copy()
    for feature in FEATURES:
        features[feature] = 0.0

    vle = tables["studentVle"]
    vle = vle[
        (vle["code_module"] == module)
        & (vle["code_presentation"].isin(presentations))
        & (vle["date"] >= 0)
        & (vle["date"] < checkpoint)
    ].copy()
    if not vle.empty:
        vle["learner_key"] = make_learner_key(vle)
        grouped = vle.groupby("learner_key")
        activity = pd.DataFrame({
            "total_clicks": grouped["sum_click"].sum(),
            "active_days": grouped["date"].nunique(),
            "distinct_resources": grouped["id_site"].nunique(),
            "last_day": grouped["date"].max(),
        })
        last = vle[vle["date"].between(checkpoint - 14, checkpoint - 1)].groupby("learner_key")["sum_click"].sum()
        prev = vle[vle["date"].between(checkpoint - 28, checkpoint - 15)].groupby("learner_key")["sum_click"].sum()
        activity["activity_trend"] = np.log1p(last).sub(np.log1p(prev), fill_value=0)
        activity["days_since_last"] = checkpoint - activity["last_day"]
        features = features.merge(activity.drop(columns=["last_day"]), how="left", left_on="learner_key", right_index=True)
        for col in ["total_clicks", "active_days", "distinct_resources", "activity_trend", "days_since_last"]:
            features[col] = features[f"{col}_y"].fillna(features[f"{col}_x"]).fillna(0)
            features = features.drop(columns=[f"{col}_x", f"{col}_y"])
    features["days_since_last"] = features["days_since_last"].replace(0, checkpoint + 1)

    assess = _assessment_join(tables)
    assess = assess[
        (assess["code_module"] == module)
        & (assess["code_presentation"].isin(presentations))
        & (assess["date"].notna())
        & (assess["date"] < checkpoint)
    ].copy()
    if not assess.empty:
        assess["learner_key"] = (
            assess["id_student"].astype(str)
            + "_"
            + assess["code_module"].astype(str)
            + "_"
            + assess["code_presentation"].astype(str)
        )
        visible = assess[assess["date_submitted"].notna() & (assess["date_submitted"] < checkpoint)]
        if not visible.empty:
            g = visible.groupby("learner_key")
            late_counts = visible.assign(_late=(visible["date_submitted"] > visible["date"]).astype(int)).groupby("learner_key")["_late"].sum()
            submitted = pd.DataFrame({
                "n_submitted": g["id_assessment"].nunique(),
                "avg_score": g["score"].mean(),
                "late_submission": late_counts,
            })
            features = features.merge(submitted, how="left", left_on="learner_key", right_index=True, suffixes=("", "_new"))
            for col in ["n_submitted", "avg_score", "late_submission"]:
                features[col] = features[f"{col}_new"].fillna(features[col]).fillna(0)
                features = features.drop(columns=[f"{col}_new"])
        due_counts = (
            assess.groupby(["code_module", "code_presentation"])["id_assessment"]
            .nunique()
            .rename("due")
            .reset_index()
        )
        submitted_counts = visible.groupby("learner_key")["id_assessment"].nunique().rename("submitted_due")
        due_by_learner = learners.merge(due_counts, how="left", on=["code_module", "code_presentation"]).set_index("learner_key")["due"].fillna(0)
        features = features.merge(submitted_counts, how="left", left_on="learner_key", right_index=True)
        features["submitted_due"] = features["submitted_due"].fillna(0)
        features["missing_assessment"] = (features["learner_key"].map(due_by_learner) - features["submitted_due"]).clip(lower=0)
        features = features.drop(columns=["submitted_due"])
    return features[["learner_key", "label", *FEATURES]].fillna(0)


def safe_uri(text: Any) -> str:
    return str(text).replace(" ", "_").replace("/", "_").replace("#", "_")


def decimal_literal(value: Any) -> Literal:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Cannot serialize non-finite decimal literal: {value!r}")
    decimal = Decimal(str(numeric))
    return Literal(format(decimal, "f"), datatype=XSD.decimal)


def string_literal(value: Any) -> Literal:
    return Literal(str(value), datatype=XSD.string)


def learner_uri(ns: Namespace, key: str) -> URIRef:
    return ns[f"Learner_{safe_uri(key)}"]


def build_observation_graph(
    tables: dict[str, pd.DataFrame],
    module: str,
    presentations: list[str],
    checkpoint: int,
    namespace: str = "http://example.org/ldt#",
) -> tuple[Graph, dict[str, Any]]:
    start = time.perf_counter()
    ns = Namespace(namespace)
    g = Graph()
    g.bind("ldt", ns)
    cp = ns[f"Checkpoint_day{checkpoint}"]
    g.add((cp, RDF.type, ns.Checkpoint))
    g.add((cp, ns.checkpointDay, Literal(checkpoint, datatype=XSD.integer)))

    learners = cohort(tables, module, presentations)
    learner_keys = set(learners["learner_key"].astype(str))
    for row in learners.itertuples(index=False):
        key = f"{row.id_student}_{row.code_module}_{row.code_presentation}"
        lu = learner_uri(ns, key)
        cu = ns[f"Course_{row.code_module}_{row.code_presentation}"]
        g.add((lu, RDF.type, ns.Learner))
        g.add((lu, ns.enrolledIn, cu))
        g.add((cu, RDF.type, ns.Course))

    registrations = tables["studentRegistration"]
    registrations = registrations[
        (registrations["code_module"] == module)
        & (registrations["code_presentation"].isin(presentations))
    ].copy()
    if not registrations.empty:
        registrations["learner_key"] = make_learner_key(registrations)
        registrations = registrations[registrations["learner_key"].isin(learner_keys)].copy()
    for row in registrations.itertuples():
        key = f"{row.id_student}_{row.code_module}_{row.code_presentation}"
        event = ns[f"RegistrationEvent_{safe_uri(key)}"]
        raw = ns[f"Raw_studentRegistration_{row.Index}"]
        g.add((event, RDF.type, ns.RegistrationEvent))
        g.add((event, ns.registrationOf, learner_uri(ns, key)))
        g.add((event, ns.fromRecord, raw))
        g.add((raw, RDF.type, ns.RawRecord))
        g.add((raw, ns.sourceTable, string_literal("studentRegistration")))
        g.add((raw, ns.rowId, string_literal(row.Index)))

    vle = tables["vle"]
    for row in vle[vle["code_module"].eq(module) & vle["code_presentation"].isin(presentations)].itertuples():
        g.add((ns[f"Resource_{row.id_site}"], RDF.type, ns.LearningResource))

    svle = tables["studentVle"]
    svle = svle[
        (svle["code_module"] == module)
        & (svle["code_presentation"].isin(presentations))
        & (svle["date"] >= 0)
        & (svle["date"] < checkpoint)
    ].copy()
    if not svle.empty:
        svle["learner_key"] = make_learner_key(svle)
        svle = svle[svle["learner_key"].isin(learner_keys)].copy()
    for row in svle.itertuples():
        key = f"{row.id_student}_{row.code_module}_{row.code_presentation}"
        trace = ns[f"Trace_{safe_uri(key)}_{row.id_site}_{int(row.date)}_{row.Index}"]
        raw = ns[f"Raw_studentVle_{row.Index}"]
        g.add((trace, RDF.type, ns.LearningTrace))
        g.add((trace, ns.performedBy, learner_uri(ns, key)))
        g.add((trace, ns.concerns, ns[f"Resource_{row.id_site}"]))
        g.add((trace, ns.hasClickCount, Literal(int(row.sum_click), datatype=XSD.integer)))
        g.add((trace, ns.onDay, Literal(int(row.date), datatype=XSD.integer)))
        g.add((trace, ns.fromRecord, raw))
        g.add((raw, RDF.type, ns.RawRecord))
        g.add((raw, ns.sourceTable, string_literal("studentVle")))
        g.add((raw, ns.rowId, string_literal(row.Index)))

    assessments = tables["assessments"]
    for row in assessments[
        (assessments["code_module"] == module)
        & (assessments["code_presentation"].isin(presentations))
        & (assessments["date"].notna())
        & (assessments["date"] < checkpoint)
    ].itertuples():
        au = ns[f"Assessment_{row.id_assessment}"]
        cu = ns[f"Course_{row.code_module}_{row.code_presentation}"]
        raw = ns[f"Raw_assessments_{row.Index}"]
        g.add((au, RDF.type, ns.Assessment))
        g.add((au, ns.inCourse, cu))
        g.add((au, ns.onDay, Literal(int(row.date), datatype=XSD.integer)))
        g.add((au, ns.fromRecord, raw))
        g.add((raw, RDF.type, ns.RawRecord))
        g.add((raw, ns.sourceTable, string_literal("assessments")))
        g.add((raw, ns.rowId, string_literal(row.Index)))

    assess = _assessment_join(tables)
    assess = assess[
        (assess["code_module"] == module)
        & (assess["code_presentation"].isin(presentations))
        & (assess["date"].notna())
        & (assess["date"] < checkpoint)
        & (assess["date_submitted"].notna())
        & (assess["date_submitted"] < checkpoint)
    ].copy()
    if not assess.empty:
        assess["learner_key"] = (
            assess["id_student"].astype(str)
            + "_"
            + assess["code_module"].astype(str)
            + "_"
            + assess["code_presentation"].astype(str)
        )
        assess = assess[assess["learner_key"].isin(learner_keys)].copy()
    for row in assess.itertuples():
        key = f"{row.id_student}_{row.code_module}_{row.code_presentation}"
        result = ns[f"AssessmentResult_{safe_uri(key)}_{row.id_assessment}"]
        raw = ns[f"Raw_studentAssessment_{row.Index}"]
        g.add((result, RDF.type, ns.AssessmentResult))
        g.add((result, ns.concernsAssessment, ns[f"Assessment_{row.id_assessment}"]))
        if pd.notna(row.score):
            g.add((result, ns.hasScore, decimal_literal(row.score)))
        g.add((result, ns.submittedOnDay, Literal(int(row.date_submitted), datatype=XSD.integer)))
        g.add((result, ns.fromRecord, raw))
        g.add((learner_uri(ns, key), ns.submits, result))
        g.add((raw, RDF.type, ns.RawRecord))
        g.add((raw, ns.sourceTable, string_literal("studentAssessment")))
        g.add((raw, ns.rowId, string_literal(row.Index)))

    stats = {
        "learners": int(len(learners)),
        "traces": int(len(svle)),
        "assessment_results": int(len(assess)),
        "triples": int(len(g)),
        "update_time": round(time.perf_counter() - start, 3),
    }
    return g, stats


def initialize_observation_graph(
    tables: dict[str, pd.DataFrame],
    module: str,
    presentations: list[str],
    namespace: str = "http://example.org/ldt#",
) -> tuple[Graph, set[str], dict[str, set[int]]]:
    ns = Namespace(namespace)
    graph = Graph()
    graph.bind("ldt", ns)
    learners = cohort(tables, module, presentations)
    learner_keys = set(learners["learner_key"].astype(str))
    for row in learners.itertuples(index=False):
        key = f"{row.id_student}_{row.code_module}_{row.code_presentation}"
        learner = learner_uri(ns, key)
        course = ns[f"Course_{row.code_module}_{row.code_presentation}"]
        graph.add((learner, RDF.type, ns.Learner))
        graph.add((learner, ns.enrolledIn, course))
        graph.add((course, RDF.type, ns.Course))

    registrations = tables["studentRegistration"]
    registrations = registrations[
        (registrations["code_module"] == module)
        & (registrations["code_presentation"].isin(presentations))
    ].copy()
    if not registrations.empty:
        registrations["learner_key"] = make_learner_key(registrations)
        registrations = registrations[registrations["learner_key"].isin(learner_keys)].copy()
    for row in registrations.itertuples():
        key = f"{row.id_student}_{row.code_module}_{row.code_presentation}"
        event = ns[f"RegistrationEvent_{safe_uri(key)}"]
        raw = ns[f"Raw_studentRegistration_{row.Index}"]
        graph.add((event, RDF.type, ns.RegistrationEvent))
        graph.add((event, ns.registrationOf, learner_uri(ns, key)))
        graph.add((event, ns.fromRecord, raw))
        graph.add((raw, RDF.type, ns.RawRecord))
        graph.add((raw, ns.sourceTable, string_literal("studentRegistration")))
        graph.add((raw, ns.rowId, string_literal(row.Index)))

    return graph, learner_keys, {"studentVle": set(), "assessments": set(), "studentAssessment": set()}


def update_observation_graph(
    graph: Graph,
    tables: dict[str, pd.DataFrame],
    module: str,
    presentations: list[str],
    checkpoint: int,
    learner_keys: set[str],
    seen: dict[str, set[int]],
    namespace: str = "http://example.org/ldt#",
) -> dict[str, Any]:
    start = time.perf_counter()
    before = len(graph)
    ns = Namespace(namespace)
    cp = ns[f"Checkpoint_day{checkpoint}"]
    graph.add((cp, RDF.type, ns.Checkpoint))
    graph.add((cp, ns.checkpointDay, Literal(checkpoint, datatype=XSD.integer)))

    traces_added = 0
    svle = tables["studentVle"]
    svle = svle[
        (svle["code_module"] == module)
        & (svle["code_presentation"].isin(presentations))
        & (svle["date"] >= 0)
        & (svle["date"] < checkpoint)
    ].copy()
    if not svle.empty:
        svle["learner_key"] = make_learner_key(svle)
        svle = svle[svle["learner_key"].isin(learner_keys)].copy()
    for row in svle.itertuples():
        if int(row.Index) in seen["studentVle"]:
            continue
        seen["studentVle"].add(int(row.Index))
        key = f"{row.id_student}_{row.code_module}_{row.code_presentation}"
        resource = ns[f"Resource_{row.id_site}"]
        trace = ns[f"Trace_{safe_uri(key)}_{row.id_site}_{int(row.date)}_{row.Index}"]
        raw = ns[f"Raw_studentVle_{row.Index}"]
        graph.add((resource, RDF.type, ns.LearningResource))
        graph.add((trace, RDF.type, ns.LearningTrace))
        graph.add((trace, ns.performedBy, learner_uri(ns, key)))
        graph.add((trace, ns.concerns, resource))
        graph.add((trace, ns.hasClickCount, Literal(int(row.sum_click), datatype=XSD.integer)))
        graph.add((trace, ns.onDay, Literal(int(row.date), datatype=XSD.integer)))
        graph.add((trace, ns.fromRecord, raw))
        graph.add((raw, RDF.type, ns.RawRecord))
        graph.add((raw, ns.sourceTable, string_literal("studentVle")))
        graph.add((raw, ns.rowId, string_literal(row.Index)))
        traces_added += 1

    assessments = tables["assessments"]
    assessments = assessments[
        (assessments["code_module"] == module)
        & (assessments["code_presentation"].isin(presentations))
        & (assessments["date"].notna())
        & (assessments["date"] < checkpoint)
    ].copy()
    for row in assessments.itertuples():
        if int(row.Index) in seen["assessments"]:
            continue
        seen["assessments"].add(int(row.Index))
        assessment = ns[f"Assessment_{row.id_assessment}"]
        course = ns[f"Course_{row.code_module}_{row.code_presentation}"]
        raw = ns[f"Raw_assessments_{row.Index}"]
        graph.add((assessment, RDF.type, ns.Assessment))
        graph.add((assessment, ns.inCourse, course))
        graph.add((assessment, ns.onDay, Literal(int(row.date), datatype=XSD.integer)))
        graph.add((assessment, ns.fromRecord, raw))
        graph.add((raw, RDF.type, ns.RawRecord))
        graph.add((raw, ns.sourceTable, string_literal("assessments")))
        graph.add((raw, ns.rowId, string_literal(row.Index)))

    results_added = 0
    assess = _assessment_join(tables)
    assess = assess[
        (assess["code_module"] == module)
        & (assess["code_presentation"].isin(presentations))
        & (assess["date"].notna())
        & (assess["date"] < checkpoint)
        & (assess["date_submitted"].notna())
        & (assess["date_submitted"] < checkpoint)
    ].copy()
    if not assess.empty:
        assess["learner_key"] = (
            assess["id_student"].astype(str)
            + "_"
            + assess["code_module"].astype(str)
            + "_"
            + assess["code_presentation"].astype(str)
        )
        assess = assess[assess["learner_key"].isin(learner_keys)].copy()
    for row in assess.itertuples():
        if int(row.Index) in seen["studentAssessment"]:
            continue
        seen["studentAssessment"].add(int(row.Index))
        key = f"{row.id_student}_{row.code_module}_{row.code_presentation}"
        result = ns[f"AssessmentResult_{safe_uri(key)}_{row.id_assessment}"]
        raw = ns[f"Raw_studentAssessment_{row.Index}"]
        graph.add((result, RDF.type, ns.AssessmentResult))
        graph.add((result, ns.concernsAssessment, ns[f"Assessment_{row.id_assessment}"]))
        if pd.notna(row.score):
            graph.add((result, ns.hasScore, decimal_literal(row.score)))
        graph.add((result, ns.submittedOnDay, Literal(int(row.date_submitted), datatype=XSD.integer)))
        graph.add((result, ns.fromRecord, raw))
        graph.add((learner_uri(ns, key), ns.submits, result))
        graph.add((raw, RDF.type, ns.RawRecord))
        graph.add((raw, ns.sourceTable, string_literal("studentAssessment")))
        graph.add((raw, ns.rowId, string_literal(row.Index)))
        results_added += 1

    return {
        "learners": int(len(learner_keys)),
        "new_traces": int(traces_added),
        "new_assessment_results": int(results_added),
        "new_triples": int(len(graph) - before),
        "cumulative_traces": int(len(set(graph.subjects(RDF.type, ns.LearningTrace)))),
        "cumulative_assessment_results": int(len(set(graph.subjects(RDF.type, ns.AssessmentResult)))),
        "cumulative_triples": int(len(graph)),
        "update_time": round(time.perf_counter() - start, 3),
    }


def _kg_features_indexed(
    graph: Graph,
    raw_frame: pd.DataFrame,
    checkpoint: int,
    namespace: str = "http://example.org/ldt#",
) -> pd.DataFrame:
    ns = Namespace(namespace)

    def learner_key(uri: Any) -> str:
        return str(uri).split("Learner_", 1)[1]

    learners = list(graph.subjects(RDF.type, ns.Learner))
    rows = [{"learner_key": learner_key(learner), "_learner": learner} for learner in learners]
    out = pd.DataFrame(rows)
    for feature in FEATURES:
        out[feature] = 0.0

    clicks: dict[URIRef, list[tuple[int, int, URIRef]]] = defaultdict(list)
    for trace in graph.subjects(RDF.type, ns.LearningTrace):
        learner = graph.value(trace, ns.performedBy)
        day = graph.value(trace, ns.onDay)
        click_count = graph.value(trace, ns.hasClickCount)
        resource = graph.value(trace, ns.concerns)
        if learner is None or day is None or click_count is None:
            continue
        clicks[learner].append((int(day), int(click_count), resource))

    learner_to_courses: dict[URIRef, set[URIRef]] = defaultdict(set)
    for learner in learners:
        learner_to_courses[learner].update(graph.objects(learner, ns.enrolledIn))

    due_by_course: dict[URIRef, set[URIRef]] = defaultdict(set)
    assessment_due_day: dict[URIRef, int] = {}
    for assessment in graph.subjects(RDF.type, ns.Assessment):
        course = graph.value(assessment, ns.inCourse)
        due_day = graph.value(assessment, ns.onDay)
        if course is not None:
            due_by_course[course].add(assessment)
        if due_day is not None:
            assessment_due_day[assessment] = int(due_day)

    submitted_by_learner: dict[URIRef, set[URIRef]] = defaultdict(set)
    scores_by_learner: dict[URIRef, list[float]] = defaultdict(list)
    late_by_learner: dict[URIRef, int] = defaultdict(int)
    for learner in learners:
        for result in graph.objects(learner, ns.submits):
            assessment = graph.value(result, ns.concernsAssessment)
            if assessment is not None:
                submitted_by_learner[learner].add(assessment)
            score = graph.value(result, ns.hasScore)
            if score is not None:
                scores_by_learner[learner].append(float(score))
            submit_day = graph.value(result, ns.submittedOnDay)
            due_day = assessment_due_day.get(assessment)
            if submit_day is not None and due_day is not None and int(submit_day) > due_day:
                late_by_learner[learner] += 1

    def feature_values(learner: URIRef) -> dict[str, float]:
        learner_clicks = clicks.get(learner, [])
        days = {day for day, _, _ in learner_clicks}
        resources = {resource for _, _, resource in learner_clicks if resource is not None}
        last_day = max(days) if days else None
        scores = scores_by_learner.get(learner, [])
        due = set()
        for course in learner_to_courses.get(learner, set()):
            due.update(due_by_course.get(course, set()))
        submitted = submitted_by_learner.get(learner, set())
        recent = sum(click_count for day, click_count, _ in learner_clicks if checkpoint - 14 <= day <= checkpoint - 1)
        previous = sum(click_count for day, click_count, _ in learner_clicks if checkpoint - 28 <= day <= checkpoint - 15)
        return {
            "total_clicks": float(sum(click_count for _, click_count, _ in learner_clicks)),
            "active_days": float(len(days)),
            "distinct_resources": float(len(resources)),
            "activity_trend": float(math.log1p(recent) - math.log1p(previous)),
            "days_since_last": float(checkpoint - last_day) if last_day is not None else float(checkpoint + 1),
            "n_submitted": float(len(submitted)),
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "missing_assessment": float(max(len(due) - len(submitted), 0)),
            "late_submission": float(late_by_learner.get(learner, 0)),
        }

    for idx, row in out.iterrows():
        values = feature_values(row["_learner"])
        for feature, value in values.items():
            out.at[idx, feature] = value
    out = out.drop(columns=["_learner"])
    out = out.merge(raw_frame[["learner_key", "label"]], on="learner_key", how="left")
    return out[["learner_key", "label", *FEATURES]].fillna(0)


def _kg_features_sparql(
    graph: Graph,
    raw_frame: pd.DataFrame,
    checkpoint: int,
    namespace: str = "http://example.org/ldt#",
) -> pd.DataFrame:
    ns = Namespace(namespace)
    def learner_key(uri: Any) -> str:
        return str(uri).split("Learner_", 1)[1]

    rows = [{"learner_key": learner_key(learner)} for learner in graph.subjects(RDF.type, ns.Learner)]
    out = pd.DataFrame(rows)
    for feature in FEATURES:
        out[feature] = 0.0

    def run_query(name: str, bindings: dict[str, Any] | None = None):
        return graph.query(prepare_feature_query(name), initBindings=bindings or {})

    simple_queries = {
        "total_clicks": "total_clicks",
        "active_days": "active_days",
        "distinct_resources": "distinct_resources",
        "n_submitted": "n_submitted",
        "avg_score": "avg_score",
    }
    for feature, query_name in simple_queries.items():
        values = {}
        for row in run_query(query_name):
            val = getattr(row, feature)
            values[learner_key(row.learner)] = float(val) if val is not None else 0.0
        out[feature] = out["learner_key"].map(values).fillna(0.0)

    last_active = {}
    for row in run_query("days_since_last"):
        if row.last_active_day is not None:
            last_active[learner_key(row.learner)] = int(row.last_active_day)
    out["days_since_last"] = out["learner_key"].map(
        lambda key: float(checkpoint - last_active[key]) if key in last_active else float(checkpoint + 1)
    )

    clicks_by_learner: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in run_query(
        "activity_trend",
        {
            Variable("window_start"): Literal(checkpoint - 28, datatype=XSD.integer),
            Variable("window_end"): Literal(checkpoint - 1, datatype=XSD.integer),
        },
    ):
        clicks_by_learner[learner_key(row.learner)].append((int(row.day), int(row.clicks)))
    out["activity_trend"] = out["learner_key"].map(
        lambda key: float(
            math.log1p(sum(c for d, c in clicks_by_learner.get(key, []) if checkpoint - 14 <= d <= checkpoint - 1))
            - math.log1p(sum(c for d, c in clicks_by_learner.get(key, []) if checkpoint - 28 <= d <= checkpoint - 15))
        )
    )

    missing = {}
    for row in run_query("missing_assessment"):
        due = int(row.due_assessments) if row.due_assessments is not None else 0
        submitted = int(row.submitted_assessments) if row.submitted_assessments is not None else 0
        missing[learner_key(row.learner)] = max(due - submitted, 0)
    out["missing_assessment"] = out["learner_key"].map(missing).fillna(0.0)

    late_counts: dict[str, int] = defaultdict(int)
    for row in run_query("late_submission"):
        late_counts[learner_key(row.learner)] += 1
    out["late_submission"] = out["learner_key"].map(late_counts).fillna(0.0)

    out = out.merge(raw_frame[["learner_key", "label"]], on="learner_key", how="left")
    return out[["learner_key", "label", *FEATURES]].fillna(0)


def kg_features(
    graph: Graph,
    raw_frame: pd.DataFrame,
    checkpoint: int,
    namespace: str = "http://example.org/ldt#",
    engine: str = "indexed",
) -> pd.DataFrame:
    if engine == "sparql":
        return _kg_features_sparql(graph, raw_frame, checkpoint, namespace)
    if engine == "indexed":
        return _kg_features_indexed(graph, raw_frame, checkpoint, namespace)
    raise ValueError(f"Unknown KG feature engine: {engine}")


def validate_graph(graph: Graph, shapes_path: str | Path = "ontology/ldt-shapes.ttl") -> tuple[bool, int]:
    if shacl_validate is None:
        return True, -1
    conforms, report_graph, _ = shacl_validate(graph, shacl_graph=str(shapes_path), inference="rdfs")
    violations = len(list(report_graph.subjects(RDF.type, URIRef("http://www.w3.org/ns/shacl#ValidationResult"))))
    return bool(conforms), int(violations)


def model_specs(cfg: dict[str, Any]) -> list[dict[str, str]]:
    specs = cfg.get("models")
    if specs:
        return specs
    return EXTENDED_MODEL_SPECS if cfg.get("extended_models", False) else DEFAULT_MODEL_SPECS


def _class_counts(y: pd.Series) -> tuple[int, int, float]:
    pos = max(int(y.sum()), 1)
    neg = max(int((1 - y).sum()), 1)
    return pos, neg, neg / pos


def build_model(method: str, y_train: pd.Series, cfg: dict[str, Any]) -> Any:
    random_state = int(cfg.get("random_state", 42))
    pos, neg, scale_pos_weight = _class_counts(y_train)
    max_scale_pos_weight = cfg.get("max_scale_pos_weight", 20)
    if max_scale_pos_weight is not None:
        scale_pos_weight = min(scale_pos_weight, float(max_scale_pos_weight))
    if method == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    if method == "RF":
        params = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "n_jobs": cfg.get("n_jobs", 1),
            "random_state": random_state,
            **cfg.get("random_forest", {}),
        }
        return RandomForestClassifier(**params)
    if method == "ET":
        params = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "n_jobs": cfg.get("n_jobs", 1),
            "random_state": random_state,
            **cfg.get("extra_trees", {}),
        }
        return ExtraTreesClassifier(**params)
    if method == "HGB":
        params = {
            "learning_rate": 0.05,
            "max_iter": 300,
            "l2_regularization": 0.1,
            "random_state": random_state,
            **cfg.get("hist_gradient_boosting", {}),
        }
        return HistGradientBoostingClassifier(**params)
    if method == "XGB":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed. Install requirements.txt to run XGBoost experiments.")
        params = dict(cfg.get("xgboost", {}))
        default_xgb_jobs = 1 if cfg.get("xgboost_grid_search", False) else cfg.get("n_jobs", 1)
        params.setdefault("n_jobs", cfg.get("xgboost_n_jobs", default_xgb_jobs))
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            **params,
        )
        if cfg.get("xgboost_grid_search", False):
            search_space = cfg.get("xgboost_search_space", {
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [1, 5, 10],
            })
            min_class = max(2, int(min(pos, neg)))
            cv = StratifiedKFold(n_splits=min(3, min_class), shuffle=True, random_state=int(cfg.get("random_state", 42)))
            model = GridSearchCV(model, search_space, scoring="roc_auc", cv=cv, n_jobs=cfg.get("n_jobs", 1))
        return model
    if method == "LGBM":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed. Install requirements.txt to run LightGBM experiments.")
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": random_state,
            "n_jobs": cfg.get("n_jobs", 1),
            **cfg.get("lightgbm", {}),
        }
        return LGBMClassifier(**params)
    if method == "CAT":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed. Install requirements.txt to run CatBoost experiments.")
        params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": False,
            "random_seed": random_state,
            "class_weights": [1.0, scale_pos_weight],
            "thread_count": cfg.get("n_jobs", -1),
            **cfg.get("catboost", {}),
        }
        return CatBoostClassifier(**params)
    if method == "TABTX":
        params = {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "max_epochs": 50,
            "patience": 8,
            "val_fraction": 0.15,
            "random_state": random_state,
            **cfg.get("tab_transformer", {}),
        }
        return TabTransformerClassifier(**params)
    raise ValueError(f"Unknown method: {method}")


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, method: str, cfg: dict[str, Any]) -> ModelEvaluation:
    x_train = train[FEATURES]
    y_train = train["label"]
    x_test = test[FEATURES]
    y_test = test["label"]
    model = build_model(method, y_train, cfg)
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x_test)[:, 1]
    else:
        prob = model[-1].predict_proba(model[:-1].transform(x_test))[:, 1]
    pred = (prob >= float(cfg.get("alert_threshold", 0.5))).astype(int)
    auc = roc_auc_score(y_test, prob) if len(set(y_test)) > 1 else np.nan
    return ModelEvaluation(
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        auc=auc,
        model=model,
        prob=prob,
    )


def shap_values(model: Any, frame: pd.DataFrame) -> np.ndarray:
    model = getattr(model, "best_estimator_", model)
    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(frame[FEATURES])
            return np.asarray(values[1] if isinstance(values, list) else values)
        except Exception:
            # SHAP can lag behind XGBoost model metadata changes. XGBoost's
            # native contribution API gives per-feature SHAP contributions plus
            # a final bias column, so it is the safest fallback for XGB models.
            pass
    if XGBClassifier is not None and isinstance(model, XGBClassifier):
        try:
            import xgboost as xgb

            dmatrix = xgb.DMatrix(frame[FEATURES])
            contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
            return np.asarray(contribs[:, :-1])
        except Exception:
            pass
    raise RuntimeError(
        "Could not compute SHAP values. Install compatible shap/model versions "
        "or use XGBoost, which supports the native pred_contribs fallback."
    )


def evidence_for_feature(raw_row: pd.Series, feature: str) -> dict[str, Any]:
    value = float(raw_row[feature])
    if feature in {"total_clicks", "active_days", "distinct_resources"}:
        etype = "activity"
        text = f"{feature} is {value:.2f} before the checkpoint."
    elif feature in {"avg_score", "n_submitted", "late_submission"}:
        etype = "assessment"
        text = f"{feature} is {value:.2f} for assessments due before the checkpoint."
    elif feature in {"activity_trend", "days_since_last"}:
        etype = "temporal"
        text = f"{feature} is {value:.2f}, summarising recent temporal activity."
    else:
        etype = "absence"
        text = f"{feature} is {value:.2f}, indicating due work without a visible submission."
    return {"type": etype, "text": text, "strength": abs(value)}


def evidence_class_for_type(ns: Namespace, evidence_type: str) -> URIRef:
    return {
        "activity": ns.ActivityEvidence,
        "assessment": ns.AssessmentEvidence,
        "temporal": ns.TemporalEvidence,
        "absence": ns.AbsenceEvidence,
    }.get(evidence_type, ns.EvidenceItem)


def source_records_for_feature(
    graph: Graph,
    learner: URIRef,
    feature: str,
    checkpoint: int,
    namespace: str,
) -> tuple[list[URIRef], str]:
    ns = Namespace(namespace)
    records: list[URIRef] = []

    def registration_records() -> list[URIRef]:
        out = []
        for event in graph.subjects(ns.registrationOf, learner):
            raw = graph.value(event, ns.fromRecord)
            if raw is not None:
                out.append(raw)
        return out

    def course_assessment_records() -> list[URIRef]:
        out = []
        for course in graph.objects(learner, ns.enrolledIn):
            for assessment in graph.subjects(RDF.type, ns.Assessment):
                if (assessment, ns.inCourse, course) in graph:
                    raw = graph.value(assessment, ns.fromRecord)
                    if raw is not None:
                        out.append(raw)
        return out

    if feature in {"total_clicks", "active_days", "distinct_resources", "activity_trend", "days_since_last"}:
        traces = list(graph.subjects(ns.performedBy, learner))
        if feature == "days_since_last":
            dated = []
            for trace in traces:
                day = graph.value(trace, ns.onDay)
                if day is not None:
                    dated.append((int(day), trace))
            traces = [trace for _, trace in sorted(dated, reverse=True)[:1]]
        elif feature == "activity_trend":
            traces = [
                trace for trace in traces
                if (day := graph.value(trace, ns.onDay)) is not None and checkpoint - 28 <= int(day) <= checkpoint - 1
            ]
        for trace in traces[:100]:
            raw = graph.value(trace, ns.fromRecord)
            if raw is not None:
                records.append(raw)
        if not records:
            records.extend(registration_records())
            return list(dict.fromkeys(records)), "indirect" if records else "synthetic"
        return list(dict.fromkeys(records)), "direct"
    elif feature in {"n_submitted", "avg_score", "late_submission"}:
        for result in list(graph.objects(learner, ns.submits))[:100]:
            if feature == "avg_score" and graph.value(result, ns.hasScore) is None:
                continue
            if feature == "late_submission":
                submit_day = graph.value(result, ns.submittedOnDay)
                assessment = graph.value(result, ns.concernsAssessment)
                due_day = graph.value(assessment, ns.onDay) if assessment is not None else None
                if submit_day is None or due_day is None or int(submit_day) <= int(due_day):
                    continue
            raw = graph.value(result, ns.fromRecord)
            if raw is not None:
                records.append(raw)
        if not records:
            records.extend(registration_records())
            records.extend(course_assessment_records())
            return list(dict.fromkeys(records)), "indirect" if records else "synthetic"
        return list(dict.fromkeys(records)), "direct"
    elif feature == "missing_assessment":
        records.extend(registration_records())
        records.extend(course_assessment_records())
        return list(dict.fromkeys(records)), "indirect" if records else "synthetic"
    return list(dict.fromkeys(records)), "direct" if records else "synthetic"


def materialize_alerts(
    graph: Graph,
    feature_frame: pd.DataFrame,
    model: Any,
    probabilities: np.ndarray,
    checkpoint: int,
    namespace: str,
    threshold: float,
    top_k: int | list[int] | tuple[int, ...] = 5,
) -> dict[str, float]:
    ns = Namespace(namespace)
    top_ks = sorted({int(k) for k in (top_k if isinstance(top_k, (list, tuple)) else [top_k]) if int(k) > 0})
    if not top_ks:
        top_ks = [5]
    max_top_k = max(top_ks)
    run = ns[f"PredictionRun_XGB_LDT_day{checkpoint}"]
    cp = ns[f"Checkpoint_day{checkpoint}"]
    graph.add((run, RDF.type, ns.PredictionRun))
    graph.add((run, ns.modelName, string_literal("XGB-LDT")))
    shap_matrix = shap_values(model, feature_frame)
    coverage_direct = {k: [] for k in top_ks}
    coverage_any = {k: [] for k in top_ks}
    fidelity3 = []
    evidence_count = 0
    alerts = 0
    traceable = 0
    for idx, row in feature_frame.reset_index(drop=True).iterrows():
        if probabilities[idx] < threshold:
            continue
        alerts += 1
        key = row["learner_key"]
        learner = learner_uri(ns, key)
        alert = ns[f"RiskAlert_{safe_uri(key)}_day{checkpoint}"]
        graph.add((alert, RDF.type, ns.RiskAlert))
        graph.add((alert, ns.describes, learner))
        graph.add((alert, ns.atCheckpoint, cp))
        graph.add((alert, ns.generatedBy, run))
        graph.add((alert, ns.riskProbability, decimal_literal(probabilities[idx])))

        ranked = sorted(range(len(FEATURES)), key=lambda j: abs(shap_matrix[idx, j]), reverse=True)
        selected_types = set()
        presented = []
        feature_direct_by_rank: list[bool] = []
        feature_any_by_rank: list[bool] = []
        for rank_pos, feat_idx in enumerate(ranked[:max_top_k], start=1):
            feature = FEATURES[feat_idx]
            feat_node = ns[f"PredictionFeature_{safe_uri(key)}_{feature}_day{checkpoint}"]
            attr = ns[f"Attribution_{safe_uri(key)}_{feature}_day{checkpoint}"]
            evi = ns[f"Evidence_{safe_uri(key)}_{feature}_day{checkpoint}"]
            e = evidence_for_feature(row, feature)
            platform_records, coverage_kind = source_records_for_feature(graph, learner, feature, checkpoint, namespace)
            source_records = platform_records
            platform_complete = any((raw, RDF.type, ns.RawRecord) in graph for raw in platform_records)
            direct_complete = platform_complete and coverage_kind == "direct"
            if not source_records:
                source_records = [ns[f"Raw_feature_{safe_uri(key)}_{feature}_day{checkpoint}"]]
            graph.add((feat_node, RDF.type, ns.PredictionFeature))
            graph.add((feat_node, ns.describes, learner))
            graph.add((feat_node, ns.featureName, string_literal(feature)))
            graph.add((feat_node, ns.hasValue, decimal_literal(row[feature])))
            graph.add((feat_node, ns.atCheckpoint, cp))
            graph.add((feat_node, ns.supportedBy, evi))
            graph.add((attr, RDF.type, ns.FeatureAttribution))
            graph.add((attr, ns.refersTo, feat_node))
            graph.add((attr, ns.hasShapValue, decimal_literal(shap_matrix[idx, feat_idx])))
            graph.add((alert, ns.hasAttribution, attr))
            graph.add((evi, RDF.type, ns.EvidenceItem))
            graph.add((evi, RDF.type, evidence_class_for_type(ns, e["type"])))
            graph.add((evi, ns.evidenceType, string_literal(e["type"])))
            graph.add((evi, ns.coverageKind, string_literal(coverage_kind)))
            graph.add((evi, ns.text, string_literal(e["text"])))
            graph.add((evi, ns.strength, decimal_literal(e["strength"])))
            for raw in source_records:
                graph.add((evi, ns.derivedFrom, raw))
                graph.add((raw, RDF.type, ns.RawRecord))
                if str(raw).endswith(f"{feature}_day{checkpoint}"):
                    graph.add((raw, ns.sourceTable, string_literal("derivedFeature")))
                    graph.add((raw, ns.rowId, string_literal(f"{key}:{feature}:day{checkpoint}")))
            evidence_count += 1
            traceable += 1
            feature_direct_by_rank.append(direct_complete)
            feature_any_by_rank.append(platform_complete)
            if should_present_evidence(e["type"], selected_types, len(presented)):
                graph.add((alert, ns.justifiedBy, evi))
                selected_types.add(e["type"])
                presented.append(rank_pos)
        for k in top_ks:
            direct_window = feature_direct_by_rank[:k]
            any_window = feature_any_by_rank[:k]
            coverage_direct[k].append(sum(direct_window) / k if len(direct_window) == k else np.nan)
            coverage_any[k].append(sum(any_window) / k if len(any_window) == k else np.nan)
        if presented:
            fidelity3.append(sum(1 for pos in presented if pos <= 3) / len(presented))
    coverage_summary = {}
    for label, coverage in [("coverage_direct", coverage_direct), ("coverage_any", coverage_any)]:
        for k, values in coverage.items():
            finite_values = [v for v in values if not np.isnan(v)]
            coverage_summary[f"{label}_{k}"] = float(np.mean(finite_values)) if finite_values else np.nan
    return {
        "alerts": alerts,
        "evidence_items": evidence_count,
        "fidelity_3": float(np.mean(fidelity3)) if fidelity3 else np.nan,
        "traceability": traceable / evidence_count if evidence_count else np.nan,
        "avg_evidence_per_alert": evidence_count / alerts if alerts else 0.0,
    } | coverage_summary


def compare_features(raw: pd.DataFrame, kg: pd.DataFrame) -> dict[str, float]:
    joined = raw[["learner_key", *FEATURES]].merge(kg[["learner_key", *FEATURES]], on="learner_key", suffixes=("_raw", "_kg"))
    return {
        feature: float((joined[f"{feature}_raw"] - joined[f"{feature}_kg"]).abs().max())
        for feature in FEATURES
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    cfg = load_config(config_path)
    tables = preprocess(load_oulad(cfg["data_dir"]))
    modules_cfg = cfg.get("modules")
    modules = eligible_modules(tables) if modules_cfg is None else modules_cfg
    checkpoints = [Checkpoint(**cp) for cp in cfg["checkpoints"]]
    namespace = cfg.get("namespace", "http://example.org/ldt#")
    kg_feature_engine = str(cfg.get("kg_feature_engine", "indexed"))
    results_dir = Path(cfg.get("results_dir", "results"))
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict[str, Any]] = []
    construction_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    process_rows: list[dict[str, Any]] = []
    model_timing_rows: list[dict[str, Any]] = []
    training_history_rows: list[dict[str, Any]] = []

    for module in modules:
        module_start = time.perf_counter()
        train_pres, test_pres = split_presentations(tables, module)
        if not train_pres or not test_pres:
            continue
        print(
            f"[pipeline] Module {module}: train={','.join(train_pres)} test={','.join(test_pres)}",
            flush=True,
        )
        train_cohort = cohort(tables, module, train_pres)
        test_cohort = cohort(tables, module, test_pres)
        dataset_rows.append({
            "module": module,
            "train_presentations": ",".join(train_pres),
            "test_presentations": ",".join(test_pres),
            "train_learners": len(train_cohort),
            "test_learners": len(test_cohort),
            "at_risk_pct_train": round(100 * train_cohort["label"].mean(), 2),
            "at_risk_pct_test": round(100 * test_cohort["label"].mean(), 2),
        })
        train_graph, train_learner_keys, train_seen = initialize_observation_graph(tables, module, train_pres, namespace)
        test_graph, test_learner_keys, test_seen = initialize_observation_graph(tables, module, test_pres, namespace)
        for cp in checkpoints:
            checkpoint_start = time.perf_counter()
            print(f"[pipeline] {module} {cp.name}: updating graphs and extracting features", flush=True)
            step_start = time.perf_counter()
            train_raw = raw_features(tables, module, train_pres, cp.day)
            test_raw = raw_features(tables, module, test_pres, cp.day)
            raw_feature_time = time.perf_counter() - step_start
            print(
                f"[pipeline] {module} {cp.name}: raw features in {raw_feature_time:.1f}s",
                flush=True,
            )
            step_start = time.perf_counter()
            train_stats = update_observation_graph(
                train_graph, tables, module, train_pres, cp.day, train_learner_keys, train_seen, namespace
            )
            test_stats = update_observation_graph(
                test_graph, tables, module, test_pres, cp.day, test_learner_keys, test_seen, namespace
            )
            graph_update_time = time.perf_counter() - step_start
            print(
                f"[pipeline] {module} {cp.name}: graph update in {graph_update_time:.1f}s",
                flush=True,
            )
            step_start = time.perf_counter()
            train_kg = kg_features(train_graph, train_raw, cp.day, namespace, kg_feature_engine)
            test_kg = kg_features(test_graph, test_raw, cp.day, namespace, kg_feature_engine)
            kg_feature_time = time.perf_counter() - step_start
            print(
                f"[pipeline] {module} {cp.name}: KG features ({kg_feature_engine}) "
                f"in {kg_feature_time:.1f}s",
                flush=True,
            )
            step_start = time.perf_counter()
            diff = compare_features(test_raw, test_kg)
            conforms, violations = validate_graph(test_graph)
            validation_time = time.perf_counter() - step_start
            print(
                f"[pipeline] {module} {cp.name}: parity + SHACL in {validation_time:.1f}s "
                f"(conforms={conforms}, violations={violations})",
                flush=True,
            )
            construction_rows.append({
                "module": module,
                "checkpoint": cp.name,
                **test_stats,
                "shacl_conforms": conforms,
                "shacl_violations": violations,
            })
            parity_rows.append({"module": module, "checkpoint": cp.name, **diff})

            feature_frames = {
                "raw": (train_raw, test_raw),
                "kg": (train_kg, test_kg),
            }
            evidence_materialization_time = 0.0
            analytics_validation_time = 0.0
            model_train_eval_time = 0.0
            for spec in model_specs(cfg):
                method = spec["method"]
                source = spec["feature_source"]
                train_frame, test_frame = feature_frames[source]
                model_start = time.perf_counter()
                print(f"[pipeline] {module} {cp.name}: training {method}/{source}", flush=True)
                metrics = train_and_evaluate(train_frame, test_frame, method, cfg)
                model_time = time.perf_counter() - model_start
                model_train_eval_time += model_time
                print(
                    f"[pipeline] {module} {cp.name}: finished {method}/{source} "
                    f"auc={metrics['auc']:.4f} f1={metrics['f1']:.4f} "
                    f"in {model_time:.1f}s",
                    flush=True,
                )
                model_timing_rows.append({
                    "module": module,
                    "checkpoint": cp.name,
                    "method": method,
                    "feature_source": source,
                    "train_eval_time": round(model_time, 3),
                    "auc": metrics["auc"],
                    "f1": metrics["f1"],
                })
                fitted_model = getattr(metrics["model"], "best_estimator_", metrics["model"])
                for history_row in getattr(fitted_model, "history_", []) or []:
                    training_history_rows.append({
                        "module": module,
                        "checkpoint": cp.name,
                        "method": method,
                        "feature_source": source,
                        **history_row,
                    })
                performance_rows.append({
                    "module": module,
                    "checkpoint": cp.name,
                    "method": method,
                    "feature_source": source,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "auc": metrics["auc"],
                })
                if method == "XGB" and source == "kg":
                    grounding_start = time.perf_counter()
                    print(f"[pipeline] {module} {cp.name}: materializing XGB-LDT evidence", flush=True)
                    grounding = materialize_alerts(
                        test_graph,
                        test_kg,
                        metrics["model"],
                        metrics["prob"],
                        cp.day,
                        namespace,
                        float(cfg.get("alert_threshold", 0.5)),
                        cfg.get("top_k", [3, 5]),
                    )
                    evidence_materialization_time = time.perf_counter() - grounding_start
                    analytics_validation_start = time.perf_counter()
                    conforms2, violations2 = validate_graph(test_graph)
                    analytics_validation_time = time.perf_counter() - analytics_validation_start
                    grounding_rows.append({
                        "module": module,
                        "checkpoint": cp.name,
                        **grounding,
                        "shacl_conforms_after_analytics": conforms2,
                        "shacl_violations_after_analytics": violations2,
                    })
                    ttl_path = results_dir / "graphs" / f"{module}_day{cp.day}.ttl"
                    ttl_path.parent.mkdir(parents=True, exist_ok=True)
                    test_graph.serialize(destination=str(ttl_path), format="turtle")
                    print(
                        f"[pipeline] {module} {cp.name}: evidence materialized "
                        f"in {evidence_materialization_time:.1f}s; analytics SHACL "
                        f"in {analytics_validation_time:.1f}s",
                        flush=True,
                    )
            checkpoint_time = time.perf_counter() - checkpoint_start
            process_rows.append({
                "module": module,
                "checkpoint": cp.name,
                "raw_feature_time": round(raw_feature_time, 3),
                "graph_update_time": round(graph_update_time, 3),
                "kg_feature_time": round(kg_feature_time, 3),
                "validation_time": round(validation_time, 3),
                "model_train_eval_time": round(model_train_eval_time, 3),
                "evidence_materialization_time": round(evidence_materialization_time, 3),
                "analytics_validation_time": round(analytics_validation_time, 3),
                "checkpoint_time": round(checkpoint_time, 3),
                "kg_feature_engine": kg_feature_engine,
            })
            print(
                f"[pipeline] {module} {cp.name}: checkpoint complete "
                f"in {checkpoint_time:.1f}s",
                flush=True,
            )
        print(
            f"[pipeline] Module {module}: complete in {time.perf_counter() - module_start:.1f}s",
            flush=True,
        )

    write_csv(results_dir / "tables" / "dataset_distribution.csv", dataset_rows)
    write_csv(results_dir / "tables" / "incremental_construction.csv", construction_rows)
    write_csv(results_dir / "tables" / "predictive_performance.csv", performance_rows)
    write_csv(results_dir / "tables" / "explanation_grounding.csv", grounding_rows)
    write_csv(results_dir / "tables" / "feature_parity.csv", parity_rows)
    write_csv(results_dir / "tables" / "experimental_process.csv", process_rows)
    write_csv(results_dir / "tables" / "model_timing.csv", model_timing_rows)
    write_csv(results_dir / "tables" / "training_history.csv", training_history_rows)
    aggregation = aggregate_result_tables(results_dir, results_dir / "tables")
    summary = {
        "modules": modules,
        "config_path": str(config_path),
        "models": model_specs(cfg),
        "aggregation": aggregation,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.perf_counter() - wall_start, 3),
        "tables": [
            "dataset_distribution.csv",
            "incremental_construction.csv",
            "predictive_performance.csv",
            "explanation_grounding.csv",
            "feature_parity.csv",
            "experimental_process.csv",
            "model_timing.csv",
            "training_history.csv",
            "publication_readiness.json",
        ],
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run_log = logs_dir / f"pipeline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    run_log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LDT evidence-grounded early-warning experiment.")
    parser.add_argument("--config", default="config/experiment.yaml")
    args = parser.parse_args()
    summary = run_experiment(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
