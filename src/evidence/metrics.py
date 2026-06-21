from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize explanation-grounding result rows.")
    parser.add_argument("--input", default="results/tables/explanation_grounding.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    coverage_cols = [col for col in frame.columns if col.startswith("coverage_")]
    numeric = coverage_cols + ["fidelity_3", "traceability", "avg_evidence_per_alert"]
    numeric = [col for col in numeric if col in frame.columns]
    summary = frame[numeric].mean(numeric_only=True).to_dict() if not frame.empty else {}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
