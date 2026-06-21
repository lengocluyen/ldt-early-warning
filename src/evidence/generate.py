from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ldt_pipeline import FEATURES, evidence_for_feature


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate textual evidence previews for a feature CSV.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.features)
    rows = []
    for _, row in frame.iterrows():
        for feature in FEATURES:
            evidence = evidence_for_feature(row, feature)
            rows.append({"learner_key": row["learner_key"], "feature": feature, **evidence})
    out = pd.DataFrame(rows)
    if args.output:
        out.to_csv(args.output, index=False)
    print(json.dumps({"rows": len(out), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
