from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence.policy import should_present_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Select compact presented evidence from generated evidence rows.")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    evidence = pd.read_csv(args.evidence)
    selected = []
    for learner, group in evidence.groupby("learner_key"):
        group = group.sort_values("strength", ascending=False)
        used_types = set()
        rows = []
        for _, row in group.iterrows():
            if should_present_evidence(row["type"], used_types, len(rows)):
                rows.append(row)
                used_types.add(row["type"])
            if len(rows) >= args.max_items:
                break
        selected.extend(rows)
    out = pd.DataFrame(selected)
    if args.output:
        out.to_csv(args.output, index=False)
    print(json.dumps({"rows": len(out), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
