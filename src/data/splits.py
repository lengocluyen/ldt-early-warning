from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ldt_pipeline import cohort, eligible_modules, load_oulad, preprocess, split_presentations


def main() -> None:
    parser = argparse.ArgumentParser(description="Build presentation-based temporal OULAD splits.")
    parser.add_argument("--input", default="Oulab")
    parser.add_argument("--output", default="data/splits")
    parser.add_argument("--modules", nargs="*", default=None)
    args = parser.parse_args()
    tables = preprocess(load_oulad(args.input))
    modules = args.modules or eligible_modules(tables)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    split_json = {}
    for module in modules:
        train, test = split_presentations(tables, module)
        if not train or not test:
            continue
        train_cohort = cohort(tables, module, train)
        test_cohort = cohort(tables, module, test)
        split_json[module] = {"train": train, "test": test}
        rows.append({
            "module": module,
            "train_presentations": ",".join(train),
            "test_presentations": ",".join(test),
            "train_learners": len(train_cohort),
            "test_learners": len(test_cohort),
            "train_at_risk": int(train_cohort["label"].sum()),
            "test_at_risk": int(test_cohort["label"].sum()),
            "train_not_at_risk": int((1 - train_cohort["label"]).sum()),
            "test_not_at_risk": int((1 - test_cohort["label"]).sum()),
        })
    pd.DataFrame(rows).to_csv(output / "splits.csv", index=False)
    (output / "splits.json").write_text(json.dumps(split_json, indent=2), encoding="utf-8")
    print(json.dumps({"modules": list(split_json), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
