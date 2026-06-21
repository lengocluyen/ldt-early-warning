from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ldt_pipeline import load_oulad, preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess OULAD tables for the LDT pipeline.")
    parser.add_argument("--input", default="Oulab")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_oulad(args.input)
    processed = preprocess(raw)
    for name, frame in processed.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    summary = {
        "input": str(args.input),
        "output": str(output),
        "rows": {name: int(len(frame)) for name, frame in processed.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
