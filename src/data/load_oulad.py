from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ldt_pipeline import load_oulad


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect available OULAD CSV tables.")
    parser.add_argument("--input", default="Oulab")
    args = parser.parse_args()
    tables = load_oulad(args.input)
    summary = {
        name: {"rows": int(len(df)), "columns": list(df.columns)}
        for name, df in tables.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
