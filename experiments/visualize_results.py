from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.results.visualize import visualize_result_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from experiment result CSVs.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print(json.dumps(visualize_result_tables(args.results, args.output), indent=2))


if __name__ == "__main__":
    main()
