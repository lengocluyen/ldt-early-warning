from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rdflib import Graph

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an RDF graph between serializations.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="turtle")
    args = parser.parse_args()
    graph = Graph().parse(args.input)
    graph.serialize(destination=args.output, format=args.format)
    print(args.output)


if __name__ == "__main__":
    main()
