from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rdflib import Graph

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ldt_pipeline import validate_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHACL validation on an RDF graph.")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--shapes", default="ontology/ldt-shapes.ttl")
    args = parser.parse_args()
    graph = Graph().parse(args.graph)
    conforms, violations = validate_graph(graph, args.shapes)
    print(json.dumps({"conforms": conforms, "violations": violations}, indent=2))


if __name__ == "__main__":
    main()
