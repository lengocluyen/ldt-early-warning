from __future__ import annotations

from pathlib import Path
import unittest

import yaml
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery


ROOT = Path(__file__).resolve().parents[1]


class PublicArtifactValidationTests(unittest.TestCase):
    def test_ontology_and_shapes_parse_as_turtle(self) -> None:
        for filename in ("ontology/ldt.ttl", "ontology/ldt-shapes.ttl"):
            graph = Graph()
            graph.parse(ROOT / filename, format="turtle")
            self.assertGreater(len(graph), 0, filename)

    def test_feature_queries_compile(self) -> None:
        feature_dir = ROOT / "sparql" / "features"
        queries = sorted(feature_dir.glob("*.rq"))
        self.assertGreater(len(queries), 0)
        for query_path in queries:
            with self.subTest(query=query_path.name):
                prepareQuery(query_path.read_text(encoding="utf-8"))

    def test_experiment_configurations_are_well_formed(self) -> None:
        required = {"data_dir", "results_dir", "checkpoints", "top_k", "random_state"}
        config_paths = sorted((ROOT / "config").glob("experiment*.yaml"))
        self.assertGreater(len(config_paths), 0)
        for config_path in config_paths:
            with self.subTest(config=config_path.name):
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                self.assertIsInstance(config, dict)
                self.assertTrue(required.issubset(config), config_path.name)
                self.assertTrue(config["checkpoints"])

    def test_public_release_files_are_present(self) -> None:
        for filename in ("README.md", "LICENSE", "CITATION.cff", "CONTRIBUTING.md"):
            self.assertTrue((ROOT / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
