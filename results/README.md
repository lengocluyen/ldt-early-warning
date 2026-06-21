# Generated Results

This directory is the local destination for experiment outputs. It is kept out
of version control because it can contain multi-gigabyte RDF graphs, learner
identifiers inherited from the source data, and machine-specific logs.

`pipeline.svg` is a curated static pipeline diagram intentionally versioned for
the repository README. All other result artifacts remain local-only.

Run an experiment to populate it:

```bash
./run_pipeline.sh --preset smoke
```

The main generated subdirectories are:

```text
tables/   Aggregated CSV and JSON results
figures/  Generated PNG figures
graphs/   RDF graphs with alerts, features, evidence, and raw-record links
logs/     Timestamped run logs
```

Use `python experiments/visualize_results.py --results results --output results/figures`
to regenerate visualizations from existing tables.
