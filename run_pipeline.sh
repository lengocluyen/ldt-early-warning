#!/usr/bin/env bash
set -euo pipefail

PRESET="full_run"
SKIP_INSTALL=0
NO_VENV=0

usage() {
  cat <<'EOF'
Usage: ./run_pipeline.sh [--preset NAME] [--skip-install] [--no-venv]

Presets:
  smoke             DDD Week 4 only; fastest end-to-end sanity check
  full_run          all modules with LR/XGB/XGB-LDT (default)
  publication_full  expensive all-modules modern benchmark with XGB grid search
  core              LR-raw, XGB-raw, XGB-LDT
  tab_transformer   core + TABTX
  sklearn_modern    core + RF, ET, HGB, TABTX
  modern            core + RF, ET, HGB, LightGBM, CatBoost, TABTX
  all_modules       alias of full_run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset|-p)
      if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "--preset requires a preset name." >&2
        usage
        exit 2
      fi
      PRESET="${2:-}"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --no-venv)
      NO_VENV=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$PRESET" in
  smoke)
    CONFIG="config/experiment_smoke.yaml"
    REQUIREMENTS="requirements-core.txt"
    ;;
  core)
    CONFIG="config/experiment.yaml"
    REQUIREMENTS="requirements-core.txt"
    ;;
  tab_transformer)
    CONFIG="config/experiment_tab_transformer.yaml"
    REQUIREMENTS="requirements-tab-transformer.txt"
    ;;
  modern)
    CONFIG="config/experiment_modern_models.yaml"
    REQUIREMENTS="requirements.txt"
    ;;
  sklearn_modern)
    CONFIG="config/experiment_sklearn_modern.yaml"
    REQUIREMENTS="requirements-tab-transformer.txt"
    ;;
  all_modules|publication|full_run)
    CONFIG="config/experiment_all_modules.yaml"
    REQUIREMENTS="requirements-core.txt"
    ;;
  publication_full)
    CONFIG="config/experiment_publication_full.yaml"
    REQUIREMENTS="requirements.txt"
    ;;
  *)
    echo "Unknown preset: $PRESET" >&2
    usage
    exit 2
    ;;
esac

case "$PRESET" in
  publication_full)
    if [[ "${CONFIRM_LONG_RUN:-0}" != "1" ]]; then
      cat >&2 <<'EOF'
Preset 'publication_full' is intentionally expensive. Your last DDD modern run took about 5 days,
and this preset can take substantially longer because it runs all modules with modern models and
XGBoost grid search.

To run it anyway:
  CONFIRM_LONG_RUN=1 ./run_pipeline.sh --preset publication_full

For the normal paper tables, use the default:
  ./run_pipeline.sh
EOF
      exit 2
    fi
    ;;
esac

cd "$(dirname "$0")"

mkdir -p results/logs
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${PRESET}"
LOG_FILE="results/logs/${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Run ID: $RUN_ID"
echo "Log file: $LOG_FILE"
echo "Started at UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Preset: $PRESET"
echo "Config: $CONFIG"
echo "Requirements: $REQUIREMENTS"

find_python() {
  if [[ "$NO_VENV" -eq 0 && -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "Python was not found. Install Python 3.10+ and re-run this command." >&2
  return 1
}

if [[ "$NO_VENV" -eq 0 && ! -x ".venv/bin/python" ]]; then
  SYSTEM_PYTHON="$(find_python)"
  echo "Creating virtual environment in .venv ..."
  "$SYSTEM_PYTHON" -m venv .venv
fi

PYTHON="$(find_python)"

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo "Installing requirements ..."
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r "$REQUIREMENTS"
fi

echo "Running preset '$PRESET' with $CONFIG ..."
"$PYTHON" experiments/run_all.py --config "$CONFIG"

echo "Aggregating result tables ..."
"$PYTHON" experiments/aggregate_results.py --results results --output results/tables

echo "Generating result figures ..."
"$PYTHON" experiments/visualize_results.py --results results --output results/figures

echo "Finished at UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cat <<'EOF'

Done. Main outputs:
  results/tables/dataset_distribution.csv
  results/tables/incremental_construction.csv
  results/tables/predictive_performance.csv
  results/tables/explanation_grounding.csv
  results/tables/feature_parity.csv
  results/tables/experimental_process.csv
  results/tables/model_timing.csv
  results/tables/training_history.csv
  results/tables/publication_readiness.json
  results/figures/
  results/logs/
EOF
