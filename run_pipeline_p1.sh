#!/bin/bash
# Cloud Migration AI Agent Pipeline - P1 (Milestone 2)
#
# Usage:
#   ./run_pipeline_p1.sh --dataset stakeholder                     # data/sample.csv (D1)
#   ./run_pipeline_p1.sh --dataset synthetic                       # data/synthetic_83.csv (D2)
#   ./run_pipeline_p1.sh --dataset bench100                        # data/benchmark/D3_100.csv (D3)
#   ./run_pipeline_p1.sh --dataset stakeholder --request "Keep waves under 2.5 TB and move CRM early"
#   ./run_pipeline_p1.sh --dataset stakeholder --no-llm            # rule-only mode, no API key needed
#   ./run_pipeline_p1.sh --ui                                      # start the Streamlit web UI
#
# Steps: venv + pinned requirements -> Neo4j (ports 7475/7688) -> Constraint Agent ->
# supervisor (graph, sizing, FinOps, risk, waves, validation, repair, review, explanations) -> outputs.
# Exit code: 0 = plan passed validation, 2 = blocked/rejected, 1 = error.

set -euo pipefail
cd "$(dirname "$0")"

DATASET="stakeholder"
REQUEST="Keep each migration wave under 3000 GB of transfer volume, target AWS, and move any server flagged for early migration as early as dependency order allows."
EXTRA=()
UI=0
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --request) REQUEST="$2"; shift ;;
        --no-llm|--no-repair|--strict-waves|--interactive|--no-explain) EXTRA+=("$1") ;;
        --ui) UI=1 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

case "$DATASET" in
    stakeholder) INPUT_CSV="data/sample.csv" ;;
    synthetic)   INPUT_CSV="data/synthetic_83.csv" ;;
    bench30)     INPUT_CSV="data/benchmark/D3_30.csv" ;;
    bench100)    INPUT_CSV="data/benchmark/D3_100.csv" ;;
    bench500)    INPUT_CSV="data/benchmark/D3_500.csv" ;;
    *) INPUT_CSV="$DATASET" ;;
esac

echo "[1/4] Python environment"
PYTHON_BIN=$(command -v python3.11 || command -v python3)
[ -d venv ] || "$PYTHON_BIN" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install -q -r requirements.txt

echo "[2/4] Neo4j (ports 7475/7688)"
if docker compose version >/dev/null 2>&1; then COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then COMPOSE=(docker-compose)
else COMPOSE=(); echo "  Docker Compose not found: the planner will use its networkx graph fallback."; fi
if [ ${#COMPOSE[@]} -gt 0 ]; then
    "${COMPOSE[@]}" up -d
    for i in {1..60}; do
        if python -c "from p1.core.graphdb import GraphStore; import sys; sys.exit(0 if GraphStore().ping() else 1)" 2>/dev/null; then
            echo "  Neo4j is ready"; break; fi
        sleep 2
        [ "$i" -eq 60 ] && echo "  Neo4j did not start in time; continuing with the networkx fallback."
    done
fi

if [ "$UI" -eq 1 ]; then
    echo "[3/4] Starting the web UI at http://localhost:8501"
    exec streamlit run app.py
fi

echo "[3/4] Constraint Agent + supervisor on $INPUT_CSV"
set +e
if [[ " ${EXTRA[*]-} " == *" --no-llm "* ]]; then
    python run_p1.py --dataset "$INPUT_CSV" --yes "${EXTRA[@]}"
else
    python run_p1.py --dataset "$INPUT_CSV" --request "$REQUEST" --yes "${EXTRA[@]+"${EXTRA[@]}"}"
fi
CODE=$?
set -e

echo "[4/4] Done (exit code $CODE). Outputs are in runs/<run_id>/ ; Neo4j Browser: http://localhost:7475"
exit $CODE
