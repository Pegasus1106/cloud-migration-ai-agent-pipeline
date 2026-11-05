#!/bin/bash
# Cloud Migration AI Agent Pipeline - P1 (Milestone 2 prototype)
#
# Usage:
#   ./run_pipeline_p1.sh --dataset stakeholder   # data/sample.csv
#   ./run_pipeline_p1.sh --dataset synthetic     # data/synthetic_83.csv
#   ./run_pipeline_p1.sh --dataset stakeholder --request "Keep waves under 2500 GB"
#
# What it does:
#   1. Creates/activates venv, installs requirements.
#   2. Starts this project's own Neo4j container (ports 7475/7688, separate
#      from the baseline's 7474/7687 so both can run side by side).
#   3. Runs the Constraint Agent (chat-style interface) to produce a
#      confirmed run_config.json (auto-confirmed here for reproducibility;
#      drop --yes in constraint_agent.py to confirm interactively instead).
#   4. Runs the Supervisor, which executes data -> graph -> cloud -> sizing
#      -> finops -> risk -> wave agents in order and writes a run log.
#   5. Copies the final plan to a dataset-labeled file and runs the
#      validator against it.

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

DATASET="stakeholder"
REQUEST="Keep each migration wave under 3000 GB of transfer volume, target AWS, and move any server flagged Early_Migration_Requested as early as dependency order allows."

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --request) REQUEST="$2"; shift ;;
        -h|--help)
            echo "Usage: ./run_pipeline_p1.sh --dataset [stakeholder|synthetic] [--request \"text\"]"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$DATASET" == "stakeholder" ]; then
    INPUT_CSV="data/sample.csv"
elif [ "$DATASET" == "synthetic" ]; then
    INPUT_CSV="data/synthetic_83.csv"
else
    echo "Unknown --dataset '$DATASET'. Use 'stakeholder' or 'synthetic'."
    exit 1
fi

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}  Cloud Migration AI Agent Pipeline - P1 prototype    ${NC}"
echo -e "${BLUE}  Dataset: $DATASET ($INPUT_CSV)                      ${NC}"
echo -e "${BLUE}======================================================${NC}\n"

echo -e "${YELLOW}[1/5] Setting up Python virtual environment...${NC}"
PYTHON_BIN="python3"
if command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
elif command -v /opt/homebrew/bin/python3.11 &> /dev/null; then
    PYTHON_BIN="/opt/homebrew/bin/python3.11"
fi
if [ ! -d "venv" ]; then
    "$PYTHON_BIN" -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
echo -e "${GREEN}Dependencies installed.${NC}\n"

echo -e "${YELLOW}[2/5] Starting this project's Neo4j (ports 7475/7688)...${NC}"
docker compose up -d
for i in {1..30}; do
    if python3 -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://localhost:7688', auth=('neo4j','password123'))
d.verify_connectivity()
" &> /dev/null; then
        echo -e "${GREEN}Neo4j is ready!${NC}\n"
        break
    fi
    sleep 2
    if [ "$i" -eq 30 ]; then
        echo "Error: Neo4j did not start in time."
        exit 1
    fi
done

echo -e "${YELLOW}[3/5] Running Constraint Agent...${NC}"
python constraint_agent.py "$REQUEST" --yes
echo -e "${GREEN}Configuration confirmed.${NC}\n"

echo -e "${YELLOW}[4/5] Running Supervisor (data -> graph -> cloud -> sizing -> finops -> risk -> wave)...${NC}"
python supervisor.py --input "$INPUT_CSV" --run-id "$DATASET" --clear-graph
echo -e "${GREEN}Pipeline stages complete.${NC}\n"

echo -e "${YELLOW}[5/5] Saving labeled outputs and validating...${NC}"
cp data/final_migration_plan.csv "data/${DATASET}_p1_plan.csv"
cp data/wave_summary.json "data/${DATASET}_p1_wave_summary.json"
MAX_GB=$(python3 -c "import json; print(json.load(open('data/run_config.json'))['max_transfer_gb'])")
python validator.py --plan "data/${DATASET}_p1_plan.csv" --cleaned "data/cleaned_samples.csv" \
    --label "${DATASET}_P1" --max-transfer-gb "$MAX_GB" \
    --infeasible "data/${DATASET}_p1_wave_summary.json"

echo -e "\n${BLUE}======================================================${NC}"
echo -e "${GREEN}P1 pipeline execution complete for dataset: $DATASET${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "Final plan   : data/${DATASET}_p1_plan.csv"
echo -e "Wave summary : data/${DATASET}_p1_wave_summary.json"
echo -e "Validation   : data/validation_report_${DATASET}_P1.json"
echo -e "Neo4j Browser: http://localhost:7475 (neo4j / password123)"
echo -e "\nTo stop Neo4j: docker compose down"
