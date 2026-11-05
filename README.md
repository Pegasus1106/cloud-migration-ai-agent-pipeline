# Cloud Migration AI Agent Pipeline - P1 Prototype (Milestone 2)

This is the Milestone 1 "P1 proposed system" implementation. It was
originally designed as a refinement of a separate stakeholder baseline
pipeline (referred to below as **B0**), but this repository is fully
standalone: it has no runtime or path dependency on that other project.
A static, unmodified copy of the original baseline input
(`data/baseline_raw_sample.csv`) is bundled here for reproducibility.

This project adds inbound-dependency derivation, WCC/SCC/community-aware
wave planning under a configurable transfer-volume limit, a static FinOps
cost/savings estimate, a chat-style constraint interface, and a
deterministic **B1** greedy baseline for comparison.

## Prerequisites

- **Python 3.10+** (the repo was built and tested with 3.11; the system
  default `python3` on macOS is often 3.9, which is too old - see step 1).
- **Docker Desktop** (or another Docker Engine + Compose v2 runtime),
  running before you start the pipeline. No custom image is built here;
  `compose.yml` pulls the public `neo4j:5` image (pinned by digest) plus
  the `apoc` and `graph-data-science` plugins automatically, so nothing
  needs to be shared or built separately for reproducibility.
- **ASU VPN or ASU Wi-Fi**, and an **ASU_RC_API_KEY** from
  https://voyager.rc.asu.edu/ (LLM Access tab -> Create Key). The LLM
  steps call `https://openai.rc.asu.edu/v1`, which requires this.
- Ports **7475** and **7688** free on localhost (offset from the
  baseline's 7474/7687 so both projects can run side by side).

## Steps to reproduce

```bash
# 1. From the workspace root, enter this project.
cd Cloud_Migration_AI_Agent_Pipeline

# 2. Add your API key (never commit this file - it is already gitignored).
cp .env.example .env
#    then edit .env and set: ASU_RC_API_KEY=your_key_here

# 3. Make the run script executable (first time only).
chmod +x run_pipeline_p1.sh

# 4. Run the full pipeline on either dataset. This creates the venv,
#    installs requirements.txt, starts the pinned Neo4j container,
#    runs the Constraint Agent, then all 7 agent stages, then validates
#    the result.
./run_pipeline_p1.sh --dataset stakeholder   # data/sample.csv, 50 servers
./run_pipeline_p1.sh --dataset synthetic     # data/synthetic_83.csv, 83 servers

# Optional: pass a free-text constraint request instead of the default.
./run_pipeline_p1.sh --dataset stakeholder --request "Keep waves under 2500 GB"
```

**Expected output** after a run: `data/<dataset>_p1_plan.csv` (the final
migration plan), `data/<dataset>_p1_wave_summary.json` (wave totals and
any infeasible groups), `data/validation_report_<dataset>_P1.json` (the
validator's pass/fail checks), and `data/run_log_<dataset>.json`
(per-stage exit codes and timing). The script prints the validation
report and these file paths at the end; exit code 0 means every stage
succeeded.

**To inspect the graph:** open http://localhost:7475 in a browser
(login `neo4j` / `password123`).

**To stop and clean up:**
```bash
docker compose down        # stop the container, keep the data volume
docker compose down -v     # stop and delete the data volume
```

**To reproduce the B1 baseline and the comparison tables** (no Docker
needed for B1 itself), run the commands under "Running the baseline and
comparisons" further below, after the P1 run above has produced
`data/cleaned_stakeholder.csv` / `data/cleaned_synthetic.csv`.

## Directory contents

| File | Role |
|---|---|
| `data_agent.py` | Normalizes discovery CSV, validates Transfer_GB / Current_Monthly_Cost / Early_Migration_Requested, writes a data-quality report. |
| `neo4j_agent.py` | Loads servers + outbound edges into Neo4j (this project's own container). |
| `graph_agent.py` | **New.** Derives inbound degree; computes WCC, SCC, and (for oversized WCCs) Louvain communities. |
| `cloud_selection_agent.py`, `sizing_agent.py` | Reused from B0 unchanged (repointed at this project's Neo4j). |
| `finops_agent.py` | **New.** Static AWS pricing catalog -> estimated monthly cost & savings vs. `Current_Monthly_Cost`. |
| `risk_agent.py` | B0's agent with the tool-input bug fixed (defensive parsing) plus a post-hoc verification step that triggers a deterministic fallback if any server is left without a risk score. |
| `constraint_agent.py` | **New.** Chat-style interface: free text -> confirmed `data/run_config.json` (max_transfer_gb, target_cloud, early-migration handling). |
| `wave_agent_p1.py` | **New.** The core contribution: SCC-safe, community-prioritized, capacity-constrained wave packing (see `graph_utils.py`). |
| `graph_utils.py` | Shared packing/condensation helpers used by both P1 and B1. |
| `validator.py` | **New.** Deterministic checks (dependency order, transfer-volume limit, sizing safety, risk completeness) runnable against any plan CSV (B0, B1, or P1). |
| `supervisor.py` | **New.** Runs all P1 stages in order, writes a run log, refuses to plan with an unconfirmed configuration. |
| `baselines/b1_greedy_wave_agent.py` | **New.** Deterministic, LLM-free, community-blind baseline (SCC-granularity first-fit-decreasing packing). |
| `compare_results.py` | Computes comparison metrics (wave count, cross-wave edges, utilization, early-migration placement) across B0/B1/P1. |
| `data/baseline_raw_sample.csv` | A static, unmodified copy of the original baseline's 50-server discovery CSV, bundled here so this repo is self-contained. |
| `data/make_stakeholder_sample.py` | Extends `data/baseline_raw_sample.csv` with Transfer_GB / Current_Monthly_Cost / Early_Migration_Requested (documented assumptions) to produce `data/sample.csv`. |
| `data/generate_synthetic_data.py` | Generates an 83-server benchmark with one deliberately oversized WCC (~20.7k GB) to exercise community splitting. |

## Running the baseline (B1) and comparisons

`./run_pipeline_p1.sh` always writes its cleaned CSV to the generic
`data/cleaned_samples.csv` (whichever dataset you last ran). B1 and the
comparison script instead expect a dataset-labeled cleaned CSV, so
generate those explicitly first (requires the venv from step 4 above to
be active: `source venv/bin/activate`):

```bash
# Produce dataset-labeled cleaned CSVs (reused by both B1 and the
# validator/comparison script so every system sees identical input).
python data_agent.py data/sample.csv data/cleaned_stakeholder.csv
python data_agent.py data/synthetic_83.csv data/cleaned_synthetic.csv

# Run the B1 baseline (deterministic, no LLM, no Neo4j needed).
python baselines/b1_greedy_wave_agent.py --cleaned data/cleaned_stakeholder.csv \
    --max-transfer-gb 3000 --out-plan baselines/stakeholder_b1_plan.csv \
    --out-summary baselines/stakeholder_b1_summary.json

python baselines/b1_greedy_wave_agent.py --cleaned data/cleaned_synthetic.csv \
    --max-transfer-gb 3000 --out-plan baselines/synthetic_b1_plan.csv \
    --out-summary baselines/synthetic_b1_summary.json

# Validate any plan (B0, B1, or P1) against the same rules.
python validator.py --plan <plan.csv> --cleaned <cleaned.csv> --label <name> \
    --max-transfer-gb 3000 --infeasible <wave_summary.json>

# Compare systems side by side (wave count, cross-wave edges, utilization,
# early-migration placement, FinOps/risk-score availability).
python compare_results.py --dataset stakeholder --cleaned data/cleaned_stakeholder.csv \
    --b0-plan baselines/stakeholder_b0_plan.csv \
    --b1-plan baselines/stakeholder_b1_plan.csv \
    --p1-plan data/stakeholder_p1_plan.csv \
    --early-ids SRV-008,SRV-009,SRV-048,SRV-049
```

The `--b0-plan` argument above is optional and only needed if you want the
B0 row in the comparison table. This repository does not depend on B0 or
bundle its code; if you have access to the separate baseline repository
and have run its pipeline, copy its resulting `final_migration_plan.csv`
in under `baselines/stakeholder_b0_plan.csv` manually. Without it,
`compare_results.py` simply omits the B0 row and compares B1 vs. P1.

## What was actually found running this (real, not hypothetical)

- **B0's bug, reproduced and fixed.** B0's `risk_agent.py` failed to write
  `Risk_Score` because the model sometimes passed a Python-repr string
  instead of a JSON list, and the tool rejected it silently. P1's
  `risk_agent.py` parses defensively and adds a post-hoc verification +
  deterministic fallback. During real runs here it caught two *different*
  failures live: a partial-coverage case (40/50 servers returned) and a
  full failure case (0/83 servers returned) - both were detected and
  corrected automatically.
- **A packing-granularity bug found and fixed during this build.** An
  earlier version of `wave_agent_p1.py` treated an entire oversized
  WCC-community as one indivisible packing unit, which made P1 report
  *more* infeasible groups than the community-blind B1 baseline on the
  synthetic dataset (4 oversized ~5,000 GB communities vs. B1's 0, since
  B1 packs at fine SCC granularity). The fix: only true SCCs (cycles) are
  indivisible; `community_id` is used purely as a scheduling-priority
  signal, not a forced grouping. See `wave_agent_p1.py` docstring and
  `_assign_group_ids`.
- **Measured outcome after the fix**, both datasets, 0 dependency and 0
  transfer-volume violations for both B1 and P1 (`comparison_stakeholder.json`,
  `comparison_synthetic.json`, `validation_report_*.json`):
  - Early-migration requests: P1 moves flagged servers into wave 0-1 on
    both datasets; B1 (which ignores the flag) scatters them into waves
    4-6 (stakeholder) / 1-5 (synthetic).
  - FinOps: only P1 produces a cost estimate (stakeholder: $37,533 ->
    $9,984/mo, 73.4% estimated savings; synthetic: $60,114 -> $23,533/mo,
    60.9%). B0 and B1 have no FinOps step.
  - Risk-score completeness: B0 = 0/50 (the bug above); B1 and P1 = 100%.
  - **Negative/neutral result, reported honestly**: the hypothesized
    benefit of community detection - fewer cross-wave edges than B1's
    community-blind packing - was **not** observed on either test
    dataset (cross-wave-edge counts were equal: 30/30 on the stakeholder
    set, 117/117 on the synthetic set). Both datasets' dependency
    structure is dominated by long linear chains, which leaves little
    scheduling freedom for the community-priority tie-break to exploit.
    This is a planned follow-up before the final submission (see
    Milestone 2 report, Sections 5-6): either strengthen the priority
    weighting or redesign the synthetic benchmark with more parallel
    (fan-out) structure within a single business unit.
  - AWS sizing matrix gap (inherited from B0, not yet fixed): 5 servers
    (16+ vCPU / 128+ GB RAM) exceed every catalog instance including the
    default `m5.4xlarge`, flagged by the validator's `sizing_safety`
    check on all three systems. Needs a larger instance tier added to
    `data/aws_pricing_catalog.json`.

See `Limitations.md` for the full list of known limitations, their root
causes, and planned fixes before the final submission.

## Known assumptions

- `Transfer_GB` is approximated as `Storage_GB` (no live discovery feed).
- `Current_Monthly_Cost` is not present in the original baseline CSV and
  is synthesized with a documented formula (see
  `data/make_stakeholder_sample.py`); it is held constant across B0/B1/P1
  so FinOps comparisons are apples-to-apples, but it is not a real bill.
- AWS pricing is a static snapshot (`data/aws_pricing_catalog.json`,
  captured 2026-10-04 from instances.vantage.sh, us-east-1 On-Demand
  Linux), not a live quote.
- Same-wave placement of a server and its direct predecessor is treated
  as satisfying dependency order (not requiring strictly-earlier).
