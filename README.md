# Cloud Migration AI Agent Pipeline — P1 (Milestone 2)

An agentic decision-support system that turns server-discovery data into a validated,
explained cloud-migration wave plan. It refines the stakeholder's baseline
([ananthsekar007/migration-planning-ai](https://github.com/ananthsekar007/migration-planning-ai), "B0")
with the requirements agreed in Milestone 1: derived inbound dependencies, a 3,000 GB
per-wave transfer limit, early-migration preferences, community-aware grouping, a static
FinOps comparison, and a chat-style constraint interface with user confirmation.

**Design rule:** the LLM interprets intent, chooses bounded actions, and explains; deterministic
tools compute graph structure, sizes, prices, risk, waves, and validation. Every LLM output is
validated by a tool or verified against tool output. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
chat ──► Constraint Agent ──► user confirms ──► Supervisor (LangGraph)
          discover → graph (Neo4j GDS) → size & price → risk → waves → VALIDATE
              VALIDATE ──repairable──► Repair Agent ──► re-price ──► VALIDATE
              VALIDATE ──needs a person──► review drafting ──► human review ──► VALIDATE / re-plan / stop
              VALIDATE ──passed──► explainers (verified) ──► export ──► user approves
          + “Ask the plan” Q&A agent (read-only Text2Cypher)
```

## Quick start

Prerequisites: Python 3.10+ (3.11 used), Docker with Compose v2 (Docker Desktop), and for the
LLM agents an ASU Voyager key on ASU Wi-Fi or the ASU VPN. Without a key everything still runs
in rule-only mode (`--no-llm`); without Docker the graph stage falls back to networkx.

```bash
cp .env.example .env          # set ASU_RC_API_KEY=...  (never commit .env)
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
docker compose up -d          # Neo4j 5 + GDS + APOC on ports 7475 (browser) / 7688 (bolt)

streamlit run app.py          # web UI at http://localhost:8501
```

Or from the terminal:

```bash
# Chat request -> confirmation -> plan (auto-confirm and batch review policy for scripting)
python run_p1.py --dataset data/sample.csv \
  --request "Keep each wave under 2.5 TB, move the CRM app early, and start with a pilot wave" --yes

# Fully interactive: the Constraint Agent may ask questions; you confirm and decide review items
python run_p1.py --dataset data/sample.csv --interactive

# Ablations
python run_p1.py --dataset data/sample.csv --yes --no-llm          # every agent uses its rule fallback
python run_p1.py --dataset data/sample.csv --yes --no-repair       # no Repair Agent
python run_p1.py --dataset data/sample.csv --yes --strict-waves    # prototype packing

./run_pipeline_p1.sh --dataset stakeholder   # one command: venv, Neo4j, agents, plan (exit 0 = valid plan)
./run_pipeline_p1.sh --ui                    # same setup, then the web UI
```

Each run writes `runs/<run_id>/`: `plan.csv`, `wave_summary.json` (with verified wave
explanations), `validation_report.json`, `graph_summary.json` (inbound/outbound summary, cycles,
oversized components), `finops_summary.json`, `quality_report.json`, `review_log.json`,
`repair_log.json`, `explanations.md`, `run_log.json` (config, model, prompt versions, catalog and
lifecycle versions, timings, metrics), and `trace.jsonl` (every node, LLM call, tool call,
verification, and fallback). Neo4j Browser: http://localhost:7475 (`neo4j` / `password123`, local only).

## Results at a glance (Milestone 2)

Full tables, ablations, and failure analysis: [docs/MILESTONE2_RESULTS.md](docs/MILESTONE2_RESULTS.md).

- **Hard criteria** (0 dependency violations, no unreported wave above 3,000 GB, no undersized or non-AWS
  instance, every feasible early request on time): P1 met all of them on D1, D2, D3_30, and D3_100; B0 and B1
  met them on none of the four.
- **Early-migration requests on time:** P1 100% on every dataset; B1 0 to 25%; B0 has no early handling.
- **Dependencies split across waves:** P1 lower than B1 on D2, D3_30, and D3_100, tied on D1.
- **B0 (stakeholder baseline, run unmodified):** chose Azure instead of AWS on D2/D3 and wrote no risk labels there.
- **Constraint Agent:** 96.8% of 31 labeled requests (Milestone 1 target 85%).
- **Ask-the-plan:** 92.6% correct rows and 0% query errors, vs 77.8% and 7.4% for the B0-style GraphCypherQAChain.
- **Reliability:** 12/12 P1 runs on 4 datasets completed with valid, identical plans per dataset.
- **Cost:** a P1 run takes minutes (about 99% waiting for the LLM); B1 runs in milliseconds.

## Reproducing the Milestone 2 evaluation

```bash
python -m pytest -q tests                                   # 42 unit/integration tests
python data/benchmark/generate_benchmark.py                 # rebuild D3 from cached gpt-oss-120b output
python eval/run_eval.py --datasets D1,D2,D3_30,D3_100 --p1-repeats 3 \
       --ablation-datasets D1,D2 --b0-repo ../migration-planning-ai      # what Milestone 2 reports
python eval/eval_constraints.py                             # constraint parsing (31 labeled requests)
python eval/eval_qa.py                                      # Ask-the-plan vs B0-style GraphCypherQAChain
python eval/make_figures.py                                 # results/figures/*.png
python3 eval/progress.py --watch                            # live progress bar for a running evaluation

# Planned for the final submission (not yet run): D3_500, 10+ repeats, larger-model ablation beyond D1
python eval/run_eval.py --datasets D3_500 --p1-repeats 3 --ablations ""
python eval/run_eval.py --datasets D1,D2 --ablation-model qwen3-235b-a22b
```

B0 is run unmodified from a local clone of the stakeholder repository by
`baselines/run_b0.py` (it swaps the input file, runs `./run_pipeline.sh --clean --run`, and
restores the original file). Results, tables, plans, and figures are in [`results/`](results/);
the write-up is [docs/MILESTONE2_RESULTS.md](docs/MILESTONE2_RESULTS.md).

### Datasets

| ID | File | Servers | What it tests |
|---|---|---|---|
| D1 | `data/sample.csv` | 50 | Stakeholder sample (B0 input + synthesized `Transfer_GB`, `Current_Monthly_Cost`, early flags; see `data/make_stakeholder_sample.py`); cycles, 45 TB indivisible warehouse, legacy OS, missing data |
| D2 | `data/synthetic_83.csv` | 83 | Long dependency chains + one oversized 30-server data platform (`data/generate_synthetic_data.py`) |
| D3 | `data/benchmark/D3_{30,100,500}.csv` | 30 / 100 / 500 | Portfolio written by `gpt-oss-120b`, structure and labels by seeded code: shared AD/DNS, shared BU databases (fan-in), replication and queue cycles, data platforms larger than the sizing matrix, regional/DR instances, 3% incomplete records, ground truth in `D3_*_truth.json` |

Agent benchmarks: `data/benchmark/constraint_prompts.jsonl` (31 requests, 7 deliberately ambiguous;
text by `gpt-oss-120b`, labels by code) and `data/benchmark/qa_questions.jsonl` (27 questions across
12 templates; gold answers computed from the plan at evaluation time).

## Repository layout

| Path | Contents |
|---|---|
| `p1/core/` | Deterministic tools: data contract (`data.py`), Neo4j/GDS store (`graphdb.py`), networkx fallback, catalog & sizing, FinOps, risk scoring, scheduler, planning policies (P1 and B1), validator, verifiers, Cypher guard |
| `p1/agents/` | Constraint, Repair, Review-drafting, Risk-explainer, Wave-explainer, Q&A agents |
| `p1/supervisor.py` | LangGraph state machine, routing, human-review interrupt, auto-review policy |
| `p1/export.py`, `p1/tracing.py`, `p1/state.py` | Outputs, audit trail, typed config/state |
| `app.py`, `run_p1.py`, `run_pipeline_p1.sh` | Web UI, CLI, one-command runner |
| `baselines/` | `b1_greedy_wave_agent.py` (B1), `run_b0.py` (runs the stakeholder repo unmodified) |
| `eval/` | Evaluation harness, agent benchmarks, figures |
| `data/` | Datasets, AWS catalog (`aws_pricing_catalog.json`), OS lifecycle table (`os_lifecycle.json`), D3 generator |
| `tests/` | pytest suite (data contract, inbound derivation, scheduling, FinOps arithmetic, risk, validator, agent guardrails, end-to-end interrupt/resume) |
| `results/` | Evaluation outputs, plans, figures, prototype failure evidence |

The pre-Milestone-2 prototype scripts (`*_agent.py`, `supervisor.py`, `graph_utils.py`) were
replaced by `p1/`; they remain in git history at commit `d8ef245`.

## Assumptions and limits

- `Transfer_GB` defaults to `Storage_GB`; `Current_Monthly_Cost` in D1–D3 is synthesized with a
  documented formula, so savings percentages are illustrative, not real bills.
- AWS prices are a static us-east-1 Linux On-Demand snapshot, compute only; Windows Server and
  SQL Server license costs are not included and are flagged per server.
- Dependency direction: an edge A → B ("A calls B") requires wave(A) ≤ wave(B); same wave is allowed.
- Neo4j holds one plan at a time; runs are serialized with a file lock.

See [Limitations.md](Limitations.md) for the full list.
