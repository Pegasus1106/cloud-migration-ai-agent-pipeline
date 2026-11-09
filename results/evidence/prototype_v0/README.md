# Prototype (v0) failure evidence

Logs captured on 2026-11-08 while running the team's pre-Milestone-2 prototype
(commit `d8ef245`, the per-stage `*_agent.py` scripts) before it was replaced by
`p1/`. They are the observed evidence for the failure analysis (template Section 5).
ANSI colour codes and library deprecation noise were stripped; nothing else was edited.

| # | Failure | Where | Evidence | Observed | Fixed in P1 by |
|---|---|---|---|---|---|
| 1 | Pipeline crashes without an LLM key, yet reports "Pipeline stages complete" | tool calling / error handling | `original_run_stakeholder.log` | `cloud_selection_agent` fallback itself calls the LLM (401) → stage exit 1 → supervisor still exits 0 → script prints green "complete", then fails on a missing file | Every agent has a deterministic fallback (`--no-llm` runs end to end); supervisor/CLI return non-zero on failure |
| 2 | Risk Agent invents labels | retrieval + tool calling | `ai_risk_full.log` | The LLM-written Cypher returned OS and tags but **no server IDs**; the model then assigned High/Low to `SRV-001…SRV-040` by guesswork. Compared with the stakeholder's own rule, **22 of the 40 labels it wrote were wrong (55%)**, e.g. `SRV-004 pay-app-01` (Windows 2008, PCI-DSS) → "Low" | Scores computed by code (`p1/core/risk.py`); the LLM only explains them and every explanation is verified (`p1/core/verify.py`) |
| 3 | Risk Agent writes to servers that do not exist, and reports success | state management / verification | `ai_run_synthetic.log` (risk_agent stage) | On the synthetic set (IDs `SYN-…`) the model sent labels for `SRV-001…SRV-015`, copied from its prompt example. The tool printed "Total updated: 15" although **0 rows matched**; only the post-hoc null check rescued the run | Tools validate IDs and report what was actually written; repair/explanation tools reject unknown IDs |
| 4 | Cloud decision made on truncated data | retrieval | `ai_cloud_full.log` | `GraphCypherQAChain` default `top_k=10` returned 10 of 17 OS types (39 of 50 servers); the model also miscounted (said 21 Linux vs 16 Windows; true 30 vs 20). AWS was chosen correctly by luck | Target cloud comes from the user-confirmed configuration; the Q&A agent reports truncation and never silently cuts rows |
| 5 | Same input, different plan (30 vs 33 waves) | planning (determinism) | `determinism_rule_only_run.log` + run logs in the session | Tie-breaks used Neo4j-internal component IDs sorted as text, so plans depended on database history | Components relabelled by smallest Server_ID; Louvain single-threaded; `tests/test_scheduling.py::test_plans_do_not_depend_on_input_order` |
| 6 | Every dependency split across waves | planning | prototype plans | A group became "ready" only after its predecessors were in an *earlier* wave, so all non-cycle dependencies were cut (30/30 on D1, 117/117 on D2) and D2 needed 30–33 waves | `pack_waves(allow_same_wave=True)`; ablation `P1-strict` reproduces the old behaviour |
