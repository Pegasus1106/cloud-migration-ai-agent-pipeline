# Limitations and Scope for Improvement

This document lists the limitations found while building and running the
P1 prototype and B1 baseline (see `README.md` for what was built and the
measured results). Each item distinguishes what was actually observed
from what is a planned fix, per the Milestone 2 reporting standard.

## 1. Community detection's claimed benefit is not yet demonstrated

**Observed:** On both the stakeholder dataset (50 servers) and the
synthetic dataset (83 servers), P1's cross-wave-edge count was identical
to B1's community-blind packing (30/30 and 117/117 respectively; see
`comparison_stakeholder.json`, `comparison_synthetic.json`). The core
technical claim from Milestone 1 - that community-aware grouping reduces
cross-wave dependency cuts versus naive packing - has not been
empirically shown.

**Likely cause:** Both test datasets are dominated by long linear
dependency chains (`build_chain` in `data/generate_synthetic_data.py`),
which leaves little scheduling freedom at each step. When almost every
server has exactly one ready predecessor and one ready successor, the
priority tie-break (which favors same-community continuity) rarely gets
a chance to choose between multiple valid candidates.

**Planned fix:**
- Redesign the synthetic benchmark to include a wider, more parallel
  (fan-out) structure within one oversized business unit - e.g. multiple
  independent sibling sub-clusters that can be scheduled in different
  orders - so the packer actually has choices to make.
- Alternatively, or in addition, strengthen the priority function so
  community continuity is weighted before raw size, not only as a late
  tie-break (currently `(early_first, risk_rank, community_id, -savings,
  group_id)` in `wave_agent_p1.py`).
- Re-run the comparison and report the corrected numbers before claiming
  the benefit in the final submission.

## 2. AWS sizing matrix has a coverage gap (inherited from B0)

**Observed:** The validator's `sizing_safety` check flags 5 servers on
the stakeholder dataset (SRV-018, SRV-019, SRV-040, SRV-041, SRV-042 -
needing 16-32 vCPU and 128-256 GB RAM) as undersized on every system
(B0, B1, and P1), because the largest catalog instance,
`m5.4xlarge`, only offers 16 vCPU / 64 GB RAM. This bug existed in the
original B0 sizing matrix and was carried over unchanged into P1 and B1.

**Planned fix:** Add larger instance tiers (e.g. `m5.8xlarge`,
`m5.12xlarge`, `m5.16xlarge`) to `sizing_agent.py`'s `AWS_SIZING` matrix
and `data/aws_pricing_catalog.json`, then re-run FinOps and the
validator to confirm the violation count drops to zero.

## 3. FinOps estimate rests on a synthetic cost input

**Observed:** `Current_Monthly_Cost` is not present in the original
discovery CSV. It is synthesized with a documented formula (`40*vCPU +
10*RAM_GB + 0.05*Storage_GB + 50`, see `data/make_stakeholder_sample.py`)
rather than a real on-prem bill. The reported savings (73.4% and 60.9%
on the two datasets) are only as credible as that formula.

**Planned fix:** Replace the synthetic formula with a stakeholder-
supplied or industry-benchmark on-prem TCO figure before the final
submission, and re-state the savings numbers. Keep the AWS side of the
estimate (static catalog, dated, sourced) as-is since it is already
labeled as an estimate, not a live quote.

## 4. Risk Agent's LLM path is unreliable; the system depends on a safety net

**Observed:** Across the runs performed for this milestone, the LLM-
driven risk classification failed in two different ways: once it
returned risk updates for only 40 of 50 servers (partial coverage), and
once it returned none of 83 servers (total failure). Both were caught by
a post-hoc verification query (`MATCH (s:Server) WHERE s.risk_score IS
NULL`) and corrected by a deterministic rule-based fallback. The system
produced a correct final result in both cases, but only because of the
safety net - the LLM step itself was wrong both times it was tested.

**Planned fix:** Investigate why the model under-reports server counts
(possibly output-length truncation or the agent loop ending early);
consider chunking the server list into smaller batches per tool call, or
making the deterministic classification the primary path and using the
LLM only to generate the natural-language explanation (removing the
unreliable step from the critical path entirely).

## 5. Constraint Agent is not yet evaluated at scale

**Observed:** `constraint_agent.py` was tested manually against a
handful of free-text requests and correctly extracted `max_transfer_gb`,
`target_cloud`, and `honor_early_migration_flags` each time. It has not
been tested against the ~20 labeled prompts with a measured accuracy
target that Milestone 1 specified, and it has not been tested against
ambiguous, contradictory, or adversarial phrasing.

**Planned fix:** Build the 20-prompt labeled test set (varied phrasing,
some with no numeric limit stated, some with conflicting values) and
measure field-extraction accuracy before the final submission, per the
Milestone 1 evaluation plan.

## 6. Evaluation is a single-pass pilot, not a repeated-trial study

**Observed:** Each dataset was run once through B0 (two runs total,
captured earlier), B1, and P1. Milestone 1 called for repeating
nondeterministic runs ~20 times to measure completion reliability and
output variance; that has not been done. The two real LLM failures
documented above were found incidentally during these single runs, not
through a systematic reliability study.

**Planned fix:** Script repeated runs (e.g. 10-20x per dataset per
system) capturing completion rate, latency, and tool-call error rate,
and report variance rather than a single observation.

## 7. No stakeholder review of the P1 output yet

**Observed:** The "Human usefulness" and "Explainability" metrics from
the Milestone 1 evaluation plan require the stakeholder to blind-review
three plans per system; this has not happened. All current evidence is
from automated validation (dependency order, capacity limits, sizing
safety) and agent-run logs, not human judgment of plan quality.

**Planned fix:** Schedule a stakeholder review session before the final
submission using `data/stakeholder_p1_plan.csv` and the B0/B1
equivalents, and record the 1-5 ratings called for in Milestone 1.

## 8. Same-wave predecessor/successor placement is a simplifying assumption

**Observed:** `graph_utils.topological_waves_first_fit` allows a server
to be scheduled in the same wave as its direct predecessor (the
dependency check is `wave(predecessor) <= wave(successor)`, not strictly
less than). This was a deliberate simplification to keep the packer
simple, but it has not been validated against real migration-runbook
practice, where same-wave concurrent cutover of a dependent pair may not
always be operationally safe.

**Planned fix:** Confirm with the stakeholder whether same-wave
placement of a direct dependency is acceptable, or whether the
constraint should be strict (`wave(predecessor) < wave(successor)`), and
adjust if needed.

## 9. Single Neo4j instance reused across dataset runs

**Observed:** Both datasets are loaded into the same Neo4j container,
cleared between runs via `MATCH (n) DETACH DELETE n`. There is no
per-dataset database isolation (Neo4j Community Edition only supports one
default database), so a run against one dataset must fully complete and
clean up before the next begins; parallel evaluation runs are not
currently possible.

**Planned fix:** Acceptable for the current single-operator evaluation
workflow; if parallel/automated repeated-trial evaluation (see item 6) is
implemented, move to either Neo4j Enterprise multi-database support or a
disposable per-run container.

---

Let me know if you want me to strengthen the community-priority heuristic or redesign the synthetic benchmark to actually demonstrate that benefit before you write the Milestone 2 report, or if you'd like to move to drafting the report now using these real results.
