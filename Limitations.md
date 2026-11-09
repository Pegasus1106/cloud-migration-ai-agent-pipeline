# Limitations and open questions (Milestone 2)

Observed behaviour is separated from suspected causes and planned work. Measured numbers
are in [docs/MILESTONE2_RESULTS.md](docs/MILESTONE2_RESULTS.md); prototype failures and their
evidence are in [results/evidence/prototype_v0/](results/evidence/prototype_v0/README.md).

## Data and cost realism

1. **Synthetic "current cost".** `Current_Monthly_Cost` in all datasets comes from the documented
   formula `40·vCPU + 10·RAM_GB + 0.05·Storage_GB + 50`, not real bills, so savings percentages
   are illustrative. Needed: a real cost export from the stakeholder.
2. **Compute-only, single-region prices.** The catalog is a static us-east-1 Linux On-Demand
   snapshot (17 SKUs, re-read 2026-11-08). Storage (EBS), data transfer, backups, support, and
   Windows Server / SQL Server licenses are excluded; affected servers are flagged
   (`License_Note`) but not priced. No Windows price was available on the source pages, so none
   was invented. Azure plans are sized but not priced.
3. **Sizing follows allocation.** The stakeholder's tier matrix sizes by allocated vCPU/RAM; the
   usage-based alternative (`RightSized_Size`) is reported as a what-if only, and memory sizing
   has no telemetry behind it.
4. **Transfer volume ≈ storage, no time model.** Waves are capped by GB, but connection
   bandwidth is not used to estimate transfer hours, cutover windows, or downtime. The
   Milestone 1 question "is 3,000 GB per wave or per migration window?" is still open.
5. **Benchmark realism.** D3's portfolio text comes from `gpt-oss-120b`, but its structure
   (patterns, shared services, defects) is our design and may favour behaviours we planned for.
   The model marked only 4 of 119 apps for early migration; the generator topped this up to 10%
   deterministically and records the source of every flag.

## Agent behaviour

6. **LLM latency dominates.** Most of a P1 run is the explainers and the repair loop; D1 takes
   about 2–3 minutes with the default model versus seconds in rule-only mode.
7. **Verifiers check facts, not reasoning.** Number, name, and factor checks reject invented
   values, but cannot prove that a sentence's causal claim is right. Wave explanations are
   restricted to the planner's own `scheduling_reasons` to reduce this risk.
8. **Run-to-run variation.** The plan structure is deterministic for a given configuration;
   LLM-dependent parts (repair instance choice, wording) can vary between runs. Variation is
   measured over repeated runs.
9. **Batch review policy stands in for a person.** Evaluation runs use a fixed, documented
   auto-policy (approve over-limit indivisible groups, acknowledge data gaps, defer anything else).
   The Milestone 1 human-usefulness and explainability ratings (target ≥ 4/5) are not measured yet.
10. **Constraint coverage.** The Constraint Agent fills wave limit, cloud, early applications or
    servers, CSV-flag handling, and pilot wave; anything else is recorded in `notes` and not
    enforced (e.g. region preferences beyond us-east-1, blackout dates).

## Engineering

11. **One plan in Neo4j at a time.** Runs are serialized with a file lock; the UI reloads its plan
    before answering questions. Not multi-user.
12. **Local credentials.** The Neo4j password default matches `compose.yml` for local use only;
    override `P1_NEO4J_PASSWORD` elsewhere. Q&A queries are read-only (regex guard plus a
    READ_ACCESS session).
13. **Dependency-direction convention.** Callers move no later than the servers they call
    (same wave allowed). This keeps the team's prototype convention and should be confirmed
    with the stakeholder.
