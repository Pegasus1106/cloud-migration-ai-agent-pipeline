"""End-to-end case study (Milestone 2 template, Section 4) from a real run.

Runs one scripted session on D1 and writes a readable transcript of every agent
decision, tool call, verification, and human decision to results/case_study/:

  1. User request in chat -> Constraint Agent tool calls -> proposed configuration
  2. Confirmation (scripted on behalf of the user and labelled as such)
  3. Supervisor run: validator violations -> Repair Agent tool calls -> re-validation
  4. Human review: agent-drafted items -> decisions (one server excluded -> re-planning)
  5. Verified explanations -> final plan -> two "Ask the plan" questions

  python eval/case_study.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import DATASETS, RESULTS  # noqa: E402
from p1 import config  # noqa: E402
from p1.agents.constraint_agent import ConstraintAgent  # noqa: E402
from p1.agents.qa_agent import ask  # noqa: E402
from p1.core.data import load_and_normalize  # noqa: E402
from p1.core.graphdb import GraphStore  # noqa: E402
from p1.export import load_plan_into_neo4j  # noqa: E402
from p1.supervisor import RunOptions, Supervisor  # noqa: E402
from p1.tracing import Tracer  # noqa: E402

REQUEST = ("Keep each wave under 3 TB, move the CRM and the payment gateway apps early, "
           "and start with a small low-risk pilot wave.")
QUESTIONS = ["Which servers depend on hr-db-01, and which waves are they in?",
             "Which high-risk servers moved in the first two waves, and why are they high risk?"]
OUT = RESULTS / "case_study"


def _tool_lines(events: list[dict], agent: str) -> list[str]:
    lines = []
    for e in events:
        if e.get("agent") != agent:
            continue
        if e["kind"] == "tool":
            lines.append(f"- **tool `{e['name']}`** ({e['status']}, {e.get('latency_ms', 0)} ms)\n"
                         f"  - input: `{e.get('input', '')[:300]}`\n  - output: `{e.get('output', e.get('error', ''))[:400]}`")
        elif e["kind"] == "llm":
            req = ", ".join(e.get("tool_calls_requested") or []) or "final answer"
            lines.append(f"- LLM call ({e.get('model')}, {e.get('latency_ms')} ms, "
                         f"{(e.get('tokens_in') or 0) + (e.get('tokens_out') or 0)} tokens) -> {req}")
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Case study: one end-to-end P1 run (D1, stakeholder sample)", "",
          f"Model: `{config.AGENT_MODEL}`. Everything below is copied from the run's trace "
          f"(`trace.jsonl`); nothing was edited by hand.", ""]

    df, _, quality = load_and_normalize(DATASETS["D1"])
    md += ["## 1. Task and context", "",
           f"Input: `data/sample.csv` — {quality['servers_after_dedup']} servers, {quality['edge_count']} dependencies, "
           f"{quality['app_count']} applications, {quality['total_transfer_gb']:,.0f} GB to move, "
           f"{quality['rows_flagged_for_manual_audit']} incomplete records.", "",
           f"User request (chat): **“{REQUEST}”**", "",
           "Expected outcome: a plan with no wave above 3,000 GB except approved indivisible groups, CRM and "
           "Payment_Gateway in the earliest feasible waves, a low-risk pilot first, valid instance sizes, and an "
           "auditable cost estimate.", ""]

    ctracer = Tracer("case-study-constraints")
    agent = ConstraintAgent(df, ctracer)
    turn = agent.send(REQUEST)
    md += ["## 2. Constraint Agent (LLM + tools)", ""] + _tool_lines(ctracer.events, "constraint_agent")
    md += ["", f"Agent reply: > {turn['reply']}", ""]
    if turn["status"] != "proposed":
        (OUT / "case_study.md").write_text("\n".join(md))
        sys.exit("Constraint Agent asked for clarification; see case_study.md")
    cfg = turn["proposed"]
    cfg.confirmed, cfg.confirmed_by = True, "case-study script on behalf of the user"
    md += ["Proposed configuration (shown to the user, then confirmed):", "", "```json",
           json.dumps(cfg.model_dump(include={"max_transfer_gb", "target_cloud", "early_app_names", "early_server_ids",
                                              "pilot_wave", "honor_early_migration_flags"}), indent=2), "```",
           *(f"- warning: {w}" for w in turn["warnings"]), ""]

    sup = Supervisor(DATASETS["D1"], cfg, RunOptions(review_mode="interactive", out_dir=RESULTS / "case_study" / "run"),
                     run_id="case-study")
    md += ["## 3. Supervisor run", "", "| Node | Result |", "|---|---|"]
    for ev in sup.stream():
        md.append(f"| {ev['label']} | {ev['progress']} |")
    first_val = next((e for e in sup.tracer.events if e["kind"] == "node" and e["name"] == "validate"
                      and e["status"] == "ok"), None)
    state = sup.state
    md += ["", "### Validator → Repair Agent", "",
           f"First validation: {first_val['progress'] if first_val else ''}.", ""]
    for rnd in state.get("repair_log") or []:
        md.append(f"Repair round {rnd['round']} ({rnd['mode']}), violations: {', '.join(rnd['violations'])}")
    md += [""] + _tool_lines(sup.tracer.events, "repair_agent") + [""]
    for rnd in state.get("repair_log") or []:
        for r in rnd["resizes"]:
            md.append(f"- **{r['server_id']}**: {r['from']} → **{r['to']}** (${r['monthly_cost']:,.2f}/mo) — "
                      f"“{r['reason']}” [{r['decided_by']}]")
    md.append("")

    md += ["## 4. Human review (interrupt → decisions → resume)", ""]
    decisions = {}
    if sup.interrupt_payload:
        for item in sup.interrupt_payload["items"]:
            action = item.get("recommended_action")
            note = "accepted the agent's recommendation"
            if item["kind"] == "data_gap" and item["server_ids"] == ["SRV-032"]:
                action, note = "exclude_server", "owner must re-measure RAM first; exclude and re-plan"
            decisions[item["id"]] = {"action": action, "comment": note}
            md += [f"- `{item['id']}` ({item['kind']}, drafted by {item.get('drafted_by')}): {item.get('summary', '')}",
                   f"  - suggestion: {item.get('suggestion', '')}",
                   f"  - agent recommends **{item.get('recommended_action')}** → reviewer chose **{action}** ({note})"]
        md += ["", "| Node (after resume) | Result |", "|---|---|"]
        for ev in sup.resume(decisions):
            md.append(f"| {ev['label']} | {ev['progress']} |")
    state = sup.state
    verif = [e for e in sup.tracer.events if e["kind"] == "verification"]
    md += ["", "## 5. Verified explanations", "",
           f"Verifier rejections during this run: {len(verif)} (each was retried once, then templated if still wrong).", ""]
    for e in verif[:6]:
        md.append(f"- {e['name']} {e.get('server_id', e.get('wave', ''))} attempt {e.get('attempt')}: {e.get('errors')}")
    waves = (state.get("explanations") or {}).get("waves", {})
    for w in sorted(waves, key=int)[:4]:
        md += ["", f"**Wave {w}** ({waves[w]['source']}): {waves[w]['text']}"]
    m, f = state["validation"]["metrics"], state["finops"]
    md += ["", "## 6. Final output", "",
           f"- Status: **{state['status']}**; excluded: {state.get('excluded_ids')}",
           f"- {m['wave_count']} waves; dependencies kept together {m['dependency_edges'] - m['cross_wave_edges']}"
           f"/{m['dependency_edges']}; utilization {m['utilization_pct']}%; over-limit waves {m['waves_over_limit']} "
           f"(approved {m['over_limit_approved']}); undersized {m['undersized']}; early on time {m['early_satisfied_pct']}%",
           f"- Cost: ${f['total_current_monthly']:,.0f} → ${f['total_aws_monthly']:,.0f}/month (compute-only estimate)",
           f"- Files: `{sup.out_dir}`", ""]

    load_plan_into_neo4j(state)
    md += ["## 7. Ask the plan", ""]
    with GraphStore() as store:
        for q in QUESTIONS:
            r = ask(q, store, sup.tracer)
            md += [f"**Q:** {q}", "", "```cypher", r["cypher"] or "", "```",
                   f"**A** ({r['attempts']} queries, {r['query_errors']} corrected, {r['latency_ms'] / 1000:.1f}s): {r['answer']}", ""]
    s = sup.tracer.summary()
    md += ["## 8. Agent activity in this run", "",
           f"{s['llm_calls'] + ctracer.summary()['llm_calls']} LLM calls, {s['tool_calls'] + ctracer.summary()['tool_calls']} "
           f"tool calls ({s['tool_errors'] + ctracer.summary()['tool_errors']} rejected/errors), {s['fallbacks']} fallbacks, "
           f"{s['tokens_in'] + s['tokens_out'] + ctracer.summary()['tokens_in'] + ctracer.summary()['tokens_out']:,} tokens."]
    (OUT / "case_study.md").write_text("\n".join(md))
    print(f"Wrote {OUT / 'case_study.md'}")


if __name__ == "__main__":
    main()
