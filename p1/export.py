"""Run outputs (Milestone 1 "Outputs": migration-plan CSV, inbound/outbound
dependency summary, current vs AWS cost comparison, wave explanations, audit log)."""

import json
import time
from pathlib import Path

import pandas as pd

from p1 import config
from p1.agents import constraint_agent, qa_agent, repair_agent, review_agent, risk_explainer, wave_explainer
from p1.core.graphdb import GraphStore, neo4j_lock

PLAN_COLUMNS = [
    "Server_ID", "Hostname", "App_Name", "OS_Version", "Business_Tags", "Allocated_vCPU", "Allocated_RAM_GB",
    "Storage_GB", "Normalized_CPU_Pct", "Transfer_GB", "Target_Size", "Size_Source", "Current_Monthly_Cost",
    "AWS_Monthly_Cost", "Savings_Amount", "Savings_Pct", "License_Note", "RightSized_Size",
    "RightSized_Monthly_Cost", "Risk_Level", "Risk_Score", "Risk_Factor_Codes", "Baseline_Risk",
    "Risk_Explanation", "Inbound_Degree", "Outbound_Degree", "WCC_Id", "SCC_Id", "SCC_Size", "Community_Id",
    "Group_Id", "Wave_ID", "Early_Migration_Requested", "Early_Related", "Pilot", "Excluded",
    "requires_manual_audit",
]
PROMPT_VERSIONS = {
    "constraint_agent": constraint_agent.PROMPT_VERSION, "repair_agent": repair_agent.PROMPT_VERSION,
    "review_agent": review_agent.PROMPT_VERSION, "risk_explainer": risk_explainer.PROMPT_VERSION,
    "wave_explainer": wave_explainer.PROMPT_VERSION, "qa_agent": qa_agent.PROMPT_VERSION,
}


def _plan_props(servers: list[dict]) -> list[dict]:
    return [{"id": s["Server_ID"], "props": {
        "target_size": s.get("Target_Size"), "aws_monthly_cost": s.get("AWS_Monthly_Cost"),
        "savings_amount": s.get("Savings_Amount"), "risk_level": s.get("Risk_Level"),
        "risk_score": s.get("Risk_Score"), "risk_factors": s.get("Risk_Factor_Codes") or [],
        "wave_id": s.get("Wave_ID"), "group_id": s.get("Group_Id"), "pilot": bool(s.get("Pilot")),
        "early_related": bool(s.get("Early_Related")), "excluded": bool(s.get("Excluded")),
        "wcc_id": s.get("WCC_Id"), "scc_id": s.get("SCC_Id"), "scc_size": s.get("SCC_Size"),
        "community_id": s.get("Community_Id")}} for s in servers]


def load_plan_into_neo4j(state: dict) -> bool:
    """Make sure Neo4j holds this run's plan (it stores one plan at a time). Returns True if reloaded."""
    with neo4j_lock(), GraphStore() as gs:
        if gs.current_plan() == state["run_id"]:
            return False
        gs.reset()
        gs.ingest(pd.DataFrame(state["servers"]), state["edges"])
        gs.write_plan(_plan_props(state["servers"]), state["run_id"])
    return True


def plan_frame(servers: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(servers)
    if "Risk_Factor_Codes" in df:
        df["Risk_Factor_Codes"] = df["Risk_Factor_Codes"].map(lambda c: "|".join(c) if isinstance(c, list) else c)
    for col in PLAN_COLUMNS:
        if col not in df:
            df[col] = None
    df["Wave_ID"] = df["Wave_ID"].astype("Int64")
    return df[PLAN_COLUMNS].sort_values(["Wave_ID", "Server_ID"], na_position="last")


def _write(path: Path, obj) -> str:
    path.write_text(json.dumps(obj, indent=2, default=str))
    return str(path)


def _markdown(state: dict, sup) -> str:
    v = state["validation"]
    f = state["finops"]
    lines = [f"# Migration plan {state['run_id']}", "",
             f"- Status: **{state['status']}** {('— ' + state['stop_reason']) if state.get('stop_reason') else ''}",
             f"- Dataset: `{state['dataset_path']}`; wave limit {sup.cfg.max_transfer_gb:,.0f} GB; "
             f"target {sup.cfg.target_cloud} ({sup.cfg.region})",
             f"- Waves: {v['metrics']['wave_count']}; dependencies split across waves: "
             f"{v['metrics']['cross_wave_edges']} of {v['metrics']['dependency_edges']}",
             f"- Estimated monthly cost: ${f['total_current_monthly']:,.0f} today -> ${f['total_aws_monthly']:,.0f} on AWS "
             f"(compute only; {f['servers_with_license_not_included']} servers exclude OS/DB license cost)", ""]
    waves = (state.get("explanations") or {}).get("waves") or {}
    if waves:
        lines += ["## Waves", ""]
        for w in sorted(waves, key=int):
            lines += [f"**Wave {w}** — {waves[w]['servers']} servers, {waves[w]['total_transfer_gb']:,.0f} GB "
                      f"_(explanation: {waves[w]['source']})_", "", waves[w]["text"], ""]
    high = [s for s in state["servers"] if s.get("Risk_Level") == "High" and not s.get("Excluded")]
    if high:
        lines += ["## High-risk servers", ""]
        lines += [f"- **{s['Server_ID']}** ({s['Hostname']}, wave {s.get('Wave_ID')}): {s.get('Risk_Explanation', '')}"
                  for s in high]
        lines.append("")
    if state.get("repair_log"):
        lines += ["## Repair actions", ""]
        for rnd in state["repair_log"]:
            for r in rnd["resizes"]:
                lines.append(f"- Round {rnd['round']}: {r['server_id']} {r['from']} -> {r['to']} "
                             f"(${r['monthly_cost']:,.2f}/mo) — {r['reason']} [{r['decided_by']}]")
            for e in rnd["escalations"]:
                lines.append(f"- Round {rnd['round']}: escalated {e['violation_id']} — {e['summary']}")
        lines.append("")
    if state.get("review_items"):
        lines += ["## Human review", ""]
        for i in state["review_items"]:
            d = i.get("decision") or {}
            lines.append(f"- `{i['id']}` ({i['kind']}): {i.get('summary', i['title'])} "
                         f"-> **{d.get('action', 'open')}** by {d.get('decided_by', '-')}")
        lines.append("")
    return "\n".join(lines)


def export_run(sup, state: dict) -> dict:
    out = sup.out_dir
    out.mkdir(parents=True, exist_ok=True)
    plan = plan_frame(state["servers"])
    files = {"plan_csv": str(out / "plan.csv")}
    plan.to_csv(files["plan_csv"], index=False)
    files["validation_report"] = _write(out / "validation_report.json", state["validation"])
    files["wave_summary"] = _write(out / "wave_summary.json", {
        "max_transfer_gb": sup.cfg.max_transfer_gb, "wave_totals_gb": state["plan"]["wave_totals"],
        "infeasible_groups": state["plan"]["infeasible"], "pilot": state["plan"].get("pilot"),
        "waves": (state.get("explanations") or {}).get("waves", {})})
    files["graph_summary"] = _write(out / "graph_summary.json", state["graph_stats"])
    files["finops_summary"] = _write(out / "finops_summary.json", state["finops"])
    files["quality_report"] = _write(out / "quality_report.json", state["quality"])
    files["review_log"] = _write(out / "review_log.json", {"items": state.get("review_items", []),
                                                           "decisions": state.get("review_decisions", [])})
    files["repair_log"] = _write(out / "repair_log.json", state.get("repair_log", []))
    (out / "explanations.md").write_text(_markdown(state, sup))
    files["explanations_md"] = str(out / "explanations.md")
    files["trace"] = str(out / "trace.jsonl")

    if state["graph_stats"].get("engine") == "neo4j-gds":
        try:
            with neo4j_lock(), GraphStore() as gs:
                gs.write_plan(_plan_props(state["servers"]), state["run_id"])
        except Exception as exc:  # noqa: BLE001
            sup.tracer.event("fallback", "neo4j_write_back", reason=f"{type(exc).__name__}: {exc}")

    expl_stats = (state.get("explanations") or {}).get("stats", {})
    run_log = {
        "run_id": state["run_id"], "dataset": state["dataset_path"], "status": state["status"],
        "stop_reason": state.get("stop_reason", ""), "passed_validation": state["validation"]["passed"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(sup.started)),
        "duration_sec": round(time.time() - sup.started, 1),
        "config": sup.cfg.model_dump(),
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(sup.opt).items()},
        "model": (sup.opt.model or config.AGENT_MODEL) if sup.opt.use_llm else None,
        "prompt_versions": PROMPT_VERSIONS if sup.opt.use_llm else {},
        "catalog_version": state["finops"]["catalog_version"],
        "lifecycle_version": state["risk_summary"]["lifecycle_version"],
        "graph_engine": state["graph_stats"].get("engine"),
        "timings_ms": state.get("timings", {}),
        "metrics": {
            **state["validation"]["metrics"],
            "blocking_violations": state["validation"]["blocking_count"],
            "finops_current_monthly": state["finops"]["total_current_monthly"],
            "finops_aws_monthly": state["finops"]["total_aws_monthly"],
            "finops_savings_pct": state["finops"]["savings_pct"],
            "risk_levels": state["risk_summary"]["levels"],
            "repair_rounds": len(state.get("repair_log") or []),
            "repair_resizes": sum(len(r["resizes"]) for r in state.get("repair_log") or []),
            "repair_escalations": sum(len(r["escalations"]) for r in state.get("repair_log") or []),
            "repair_rejected_actions": sum(r["rejected_actions"] for r in state.get("repair_log") or []),
            "review_items": len(state.get("review_items") or []),
            "review_items_drafted_by_llm": sum(i.get("drafted_by") == "llm" for i in state.get("review_items") or []),
            "explanations": expl_stats,
        },
        "agent_trace_summary": sup.tracer.summary(),
        "files": files,
    }
    files["run_log"] = _write(out / "run_log.json", run_log)
    return files
