"""Supervisor: LangGraph state machine that coordinates tools and agents.

    discover -> build_graph -> size_and_price -> assess_risk -> plan_waves -> validate
    validate --(repairable violations, budget left)--> repair -> reprice -> validate
    validate --(items need a person)-----------------> draft_review -> human_review
    human_review --(server excluded)--> plan_waves | --(rejected)--> finalize | --> validate
    validate --(passed)--> explain -> finalize          validate --(stuck)--> finalize (blocked)

Shared typed state (p1.state.P1State) is checkpointed after every node
(MemorySaver), so a run paused for human review resumes exactly where it
stopped. The happy path is fixed code; LLM decisions happen at the decision
points (constraint extraction, repair, review drafting, explanations), which
keeps the pipeline reliable while still letting the model act on observations.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from p1 import config
from p1.agents import repair_agent, review_agent, risk_explainer, wave_explainer
from p1.core import catalog as cat
from p1.core import data as data_mod
from p1.core import finops, risk
from p1.core.graph_local import components_local
from p1.core.graphdb import GraphStore, neo4j_lock
from p1.core.planning import plan_p1, resolve_early_ids
from p1.core.validation import validate_plan
from p1.export import export_run
from p1.llm import llm_available
from p1.state import P1State, RunConfig
from p1.tracing import Tracer

NODE_LABELS = {
    "discover": "Discovery & data quality", "build_graph": "Dependency graph (Neo4j GDS)",
    "size_and_price": "Sizing & FinOps", "assess_risk": "Risk scoring", "plan_waves": "Wave planning",
    "validate": "Validation", "repair": "Repair Agent", "reprice": "Re-pricing",
    "draft_review": "Review drafting", "human_review": "Human review", "explain": "Explanations",
    "finalize": "Export",
}


@dataclass
class RunOptions:
    review_mode: Literal["auto", "interactive"] = "auto"
    enable_repair: bool = True
    allow_same_wave: bool = True
    use_llm: bool = True
    model: str | None = None
    explain: bool = True
    use_neo4j: bool = True
    out_dir: Path | None = None


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


class Supervisor:
    def __init__(self, dataset_path: str | Path, cfg: RunConfig, options: RunOptions | None = None,
                 run_id: str | None = None):
        if not cfg.confirmed:
            raise PermissionError("Refusing to plan: the configuration has not been confirmed by the user.")
        self.dataset_path = str(dataset_path)
        self.cfg = cfg
        self.opt = options or RunOptions()
        if self.opt.use_llm and not llm_available():
            raise RuntimeError(f"{config.LLM_API_KEY_ENV} is not set; run with use_llm=False for rule-only mode.")
        stem = Path(dataset_path).stem
        self.run_id = run_id or f"{stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.out_dir = Path(self.opt.out_dir or config.RUNS_DIR / self.run_id)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "trace.jsonl").unlink(missing_ok=True)
        self.tracer = Tracer(self.run_id, self.out_dir / "trace.jsonl")
        self.catalog = cat.load_catalog()
        self.lifecycle = risk.load_lifecycle()
        self.graph = self._build().compile(checkpointer=MemorySaver())
        self.thread = {"configurable": {"thread_id": self.run_id}, "recursion_limit": 120}
        self.interrupt_payload: dict | None = None
        self.started = time.time()

    # ------------------------------------------------------------------ nodes
    def _timed(self, name, fn):
        def wrapper(state: P1State) -> dict:
            t = time.time()
            self.tracer.event("node", name, status="start")
            try:
                update = fn(state) or {}
            except Exception as exc:
                self.tracer.event("node", name, status="error", error=f"{type(exc).__name__}: {exc}")
                raise
            ms = int((time.time() - t) * 1000)
            timings = dict(state.get("timings") or {})
            timings[name] = timings.get(name, 0) + ms
            update["timings"] = timings
            self.tracer.event("node", name, status="ok", latency_ms=ms, progress=update.get("progress", ""))
            return update
        return wrapper

    def _discover(self, state: P1State) -> dict:
        df, edges, quality = data_mod.load_and_normalize(self.dataset_path)
        return {"servers": _records(df), "edges": edges, "quality": quality, "config": self.cfg.model_dump(),
                "run_id": self.run_id, "dataset_path": self.dataset_path, "repair_round": 0,
                "review_items": [], "review_decisions": [], "approved_exceptions": [], "acknowledged": [],
                "excluded_ids": [], "repair_log": [], "status": "running",
                "progress": f"{quality['servers_after_dedup']} servers, {quality['edge_count']} dependencies, "
                            f"{quality['rows_flagged_for_manual_audit']} incomplete records"}

    def _build_graph(self, state: P1State) -> dict:
        servers = state["servers"]
        df = pd.DataFrame(servers)
        transfer = {s["Server_ID"]: float(s["Transfer_GB"]) for s in servers}
        edges = [(e["src"], e["dst"]) for e in state["edges"]]
        engine = "neo4j-gds"
        per = stats = degrees = None
        if self.opt.use_neo4j:
            try:
                with neo4j_lock(), GraphStore() as gs:
                    if not gs.ping():
                        raise ConnectionError(f"Neo4j not reachable at {config.NEO4J_URL}")
                    gs.reset()
                    ingest = gs.ingest(df, state["edges"])
                    degrees = gs.degrees()
                    per, stats = gs.components(self.cfg.max_transfer_gb, transfer)
                    stats["ingest"] = ingest
            except Exception as exc:  # noqa: BLE001
                self.tracer.event("fallback", "build_graph", reason=f"{type(exc).__name__}: {exc}")
                per = None
        if per is None:
            engine = "networkx-fallback"
            per, stats, degrees = components_local(list(transfer), edges, transfer, self.cfg.max_transfer_gb)
        for s in servers:
            s.update(per[s["Server_ID"]])
            s["Inbound_Degree"], s["Outbound_Degree"] = degrees.get(s["Server_ID"], (0, 0))
        top_in = sorted(servers, key=lambda s: (-s["Inbound_Degree"], s["Server_ID"]))[:5]
        stats.update({"engine": engine,
                      "inbound_summary": {
                          "servers_with_inbound": sum(s["Inbound_Degree"] > 0 for s in servers),
                          "top_inbound": [{"server_id": s["Server_ID"], "hostname": s["Hostname"],
                                           "inbound_degree": s["Inbound_Degree"]} for s in top_in],
                          "derived_inbound_edges": sum(s["Inbound_Degree"] for s in servers),
                          "outbound_edges": len(edges)}})
        return {"servers": servers, "graph_stats": stats,
                "progress": f"{stats['wcc_count']} portfolios, {len(stats['cycles'])} cycles, "
                            f"{len(stats['oversized_wccs'])} oversized ({engine})"}

    def _price(self, servers: list[dict], overrides: dict[str, dict]) -> tuple[list[dict], dict]:
        for s in servers:
            ov = overrides.get(s["Server_ID"])
            if ov:
                s["Target_Size"], s["Size_Source"] = ov["to"], ov.get("decided_by", "repair_agent")
            else:
                s["Target_Size"] = cat.size_by_matrix(float(s["Allocated_vCPU"]), float(s["Allocated_RAM_GB"]),
                                                      self.cfg.target_cloud)
                s["Size_Source"] = "stakeholder sizing matrix"
        per, summary = finops.price_servers(servers, self.catalog)
        for s in servers:
            s.update(per[s["Server_ID"]])
        summary["target_cloud"] = self.cfg.target_cloud
        return servers, summary

    def _overrides(self, state: P1State) -> dict[str, dict]:
        out = {}
        for rnd in state.get("repair_log") or []:
            for r in rnd["resizes"]:
                out[r["server_id"]] = r
        return out

    def _size_and_price(self, state: P1State) -> dict:
        servers, summary = self._price(state["servers"], self._overrides(state))
        return {"servers": servers, "finops": summary, "cloud": {"target_cloud": self.cfg.target_cloud,
                                                                 "source": "user-confirmed configuration"},
                "progress": f"${summary['total_current_monthly']:,.0f} -> ${summary['total_aws_monthly']:,.0f}/month"}

    def _assess_risk(self, state: P1State) -> dict:
        servers = state["servers"]
        for s in servers:
            s.update(risk.score_server(s, self.lifecycle))
            s["Risk_Factor_Codes"] = [f["code"] for f in s["Risk_Factors"]]
        levels = pd.Series([s["Risk_Level"] for s in servers]).value_counts().to_dict()
        factor_counts: dict[str, int] = {}
        for s in servers:
            for c in s["Risk_Factor_Codes"]:
                factor_counts[c] = factor_counts.get(c, 0) + 1
        summary = {"levels": levels, "factor_counts": factor_counts, "lifecycle_version": self.lifecycle["lifecycle_version"],
                   "baseline_rule_high": sum(s["Baseline_Risk"] == "High" for s in servers)}
        return {"servers": servers, "risk_summary": summary,
                "progress": ", ".join(f"{k} {v}" for k, v in sorted(levels.items()))}

    def _plan_waves(self, state: P1State) -> dict:
        servers = state["servers"]
        edges = [(e["src"], e["dst"]) for e in state["edges"]]
        excluded = set(state.get("excluded_ids") or [])
        res = plan_p1(servers, edges, self.cfg, allow_same_wave=self.opt.allow_same_wave, excluded_ids=excluded)
        early_rel, pilot = set(res.early_related_servers), set(res.pilot.get("servers", []))
        for s in servers:
            sid = s["Server_ID"]
            s["Excluded"] = sid in excluded
            s["Wave_ID"] = res.wave_of_server.get(sid)
            s["Group_Id"] = res.group_of_server.get(sid)
            s["Early_Related"] = sid in early_rel
            s["Pilot"] = sid in pilot
        plan = res.to_dict()
        plan.pop("wave_of_server")
        plan.pop("group_of_server")
        return {"servers": servers, "plan": plan, "replan": False,
                "progress": f"{len(res.wave_totals)} waves, {len(res.infeasible)} over-limit groups"
                            + (f", pilot {len(pilot)} servers" if self.cfg.pilot_wave else "")}

    def _validate(self, state: P1State) -> dict:
        servers = state["servers"]
        plan_df = pd.DataFrame(servers)
        cleaned = plan_df[["Server_ID", "Transfer_GB", "Allocated_vCPU", "Allocated_RAM_GB"]]
        edges = [(e["src"], e["dst"]) for e in state["edges"]]
        approved = set(state.get("approved_exceptions") or [])
        approved_groups = {a.split(":", 1)[1] for a in approved if a.startswith("VOLUME:")}
        report = validate_plan(
            plan_df, cleaned, edges, self.cfg.max_transfer_gb, label=self.run_id,
            reported_infeasible=state["plan"]["infeasible"], approved_exceptions=approved_groups,
            early_ids=resolve_early_ids(servers, self.cfg),
            first_regular_wave=1 if state["plan"].get("pilot", {}).get("servers") else 0,
            require_finops=self.cfg.target_cloud == "AWS", config_confirmed=self.cfg.confirmed,
            excluded_ids=set(state.get("excluded_ids") or []), catalog=self.catalog)
        for v in report["violations"]:
            if v["id"] in approved:
                v["severity"] = "approved"
        report["blocking_count"] = sum(v["severity"] == "blocking" for v in report["violations"])
        report["passed"] = report["blocking_count"] == 0
        items = self._sync_review_items(state, report)
        open_items = sum(i["status"] == "open" for i in items)
        return {"validation": report, "review_items": items,
                "progress": f"{report['blocking_count']} blocking violations, {open_items} items for review"}

    def _sync_review_items(self, state: P1State, report: dict) -> list[dict]:
        items = {i["id"]: i for i in (state.get("review_items") or [])}
        acknowledged = set(state.get("acknowledged") or [])
        repair_done = state.get("repair_round", 0) >= config.REPAIR_MAX_ROUNDS or not self.opt.enable_repair

        def add(item_id, kind, title, facts, server_ids, actions):
            if item_id not in items:
                items[item_id] = {"id": item_id, "kind": kind, "title": title, "facts": facts,
                                  "server_ids": server_ids, "allowed_actions": actions, "status": "open"}

        for v in report["violations"]:
            if v["severity"] == "approved":
                continue
            if v["category"] == "transfer_volume" and v.get("group_id"):
                add(v["id"], "over_limit_group", v["message"],
                    {"servers": len(v["server_ids"]), "total_gb": v["total_gb"], "limit_gb": self.cfg.max_transfer_gb,
                     "wave": v["wave"]}, v["server_ids"], ["approve_exception", "reject_plan"])
            elif v["category"] == "early_migration" and v["severity"] == "warning":
                reason = next((d["reason"] for d in report["checks"]["early_migration"]["details"]
                               if d["server_id"] == v["server_ids"][0]), "")
                add(v["id"], "early_infeasible", v["message"], {"reason": reason}, v["server_ids"], ["acknowledge"])
            elif v["severity"] == "blocking" and (v["needs_human"] or (v["repairable"] and repair_done)):
                add(v["id"], "violation", v["message"], {k: v[k] for k in v if k in ("wave", "total_gb", "needs_vcpu",
                                                                                     "needs_ram_gb")},
                    v["server_ids"], ["approve_exception", "exclude_server", "reject_plan"])
        for sid, fields in (state["quality"].get("audit_details") or {}).items():
            if f"DATA_GAP:{sid}" not in acknowledged:
                add(f"DATA_GAP:{sid}", "data_gap", f"{sid} had incomplete discovery data",
                    {"missing_fields": fields}, [sid], ["acknowledge", "exclude_server"])
        for rnd in state.get("repair_log") or []:
            for e in rnd["escalations"]:
                add(f"ESCALATION:{e['violation_id']}", "escalation", e["summary"],
                    {"agent_options": e["options"]}, [], ["approve_exception", "exclude_server", "reject_plan"])
        return list(items.values())

    @staticmethod
    def _escalated(state: P1State) -> set[str]:
        return {e["violation_id"] for rnd in state.get("repair_log") or [] for e in rnd["escalations"]}

    def _route_after_validate(self, state: P1State) -> str:
        rep = state["validation"]
        escalated = self._escalated(state)
        repairable = [v for v in rep["violations"]
                      if v["severity"] == "blocking" and v["repairable"] and v["id"] not in escalated]
        if repairable and self.opt.enable_repair and state.get("repair_round", 0) < config.REPAIR_MAX_ROUNDS:
            return "repair"
        if any(i["status"] == "open" for i in state.get("review_items") or []):
            return "draft_review"
        return "explain" if rep["passed"] else "finalize"

    def _repair(self, state: P1State) -> dict:
        escalated = self._escalated(state)
        violations = [v for v in state["validation"]["violations"]
                      if v["severity"] == "blocking" and v["repairable"] and v["id"] not in escalated]
        servers = {s["Server_ID"]: s for s in state["servers"]}
        result = repair_agent.run_repair(violations, servers, self.catalog, self.tracer,
                                         use_llm=self.opt.use_llm, model=self.opt.model)
        rnd = state.get("repair_round", 0) + 1
        log = list(state.get("repair_log") or []) + [{"round": rnd, "violations": [v["id"] for v in violations], **result}]
        return {"repair_round": rnd, "repair_log": log,
                "progress": f"round {rnd}: {len(result['resizes'])} resized, {len(result['escalations'])} escalated "
                            f"({result['mode']})"}

    def _reprice(self, state: P1State) -> dict:
        servers, summary = self._price(state["servers"], self._overrides(state))
        return {"servers": servers, "finops": summary,
                "progress": f"${summary['total_aws_monthly']:,.0f}/month after repair"}

    def _draft_review(self, state: P1State) -> dict:
        items = review_agent.draft_items([dict(i) for i in state["review_items"]], self.tracer,
                                         use_llm=self.opt.use_llm, model=self.opt.model)
        return {"review_items": items,
                "progress": f"{sum(i['status'] == 'open' for i in items)} items drafted for the reviewer"}

    def _human_review(self, state: P1State) -> dict:
        items = [dict(i) for i in state["review_items"]]
        open_items = [i for i in items if i["status"] == "open"]
        if self.opt.review_mode == "interactive":
            decisions = interrupt({"run_id": self.run_id, "items": open_items})
            decided_by = "human"
        else:
            decisions = {i["id"]: {"action": AUTO_POLICY.get(i["kind"], "defer"),
                                   "comment": "auto-policy for batch evaluation (fixed, documented rule)"}
                         for i in open_items}
            decided_by = "auto-policy"
        approved = list(state.get("approved_exceptions") or [])
        acknowledged = list(state.get("acknowledged") or [])
        excluded = list(state.get("excluded_ids") or [])
        log = list(state.get("review_decisions") or [])
        status, replan = state.get("status", "running"), False
        for item in open_items:
            d = decisions.get(item["id"]) or {"action": "defer", "comment": "no decision supplied"}
            action = d["action"] if d["action"] in item["allowed_actions"] + ["defer"] else "defer"
            item.update(status="deferred" if action == "defer" else "resolved",
                        decision={"action": action, "comment": d.get("comment", ""),
                                  "decided_by": d.get("decided_by", decided_by),
                                  "agrees_with_agent": action == item.get("recommended_action"),
                                  "ts": datetime.now().isoformat(timespec="seconds")})
            log.append({"item_id": item["id"], **item["decision"]})
            if action == "approve_exception":
                approved.append(item["id"].removeprefix("ESCALATION:"))
            elif action == "acknowledge":
                acknowledged.append(item["id"])
            elif action == "exclude_server":
                excluded.extend(s for s in item["server_ids"] if s not in excluded)
                acknowledged.append(item["id"])
                replan = True
            elif action == "reject_plan":
                status = "rejected"
        by_id = {i["id"]: i for i in open_items}
        items = [by_id.get(i["id"], i) for i in items]
        return {"review_items": items, "review_decisions": log, "approved_exceptions": approved,
                "acknowledged": acknowledged, "excluded_ids": excluded, "status": status,
                "stop_reason": "rejected by reviewer" if status == "rejected" else "",
                "progress": f"{len(open_items)} decisions recorded ({decided_by})" + ("; re-planning" if replan else ""),
                "replan": replan}

    def _route_after_review(self, state: P1State) -> str:
        if state.get("status") == "rejected":
            return "finalize"
        return "plan_waves" if state.get("replan") else "validate"

    def _explain(self, state: P1State) -> dict:
        servers = state["servers"]
        active = [s for s in servers if not s.get("Excluded")]
        expl: dict = {"risk": {}, "waves": {}, "stats": {}}
        if self.opt.explain:
            risk_out, rstats = risk_explainer.explain_risks(active, self.tracer, self.opt.use_llm, self.opt.model)
            edges = [(e["src"], e["dst"]) for e in state["edges"]]
            approved = {a.split(":", 1)[1] for a in state.get("approved_exceptions") or [] if a.startswith("VOLUME:")}
            facts = wave_explainer.wave_facts(active, edges, self.cfg.max_transfer_gb, state["plan"]["infeasible"], approved)
            wave_out, wstats = wave_explainer.explain_waves(
                facts, {s["Hostname"] for s in servers}, {s["App_Name"] for s in servers}, self.tracer,
                self.opt.use_llm, self.opt.model)
            for s in active:
                s["Risk_Explanation"] = risk_out[s["Server_ID"]]["text"]
                s["Risk_Explanation_Source"] = risk_out[s["Server_ID"]]["source"]
            expl = {"waves": {str(f["wave"]): {**f, **wave_out[f["wave"]]} for f in facts},
                    "stats": {"risk": rstats, "waves": wstats}}
        return {"servers": servers, "explanations": expl,
                "progress": "explanations verified" if self.opt.explain else "skipped"}

    def _finalize(self, state: P1State) -> dict:
        status = state.get("status", "running")
        if status == "running":
            status = "ready_for_approval" if state["validation"]["passed"] else "blocked"
        stop = state.get("stop_reason") or ("" if status == "ready_for_approval"
                                            else f"{state['validation']['blocking_count']} blocking violations remain")
        outputs = export_run(self, {**state, "status": status, "stop_reason": stop})
        return {"status": status, "stop_reason": stop, "outputs": outputs,
                "progress": f"status={status}; files in {self.out_dir}"}

    # ------------------------------------------------------------------ graph
    def _build(self) -> StateGraph:
        g = StateGraph(P1State)
        for name, fn in [("discover", self._discover), ("build_graph", self._build_graph),
                         ("size_and_price", self._size_and_price), ("assess_risk", self._assess_risk),
                         ("plan_waves", self._plan_waves), ("validate", self._validate), ("repair", self._repair),
                         ("reprice", self._reprice), ("draft_review", self._draft_review),
                         ("human_review", self._human_review), ("explain", self._explain),
                         ("finalize", self._finalize)]:
            g.add_node(name, self._timed(name, fn))
        g.add_edge(START, "discover")
        g.add_edge("discover", "build_graph")
        g.add_edge("build_graph", "size_and_price")
        g.add_edge("size_and_price", "assess_risk")
        g.add_edge("assess_risk", "plan_waves")
        g.add_edge("plan_waves", "validate")
        g.add_conditional_edges("validate", self._route_after_validate,
                                ["repair", "draft_review", "explain", "finalize"])
        g.add_edge("repair", "reprice")
        g.add_edge("reprice", "validate")
        g.add_edge("draft_review", "human_review")
        g.add_conditional_edges("human_review", self._route_after_review, ["plan_waves", "validate", "finalize"])
        g.add_edge("explain", "finalize")
        g.add_edge("finalize", END)
        return g

    # ------------------------------------------------------------------ API
    def _stream(self, payload) -> Iterator[dict]:
        self.interrupt_payload = None
        for chunk in self.graph.stream(payload, self.thread, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    self.interrupt_payload = update[0].value
                    yield {"node": "human_review", "label": NODE_LABELS["human_review"], "status": "waiting",
                           "progress": f"{len(self.interrupt_payload['items'])} items need your decision"}
                else:
                    yield {"node": node, "label": NODE_LABELS.get(node, node), "status": "done",
                           "progress": (update or {}).get("progress", "")}

    def stream(self) -> Iterator[dict]:
        yield from self._stream({"run_id": self.run_id})

    def resume(self, decisions: dict[str, dict]) -> Iterator[dict]:
        yield from self._stream(Command(resume=decisions))

    def run(self) -> dict:
        for _ in self.stream():
            pass
        if self.interrupt_payload is not None:
            raise RuntimeError("Run paused for human review; use stream()/resume() in interactive mode.")
        return self.state

    @property
    def state(self) -> dict:
        return self.graph.get_state(self.thread).values


# Batch-evaluation stand-in for the human reviewer. Only low-stakes items are
# auto-resolved; anything that would hide a real defect is deferred, which
# leaves the plan "blocked" instead of silently approving it.
AUTO_POLICY = {
    "over_limit_group": "approve_exception",   # same outcome as the prototype: dedicated, flagged wave
    "data_gap": "acknowledge",
    "early_infeasible": "acknowledge",
    "violation": "defer",
    "escalation": "defer",
}
