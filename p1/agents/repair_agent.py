"""Repair Agent: chooses bounded recovery actions for validator failures.

Agentic behaviour: the model reads the validator's violations, inspects the
affected servers, queries the catalog, and decides which fix to apply (e.g.
general-purpose vs memory-optimized instance, avoiding burstable types for
busy servers), or escalates to a human with concrete options. Every action
goes through a guarded tool (catalog membership, capacity fit, open violation),
and the supervisor re-validates the whole plan afterwards.
"""

import json
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from p1.agents.common import dumps, last_ai_text, run_config
from p1.core import catalog as cat
from p1.llm import get_llm
from p1.tracing import Tracer

PROMPT_VERSION = "repair-v2.0"
SYSTEM_PROMPT = """You are the Repair Agent of a cloud-migration planning system. A deterministic
validator rejected the current plan. Fix what you can with your tools and escalate the rest.

Procedure:
1. Call `list_violations`.
2. For each sizing violation: call `get_server`, then `find_instance_types` with the server's required
   vCPU and RAM. Choose ONE returned instance using this policy:
   - prefer the cheapest instance that fits;
   - if RAM per vCPU is 8 GB or more, prefer the memory-optimized family ('memory');
   - avoid 'burstable' instances when normalized CPU is above 40%.
   Then call `resize_server` with a one-sentence reason that names the policy you applied.
3. Only use instance types returned by `find_instance_types`. Never invent one.
4. If a violation cannot be fixed with these tools, call `escalate_to_human` with a short summary and
   2-3 concrete options for the reviewer.
5. Finish with one line per action you took.
"""

REPAIRABLE = {"sizing", "pricing"}


@dataclass
class RepairSession:
    violations: list[dict]
    servers: dict[str, dict]
    catalog: dict
    resizes: dict[str, dict] = field(default_factory=dict)
    escalations: list[dict] = field(default_factory=list)
    rejected_actions: int = 0

    def open_violation(self, server_id: str) -> dict | None:
        for v in self.violations:
            if v["category"] in REPAIRABLE and server_id in v["server_ids"] and server_id not in self.resizes:
                return v
        return None


def _tools(session: RepairSession):
    @tool
    def list_violations() -> str:
        """List the validator's open blocking violations that you are allowed to work on."""
        return dumps([{k: v[k] for k in ("id", "category", "message", "server_ids")} for v in session.violations])

    @tool
    def get_server(server_id: str) -> str:
        """Return a server's allocated vCPU/RAM, normalized CPU %, OS, current instance, and costs."""
        s = session.servers.get(server_id)
        if s is None:
            return f"ERROR: unknown server {server_id}"
        return dumps({"server_id": server_id, "hostname": s.get("Hostname"), "app": s.get("App_Name"),
                      "os": s.get("OS_Version"), "allocated_vcpu": s.get("Allocated_vCPU"),
                      "allocated_ram_gb": s.get("Allocated_RAM_GB"),
                      "ram_per_vcpu": round(float(s["Allocated_RAM_GB"]) / max(1.0, float(s["Allocated_vCPU"])), 1),
                      "normalized_cpu_pct": s.get("Normalized_CPU_Pct"), "current_instance": s.get("Target_Size"),
                      "current_monthly_cost_usd": s.get("Current_Monthly_Cost")})

    @tool
    def find_instance_types(min_vcpu: float, min_ram_gb: float, family: str = "") -> str:
        """Catalog instances (cheapest first) with at least min_vcpu and min_ram_gb.
        family: optional filter 'general', 'memory', or 'burstable'."""
        rows = cat.fitting_instances(float(min_vcpu), float(min_ram_gb), session.catalog, family or None)
        if not rows:
            return "ERROR: no catalog instance satisfies this requirement"
        return dumps(rows[:6])

    @tool
    def resize_server(server_id: str, instance_type: str, reason: str) -> str:
        """Change one server's target instance to fix its sizing/pricing violation."""
        v = session.open_violation(server_id)
        if v is None:
            session.rejected_actions += 1
            return f"REJECTED: {server_id} has no open sizing/pricing violation (or was already resized)"
        s = session.servers[server_id]
        spec = cat.spec(instance_type, session.catalog)
        if spec is None:
            session.rejected_actions += 1
            return f"REJECTED: '{instance_type}' is not in the catalog; use find_instance_types"
        if not cat.fits(instance_type, float(s["Allocated_vCPU"]), float(s["Allocated_RAM_GB"]), session.catalog):
            session.rejected_actions += 1
            return (f"REJECTED: {instance_type} ({spec['vcpu']} vCPU / {spec['ram_gb']} GB) is smaller than "
                    f"{server_id}'s {s['Allocated_vCPU']:.0f} vCPU / {s['Allocated_RAM_GB']:.0f} GB")
        old, new_cost = s.get("Target_Size"), cat.monthly_cost(instance_type, session.catalog)
        session.resizes[server_id] = {"server_id": server_id, "violation_id": v["id"], "from": old,
                                      "to": instance_type, "monthly_cost": new_cost, "reason": reason}
        return dumps({"status": "resized", "server_id": server_id, "from": old, "to": instance_type,
                      "new_monthly_cost_usd": new_cost})

    @tool
    def escalate_to_human(violation_id: str, summary: str, options: list[str]) -> str:
        """Send a violation you cannot fix to the human review queue with 2-3 options."""
        if not any(v["id"] == violation_id for v in session.violations):
            session.rejected_actions += 1
            return f"REJECTED: unknown violation id {violation_id}"
        session.escalations.append({"violation_id": violation_id, "summary": summary, "options": options[:4]})
        return "escalated"

    return [list_violations, get_server, find_instance_types, resize_server, escalate_to_human]


def deterministic_repair(session: RepairSession) -> None:
    """Rule-based fallback applying the same policy as the prompt."""
    for v in session.violations:
        for sid in v["server_ids"]:
            if sid in session.resizes or v["category"] not in REPAIRABLE:
                continue
            s = session.servers[sid]
            cpu, ram = float(s["Allocated_vCPU"]), float(s["Allocated_RAM_GB"])
            rows = cat.fitting_instances(cpu, ram, session.catalog)
            if ram / max(cpu, 1) >= 8:
                rows = [r for r in rows if r["family"] == "memory"] or rows
            if float(s.get("Normalized_CPU_Pct") or 0) > 40:
                rows = [r for r in rows if r["family"] != "burstable"] or rows
            if not rows:
                session.escalations.append({"violation_id": v["id"], "summary": f"No catalog instance fits {sid}",
                                            "options": ["Add a larger instance type to the catalog",
                                                        "Exclude the server from this plan"]})
                continue
            r = rows[0]
            session.resizes[sid] = {"server_id": sid, "violation_id": v["id"], "from": s.get("Target_Size"),
                                    "to": r["instance_type"], "monthly_cost": r["monthly_usd"],
                                    "reason": "rule-based fallback: cheapest fitting instance under the repair policy"}


def run_repair(violations: list[dict], servers: dict[str, dict], catalog: dict, tracer: Tracer | None,
               use_llm: bool = True, model: str | None = None) -> dict:
    session = RepairSession([v for v in violations if v["category"] in REPAIRABLE], servers, catalog)
    mode, reply, error = "llm", "", None
    if use_llm:
        try:
            agent = create_agent(get_llm(model), _tools(session), system_prompt=SYSTEM_PROMPT)
            result = agent.invoke({"messages": [HumanMessage(content="The validator rejected the plan. Repair it.")]},
                                  config=run_config(tracer, "repair_agent", PROMPT_VERSION, recursion_limit=60))
            reply = last_ai_text(result["messages"])
        except Exception as exc:  # noqa: BLE001 - any model/tool failure -> rule fallback
            error = f"{type(exc).__name__}: {exc}"
    if not use_llm or error:
        mode = "fallback"
        if tracer:
            tracer.event("fallback", "repair_agent", reason=error or "llm_disabled")
    # Anything the agent left untouched is handled by the rule policy, and recorded as such.
    untouched = [v for v in session.violations
                 if not any(sid in session.resizes for sid in v["server_ids"])
                 and not any(e["violation_id"] == v["id"] for e in session.escalations)]
    if untouched:
        before = set(session.resizes)
        deterministic_repair(RepairSession(untouched, servers, catalog, session.resizes, session.escalations))
        if set(session.resizes) - before and mode == "llm":
            mode = "llm+fallback"
            if tracer:
                tracer.event("fallback", "repair_agent", reason=f"{len(untouched)} violations left untouched by the agent")
    for r in session.resizes.values():
        r.setdefault("decided_by", "repair_agent" if mode == "llm" else mode)
    return {"mode": mode, "resizes": list(session.resizes.values()), "escalations": session.escalations,
            "rejected_actions": session.rejected_actions, "agent_reply": reply, "error": error}


if __name__ == "__main__":  # manual smoke test
    print(json.dumps(run_repair([], {}, cat.load_catalog(), None, use_llm=False), indent=2))
