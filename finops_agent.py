"""
Module 4b - Cloud & FinOps Agent (P1, new in this milestone)
================================================================
Maps each server's recommended AWS instance (written earlier by
sizing_agent.py) to a monthly cost using a versioned static pricing
catalog (data/aws_pricing_catalog.json), and compares it against the
server's Current_Monthly_Cost to produce an auditable savings estimate.
All arithmetic is deterministic; the catalog date and source are recorded
so the estimate is reproducible and explicitly labeled as an estimate.
"""

import json
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from common import PRICING_CATALOG_PATH, get_graph, get_llm


def _load_catalog(path: Path = PRICING_CATALOG_PATH) -> dict:
    return json.loads(Path(path).read_text())


@tool
def estimate_aws_cost() -> str:
    """Compute estimated monthly AWS cost for every server from its
    already-assigned target_size (set by the Sizing Agent) using the
    static pricing catalog, write aws_monthly_cost / savings_amount /
    savings_pct back to Neo4j, and return an aggregate summary.

    Returns:
        A summary of total current cost, total estimated AWS cost, and
        total savings, with the catalog version used.
    """
    catalog = _load_catalog()
    instances = catalog["instances"]
    hours = catalog["hours_per_month"]

    graph = get_graph()
    rows = graph.query(
        "MATCH (s:Server) "
        "RETURN s.id AS id, s.target_size AS size, "
        "s.current_monthly_cost AS current_cost"
    )

    if not rows:
        return "ERROR: No Server nodes found. Run sizing_agent.py first."

    updates = []
    unsupported = []
    total_current = 0.0
    total_aws = 0.0
    for r in rows:
        size = r.get("size")
        current_cost = float(r.get("current_cost") or 0.0)
        spec = instances.get(size)
        if spec is None:
            unsupported.append((r["id"], size))
            continue
        aws_monthly = round(spec["hourly_usd"] * hours, 2)
        savings = round(current_cost - aws_monthly, 2)
        savings_pct = round((savings / current_cost) * 100, 1) if current_cost > 0 else None
        updates.append({
            "id": r["id"],
            "aws_monthly_cost": aws_monthly,
            "savings_amount": savings,
            "savings_pct": savings_pct if savings_pct is not None else 0.0,
        })
        total_current += current_cost
        total_aws += aws_monthly

    if unsupported:
        return (
            "ERROR: Unsupported SKUs found in pricing catalog, aborting write: "
            f"{unsupported}. Add them to {PRICING_CATALOG_PATH} first."
        )

    graph.query(
        "UNWIND $updates AS u "
        "MATCH (s:Server {id: u.id}) "
        "SET s.aws_monthly_cost = u.aws_monthly_cost, "
        "    s.savings_amount = u.savings_amount, "
        "    s.savings_pct = u.savings_pct",
        params={"updates": updates},
    )

    total_savings = round(total_current - total_aws, 2)
    total_savings_pct = round((total_savings / total_current) * 100, 1) if total_current else 0.0

    return (
        "FinOps estimate complete (catalog "
        f"{catalog['catalog_version']}, {catalog['region']}, "
        f"{catalog['os_assumption']}, {hours} hrs/month). "
        "This is a static estimate, not a live AWS billing quote.\n"
        f"   Servers priced          : {len(updates)}\n"
        f"   Total current monthly cost : ${total_current:,.2f}\n"
        f"   Total estimated AWS cost   : ${total_aws:,.2f}\n"
        f"   Estimated monthly savings  : ${total_savings:,.2f} "
        f"({total_savings_pct:+.1f}%)"
    )


def build_agent():
    llm = get_llm()
    tools = [estimate_aws_cost]
    system_message = SystemMessage(
        content=(
            "You are the Cloud & FinOps Agent. Use the `estimate_aws_cost` "
            "tool to compute the estimated AWS monthly cost and savings "
            "for the environment, then summarize the result, explicitly "
            "noting that it is a static estimate with documented "
            "assumptions, not a live billing quote."
        )
    )
    return create_agent(llm, tools, system_prompt=system_message)


if __name__ == "__main__":
    try:
        agent = build_agent()
        result = agent.invoke({
            "messages": [
                HumanMessage(content="Estimate AWS cost and savings for the environment.")
            ]
        })
        print("\n--- Agent Response ---")
        for msg in result["messages"]:
            role = getattr(msg, "type", "unknown")
            if hasattr(msg, "content") and msg.content:
                print(f"\n[{role}] {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[tool_call] {tc['name']}({tc['args']})")
    except Exception as exc:
        print(f"Agent invocation failed ({exc}). Running tool directly...")
        print(estimate_aws_cost.invoke({}))
