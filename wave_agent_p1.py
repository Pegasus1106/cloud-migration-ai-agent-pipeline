"""
Module 6 - Wave Planning Agent (P1, new in this milestone)
===============================================================
The main technical contribution described in Milestone 1: a reproducible
wave algorithm that uses WCC to separate disconnected portfolios, SCC to
keep cycles indivisible, and community detection to subdivide any WCC
whose total Transfer_GB exceeds the configured limit. Builds a group-
level dependency DAG (collapsing any residual cycles for safety), then
topologically schedules groups into capacity-constrained waves,
prioritizing confirmed early-migration requests, lower risk, and higher
savings. All of this is deterministic; the LLM is used only to read back
and explain the resulting plan.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from common import DEFAULT_MAX_TRANSFER_GB, DEFAULT_RUN_CONFIG_PATH, get_graph, get_llm
from graph_utils import build_group_graph, ensure_dag, topological_waves_first_fit

OUTPUT_CSV = Path("data/final_migration_plan.csv")
WAVE_SUMMARY_JSON = Path("data/wave_summary.json")

RISK_RANK = {"Low": 0, "High": 1, None: 1}


def _assign_group_ids(servers: list[dict]) -> dict[str, str]:
    """Group key per server = its SCC. The SCC (a real cycle) is the only
    unit that is truly indivisible, so it is the only thing that can ever
    be reported infeasible for exceeding the capacity limit - matching
    the B1 baseline's granularity exactly, which is required for a fair
    comparison.

    Community/WCC membership is NOT used to force servers into one
    monolithic packing unit (an earlier version of this agent did that
    and it incorrectly produced more "infeasible" groups than the
    community-blind B1 baseline for an oversized WCC that Louvain could
    only split into a few still-oversized communities - see Milestone 2,
    Section 5). Instead, community_id is attached to each SCC as metadata
    and used only as a scheduling *priority* signal in
    `construct_and_pack_waves`, so members of the same community are
    offered to the packer together and tend to land in the same or
    adjacent waves, without ever being forced into an artificially
    oversized indivisible unit.
    """
    return {s["id"]: s["scc_id"] for s in servers}


@tool
def construct_and_pack_waves(max_transfer_gb: float = DEFAULT_MAX_TRANSFER_GB,
                              honor_early_migration: bool = True) -> str:
    """Build the group-level dependency DAG from WCC/SCC/community
    assignments and pack groups into capacity-constrained waves in
    dependency order, prioritizing early-migration requests, lower risk,
    and higher savings. Writes wave_id and group_id back to Neo4j.

    Args:
        max_transfer_gb: Maximum total Transfer_GB per wave.
        honor_early_migration: Whether Early_Migration_Requested servers
                                are prioritized to the earliest feasible wave.

    Returns:
        A summary of wave count, utilization, and any infeasible groups.
    """
    graph = get_graph()

    servers = graph.query(
        "MATCH (s:Server) WHERE s.hostname IS NOT NULL "
        "RETURN s.id AS id, s.wcc_id AS wcc_id, s.scc_id AS scc_id, "
        "s.community_id AS community_id, s.transfer_gb AS transfer_gb, "
        "s.risk_score AS risk_score, s.savings_amount AS savings_amount, "
        "s.early_migration_requested AS early_migration_requested"
    )
    if not servers:
        return "ERROR: No servers with graph components found. Run graph_agent.py first."

    edge_rows = graph.query(
        "MATCH (a:Server)-[:COMMUNICATES_WITH]->(b:Server) RETURN a.id AS src, b.id AS dst"
    )
    edges = [(r["src"], r["dst"]) for r in edge_rows]

    group_of = _assign_group_ids(servers)

    group_total_gb: dict[str, float] = {}
    group_has_early: dict[str, bool] = {}
    group_risk: dict[str, str] = {}
    group_savings: dict[str, float] = {}
    group_members: dict[str, list[str]] = {}
    group_community_votes: dict[str, dict[str, int]] = {}
    for s in servers:
        g = group_of[s["id"]]
        group_total_gb[g] = group_total_gb.get(g, 0.0) + (s["transfer_gb"] or 0.0)
        group_has_early[g] = group_has_early.get(g, False) or bool(s["early_migration_requested"])
        prev_rank = RISK_RANK.get(group_risk.get(g), 0)
        group_risk[g] = s["risk_score"] if RISK_RANK.get(s["risk_score"], 1) > prev_rank else group_risk.get(g, s["risk_score"])
        group_savings[g] = group_savings.get(g, 0.0) + (s["savings_amount"] or 0.0)
        group_members.setdefault(g, []).append(s["id"])
        votes = group_community_votes.setdefault(g, {})
        votes[s["community_id"]] = votes.get(s["community_id"], 0) + 1
    group_community = {g: max(votes, key=votes.get) for g, votes in group_community_votes.items()}

    group_graph = build_group_graph(group_of, edges)
    dag, merge_map = ensure_dag(group_graph)

    # Re-key per-group aggregates onto the (possibly merged) final DAG node ids.
    # (Merging only happens if a residual cycle crosses SCC boundaries, which
    # should not occur in practice since SCC is already the atomic unit; this
    # is kept purely as a safety net.)
    final_total_gb: dict[str, float] = {}
    final_has_early: dict[str, bool] = {}
    final_risk: dict[str, str] = {}
    final_savings: dict[str, float] = {}
    final_members: dict[str, list[str]] = {}
    final_community: dict[str, str] = {}
    for g in group_total_gb:
        final_id = merge_map.get(g, g)
        final_total_gb[final_id] = final_total_gb.get(final_id, 0.0) + group_total_gb[g]
        final_has_early[final_id] = final_has_early.get(final_id, False) or group_has_early[g]
        prev_rank = RISK_RANK.get(final_risk.get(final_id), 0)
        final_risk[final_id] = group_risk[g] if RISK_RANK.get(group_risk[g], 1) > prev_rank else final_risk.get(final_id, group_risk[g])
        final_savings[final_id] = final_savings.get(final_id, 0.0) + group_savings[g]
        final_members.setdefault(final_id, []).extend(group_members[g])
        final_community[final_id] = final_community.get(final_id, group_community[g])

    # Community is used only to cluster same-business-group SCCs next to
    # each other in the priority ordering (so first-fit tends to pack them
    # into the same or adjacent waves), never to force them into one
    # oversized indivisible unit. This is the actual "community-aware"
    # difference from the B1 baseline, which sorts purely by size.
    def priority_key(group_id: str):
        early_first = 0 if (honor_early_migration and final_has_early.get(group_id)) else 1
        risk_rank = RISK_RANK.get(final_risk.get(group_id), 0)
        community = final_community.get(group_id, "")
        return (early_first, risk_rank, str(community), -final_savings.get(group_id, 0.0), group_id)

    wave_of_group, infeasible_groups = topological_waves_first_fit(
        dag, final_total_gb, max_transfer_gb, priority_key
    )

    server_wave: dict[str, int] = {}
    server_final_group: dict[str, str] = {}
    for g in final_members:
        for member in final_members[g]:
            server_wave[member] = wave_of_group[g]
            server_final_group[member] = g

    updates = [
        {"id": sid, "wave_id": server_wave[sid], "group_id": server_final_group[sid]}
        for sid in server_wave
    ]
    graph.query(
        "UNWIND $updates AS u MATCH (s:Server {id: u.id}) "
        "SET s.wave_id = u.wave_id, s.group_id = u.group_id",
        params={"updates": updates},
    )

    wave_totals: dict[int, float] = {}
    for g, wid in wave_of_group.items():
        wave_totals[wid] = wave_totals.get(wid, 0.0) + final_total_gb[g]

    cross_wave_edges = sum(
        1 for u, v in group_graph.edges
        if server_wave.get(group_members.get(u, [None])[0]) != server_wave.get(group_members.get(v, [None])[0])
    )

    summary_payload = {
        "max_transfer_gb": max_transfer_gb,
        "wave_count": len(wave_totals),
        "wave_totals_gb": {str(k): round(v, 1) for k, v in sorted(wave_totals.items())},
        "infeasible_groups": [
            {"group_id": g, "total_gb": final_total_gb[g], "members": final_members[g]}
            for g in infeasible_groups
        ],
        "group_count": len(final_members),
    }
    WAVE_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    WAVE_SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2))

    wave_lines = "\n".join(
        f"  - Wave {k}: {v:.0f} GB" for k, v in sorted(wave_totals.items())
    )
    infeasible_lines = "\n".join(
        f"  - {g}: {final_total_gb[g]:.0f} GB (exceeds {max_transfer_gb:.0f} GB limit, "
        f"indivisible dependency group, placed alone)"
        for g in infeasible_groups
    ) or "  none"

    return (
        "Wave construction complete.\n"
        f"   Groups scheduled : {len(final_members)}\n"
        f"   Waves created    : {len(wave_totals)}\n"
        f"{wave_lines}\n"
        f"   Infeasible (over-limit, indivisible) groups:\n{infeasible_lines}\n"
        f"   Summary written to {WAVE_SUMMARY_JSON}"
    )


@tool
def export_migration_plan() -> str:
    """Export the final migration plan (with FinOps, risk, and wave
    columns) to data/final_migration_plan.csv.

    Returns:
        A string confirming the save location and row count.
    """
    graph = get_graph()
    results = graph.query(
        "MATCH (s:Server) WHERE s.hostname IS NOT NULL "
        "RETURN s.id AS Server_ID, s.hostname AS Hostname, s.os AS OS, "
        "s.normalized_cpu AS Normalized_CPU, s.target_size AS Target_Size, "
        "s.risk_score AS Risk_Score, s.transfer_gb AS Transfer_GB, "
        "s.current_monthly_cost AS Current_Monthly_Cost, "
        "s.aws_monthly_cost AS AWS_Monthly_Cost, "
        "s.savings_amount AS Savings_Amount, s.savings_pct AS Savings_Pct, "
        "s.wcc_id AS WCC_Id, s.scc_id AS SCC_Id, s.community_id AS Community_Id, "
        "s.group_id AS Group_Id, s.wave_id AS Wave_ID, "
        "s.early_migration_requested AS Early_Migration_Requested, "
        "s.inbound_degree AS Inbound_Degree "
        "ORDER BY s.wave_id, s.id"
    )
    if not results:
        return "ERROR: No server data found to export."

    df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    return f"Final migration plan exported to {OUTPUT_CSV.absolute()} ({len(df)} rows)."


def build_agent():
    llm = get_llm()
    tools = [construct_and_pack_waves, export_migration_plan]
    system_message = SystemMessage(
        content=(
            "You are the Wave Planning Agent. Step 1: call "
            "`construct_and_pack_waves` with the confirmed max_transfer_gb "
            "from the run configuration. Step 2: once that finishes, call "
            "`export_migration_plan`. Step 3: summarize the number of "
            "waves and whether any groups were infeasible."
        )
    )
    return create_agent(llm, tools, system_prompt=system_message)


if __name__ == "__main__":
    max_gb = DEFAULT_MAX_TRANSFER_GB
    honor_early = True
    if DEFAULT_RUN_CONFIG_PATH.exists():
        cfg = json.loads(DEFAULT_RUN_CONFIG_PATH.read_text())
        max_gb = float(cfg.get("max_transfer_gb", DEFAULT_MAX_TRANSFER_GB))
        honor_early = bool(cfg.get("honor_early_migration_flags", True))

    try:
        agent = build_agent()
        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        f"Construct and pack migration waves using a "
                        f"{max_gb} GB limit (honor_early_migration="
                        f"{honor_early}), then export the final plan."
                    )
                )
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
        print(f"Agent invocation failed ({exc}). Running tools directly...")
        print("\n-- Step 1: Construct & pack waves --")
        print(construct_and_pack_waves.invoke({
            "max_transfer_gb": max_gb, "honor_early_migration": honor_early
        }))
        print("\n-- Step 2: Export plan --")
        print(export_migration_plan.invoke({}))
