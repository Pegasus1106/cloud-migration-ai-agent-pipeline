"""
Module 2b - Graph Agent (P1, new in this milestone)
======================================================
Derives inbound dependency view and the three graph structures the P1
wave planner needs:
  - Inbound adjacency / inbound degree (reverse of the outbound-only
    baseline view) without creating duplicate reverse relationships.
  - Weakly Connected Components (WCC) to separate disconnected portfolios.
  - Strongly Connected Components (SCC) to identify cycles that must be
    migrated together as one indivisible unit.
  - Louvain community detection, used only to subdivide a WCC whose total
    Transfer_GB exceeds the configured wave limit (oversized component).

All of this is deterministic graph algorithm work (Neo4j GDS); the LLM
is used only to read back and explain the resulting structure, consistent
with the Milestone 1 design ("deterministic tools remain authoritative
for graph algorithms").
"""

import sys

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from common import DEFAULT_MAX_TRANSFER_GB, get_graph, get_llm


@tool
def derive_inbound_view() -> str:
    """Compute inbound/outbound degree for every server from the existing
    directed COMMUNICATES_WITH edges (no reverse edges are created - this
    only aggregates edge direction that already exists), and flag any
    edge endpoints that were referenced but never appeared as a source
    row (missing/placeholder nodes).

    Returns:
        A summary of inbound-degree coverage and any missing endpoints.
    """
    graph = get_graph()

    graph.query(
        "MATCH (s:Server) "
        "OPTIONAL MATCH (:Server)-[in_edge:COMMUNICATES_WITH]->(s) "
        "WITH s, count(in_edge) AS indeg "
        "SET s.inbound_degree = indeg"
    )
    graph.query(
        "MATCH (s:Server) "
        "OPTIONAL MATCH (s)-[out_edge:COMMUNICATES_WITH]->(:Server) "
        "WITH s, count(out_edge) AS outdeg "
        "SET s.outbound_degree = outdeg"
    )

    missing = graph.query(
        "MATCH (s:Server) WHERE s.hostname IS NULL RETURN s.id AS id"
    )
    missing_ids = [r["id"] for r in missing]

    top_inbound = graph.query(
        "MATCH (s:Server) WHERE s.hostname IS NOT NULL "
        "RETURN s.id AS id, s.hostname AS hostname, s.inbound_degree AS indeg "
        "ORDER BY indeg DESC LIMIT 5"
    )

    lines = "\n".join(
        f"  - {r['id']} ({r['hostname']}): inbound_degree={r['indeg']}"
        for r in top_inbound
    )

    return (
        "Inbound view derived from existing outbound edges "
        "(no duplicate reverse edges created).\n"
        f"   Missing/placeholder endpoints: {missing_ids or 'none'}\n"
        f"   Top servers by inbound dependency impact:\n{lines}"
    )


@tool
def analyze_graph_components(max_transfer_gb: float = DEFAULT_MAX_TRANSFER_GB) -> str:
    """Compute WCC (disconnected portfolios), SCC (indivisible cycles), and -
    only for any WCC whose combined Transfer_GB exceeds max_transfer_gb -
    Louvain community detection to produce smaller candidate groups.
    Writes wcc_id, scc_id, and community_id back to every Server node.

    Args:
        max_transfer_gb: The configured per-wave transfer-volume limit
                          used only to decide which WCCs need subdivision.

    Returns:
        A summary of component counts, oversized components, and cycles.
    """
    graph = get_graph()

    for name in ("p1WccGraph", "p1SccGraph", "p1LouvainGraph"):
        try:
            graph.query(f"CALL gds.graph.drop('{name}')")
        except Exception:
            pass

    # --- WCC: undirected, separates disconnected portfolios ---
    graph.query(
        "CALL gds.graph.project('p1WccGraph', 'Server', "
        "{COMMUNICATES_WITH: {orientation: 'UNDIRECTED'}})"
    )
    wcc_result = graph.query(
        "CALL gds.wcc.write('p1WccGraph', {writeProperty: 'wcc_id'}) "
        "YIELD componentCount"
    )
    wcc_count = wcc_result[0]["componentCount"] if wcc_result else 0
    graph.query("CALL gds.graph.drop('p1WccGraph')")

    # --- SCC: directed, collapses cycles into indivisible units ---
    graph.query(
        "CALL gds.graph.project('p1SccGraph', 'Server', "
        "{COMMUNICATES_WITH: {orientation: 'NATURAL'}})"
    )
    scc_result = graph.query(
        "CALL gds.scc.write('p1SccGraph', {writeProperty: 'scc_id'}) "
        "YIELD componentCount"
    )
    scc_count = scc_result[0]["componentCount"] if scc_result else 0
    graph.query("CALL gds.graph.drop('p1SccGraph')")

    cyclic = graph.query(
        "MATCH (s:Server) WITH s.scc_id AS scc, collect(s.id) AS members "
        "WHERE size(members) > 1 "
        "RETURN scc AS scc_id, members, size(members) AS size "
        "ORDER BY size DESC"
    )

    # --- WCC totals, identify oversized components ---
    wcc_totals = graph.query(
        "MATCH (s:Server) "
        "WITH s.wcc_id AS wcc_id, sum(s.transfer_gb) AS total_gb, count(s) AS n "
        "RETURN wcc_id, total_gb, n ORDER BY total_gb DESC"
    )
    oversized = [r for r in wcc_totals if (r["total_gb"] or 0) > max_transfer_gb]

    # --- Louvain community detection (only matters for oversized WCCs,
    #     but computed over the whole graph for simplicity) ---
    if oversized:
        graph.query(
            "CALL gds.graph.project('p1LouvainGraph', 'Server', "
            "{COMMUNICATES_WITH: {orientation: 'UNDIRECTED'}})"
        )
        louvain_result = graph.query(
            "CALL gds.louvain.write('p1LouvainGraph', {writeProperty: 'community_id'}) "
            "YIELD communityCount"
        )
        community_count = louvain_result[0]["communityCount"] if louvain_result else 0
        graph.query("CALL gds.graph.drop('p1LouvainGraph')")
    else:
        graph.query("MATCH (s:Server) SET s.community_id = s.wcc_id")
        community_count = wcc_count

    oversized_lines = "\n".join(
        f"  - WCC {r['wcc_id']}: {r['total_gb']:.0f} GB across {r['n']} servers "
        f"(limit {max_transfer_gb:.0f} GB -> needs community split)"
        for r in oversized
    )
    cycle_lines = "\n".join(
        f"  - SCC {r['scc_id']}: {r['size']} servers in a cycle {r['members']}"
        for r in cyclic
    )

    return (
        "Graph component analysis complete.\n"
        f"   WCC (disconnected portfolios): {wcc_count}\n"
        f"   SCC (cyclic/indivisible groups): {scc_count}\n"
        f"   Cycles detected (SCC size > 1):\n{cycle_lines or '  none'}\n"
        f"   Oversized WCCs needing community split (> {max_transfer_gb:.0f} GB):\n"
        f"{oversized_lines or '  none'}\n"
        f"   Louvain communities written: {community_count}"
    )


def build_agent():
    llm = get_llm()
    tools = [derive_inbound_view, analyze_graph_components]
    system_message = SystemMessage(
        content=(
            "You are the Graph Agent. Step 1: call `derive_inbound_view` "
            "to compute inbound dependency impact from the existing "
            "outbound edges. Step 2: call `analyze_graph_components` to "
            "compute WCC, SCC, and (where needed) community detection. "
            "Step 3: briefly explain what you found, especially any "
            "oversized components or cycles."
        )
    )
    return create_agent(llm, tools, system_prompt=system_message)


if __name__ == "__main__":
    max_gb = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAX_TRANSFER_GB
    try:
        agent = build_agent()
        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        "Derive the inbound dependency view and analyze the "
                        f"graph's components using a wave transfer limit of "
                        f"{max_gb} GB."
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
        print("\n-- Step 1: Inbound view --")
        print(derive_inbound_view.invoke({}))
        print("\n-- Step 2: Component analysis --")
        print(analyze_graph_components.invoke({"max_transfer_gb": max_gb}))
