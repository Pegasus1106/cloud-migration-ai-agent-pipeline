"""
Shared deterministic graph helpers used by wave_agent_p1.py and the B1
greedy baseline. Kept dependency-free of Neo4j/LLM so it can be unit
tested and reused in both the proposed system and the baseline.
"""

from collections import defaultdict

import networkx as nx


def build_group_graph(server_group: dict[str, str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Collapse a server-level edge list into a group-level directed graph
    using the given server_id -> group_id mapping. Self-loops (edges
    within the same group) are dropped since they do not constrain
    scheduling order between groups.
    """
    group_graph = nx.DiGraph()
    for group_id in set(server_group.values()):
        group_graph.add_node(group_id)

    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    for src, dst in edges:
        g_src = server_group.get(src)
        g_dst = server_group.get(dst)
        if g_src is None or g_dst is None or g_src == g_dst:
            continue
        edge_counts[(g_src, g_dst)] += 1

    for (g_src, g_dst), weight in edge_counts.items():
        group_graph.add_edge(g_src, g_dst, weight=weight)

    return group_graph


def ensure_dag(group_graph: nx.DiGraph) -> tuple[nx.DiGraph, dict[str, str]]:
    """If the group graph has cycles (can happen after community
    detection splits an oversized WCC into groups that still cross-call
    each other), collapse each remaining cycle into one merged group via
    strongly-connected-component condensation. Returns the resulting DAG
    and a mapping from original group_id to its final (possibly merged)
    group_id. This is the "verification and recovery" safety net: it
    guarantees a valid migration order always exists.
    """
    if nx.is_directed_acyclic_graph(group_graph):
        return group_graph, {n: n for n in group_graph.nodes}

    condensed = nx.condensation(group_graph)
    mapping: dict[str, str] = {}
    for node in condensed.nodes:
        members = condensed.nodes[node]["members"]
        merged_id = "+".join(sorted(members)) if len(members) > 1 else next(iter(members))
        for m in members:
            mapping[m] = merged_id

    dag = nx.DiGraph()
    for node in condensed.nodes:
        members = condensed.nodes[node]["members"]
        merged_id = "+".join(sorted(members)) if len(members) > 1 else next(iter(members))
        dag.add_node(merged_id)
    for u, v in condensed.edges:
        u_members = condensed.nodes[u]["members"]
        v_members = condensed.nodes[v]["members"]
        u_id = "+".join(sorted(u_members)) if len(u_members) > 1 else next(iter(u_members))
        v_id = "+".join(sorted(v_members)) if len(v_members) > 1 else next(iter(v_members))
        if u_id != v_id:
            dag.add_edge(u_id, v_id)

    return dag, mapping


def topological_waves_first_fit(
    dag: nx.DiGraph,
    group_total_gb: dict[str, float],
    max_transfer_gb: float,
    priority_key,
) -> tuple[dict[str, int], list[str]]:
    """Assign each group a wave_id (0-indexed) using dependency-respecting,
    capacity-constrained, priority-ordered first-fit packing.

    A group becomes "ready" once every predecessor already has a wave_id
    (any wave strictly before or equal to the one being filled). Each
    pass fills exactly one wave: ready groups are offered in priority
    order and packed first-fit until the capacity limit is reached, then
    the wave is closed and the next pass begins.

    Groups whose own total exceeds max_transfer_gb cannot be split
    further (they are already an indivisible SCC/community unit); each
    is placed alone in its own wave and returned in the infeasible list
    so the caller can report it instead of silently violating the limit.

    Returns:
        (wave_id map, list of infeasible group_ids)
    """
    wave_id: dict[str, int] = {}
    infeasible: list[str] = []
    unscheduled = set(dag.nodes)
    preds = {n: set(dag.predecessors(n)) for n in dag.nodes}

    current_wave = 0
    safety_limit = len(dag.nodes) * 2 + 10
    passes = 0

    while unscheduled:
        passes += 1
        if passes > safety_limit:
            # Should not happen on a true DAG; avoid an infinite loop and
            # surface the remainder as a single forced final wave.
            for n in unscheduled:
                wave_id[n] = current_wave
            break

        scheduled_ids = set(wave_id.keys())
        ready = {n for n in unscheduled if preds[n] <= scheduled_ids}
        if not ready:
            for n in unscheduled:
                wave_id[n] = current_wave
            break

        ordered_ready = sorted(ready, key=priority_key)
        capacity_used = 0.0
        placed_in_this_wave = []

        for group_id in ordered_ready:
            total = group_total_gb.get(group_id, 0.0)
            if total > max_transfer_gb:
                if placed_in_this_wave:
                    continue  # try this oversized group alone next wave
                wave_id[group_id] = current_wave
                unscheduled.discard(group_id)
                placed_in_this_wave.append(group_id)
                infeasible.append(group_id)
                break  # dedicate this wave solely to the oversized group
            if capacity_used + total <= max_transfer_gb:
                wave_id[group_id] = current_wave
                unscheduled.discard(group_id)
                capacity_used += total
                placed_in_this_wave.append(group_id)

        if not placed_in_this_wave:
            # Nothing fit this pass (shouldn't occur given the oversized
            # handling above); advance a wave to avoid spinning forever.
            current_wave += 1
            continue

        current_wave += 1

    return wave_id, infeasible
