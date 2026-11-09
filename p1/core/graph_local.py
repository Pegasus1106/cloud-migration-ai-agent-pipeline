"""networkx fallback for the graph stage when Neo4j/GDS is not reachable.

Same outputs and stable labels as GraphStore.components(); Louvain uses a fixed
seed. The supervisor records when this fallback is used.
"""

from collections import defaultdict

import networkx as nx

from p1.core.scheduler import stable_label


def components_local(server_ids: list[str], edges: list[tuple[str, str]], transfer: dict[str, float],
                     max_gb: float) -> tuple[dict[str, dict], dict, dict[str, tuple[int, int]]]:
    g = nx.DiGraph()
    g.add_nodes_from(sorted(server_ids))
    g.add_edges_from(edges)
    wcc, scc, community = {}, {}, {}
    wcc_members: dict[str, list[str]] = {}
    for comp in nx.weakly_connected_components(g):
        label = stable_label("WCC", comp)
        wcc_members[label] = sorted(comp)
        for s in comp:
            wcc[s] = label
            community[s] = "COM:" + label.split(":", 1)[1]
    scc_size: dict[str, int] = defaultdict(int)
    for comp in nx.strongly_connected_components(g):
        label = stable_label("SCC", comp)
        for s in comp:
            scc[s] = label
        scc_size[label] = len(comp)
    wcc_gb = {lbl: sum(transfer.get(s, 0.0) for s in mem) for lbl, mem in wcc_members.items()}
    oversized = sorted(lbl for lbl, gb in wcc_gb.items() if gb > max_gb)
    louvain = 0
    for lbl in oversized:
        sub = g.subgraph(wcc_members[lbl]).to_undirected()
        for com in nx.community.louvain_communities(sub, seed=42):
            louvain += 1
            for s in com:
                community[s] = stable_label("COM", com)
    per = {s: {"WCC_Id": wcc[s], "SCC_Id": scc[s], "SCC_Size": scc_size[scc[s]], "Community_Id": community[s]}
           for s in g.nodes}
    cycles = sorted(({"scc_id": lbl, "members": sorted(s for s in g.nodes if scc[s] == lbl)}
                     for lbl, n in scc_size.items() if n > 1), key=lambda c: (-len(c["members"]), c["scc_id"]))
    stats = {"wcc_count": len(wcc_members), "scc_count": len(scc_size), "cycles": cycles,
             "oversized_wccs": [{"wcc_id": lbl, "total_gb": round(wcc_gb[lbl], 1),
                                 "servers": len(wcc_members[lbl])} for lbl in oversized],
             "louvain_communities_in_oversized_wccs": louvain, "community_count": len(set(community.values()))}
    degrees = {s: (g.in_degree(s), g.out_degree(s)) for s in g.nodes}
    return per, stats, degrees
