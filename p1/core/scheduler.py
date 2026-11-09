"""Group-level dependency DAG and capacity-constrained wave packing.

Shared by P1 and the B1 baseline so that the two systems differ only in the
packing *priority*, which is what the Milestone 1 comparison is meant to test.

Dependency convention (unchanged from the team's prototype): an edge A -> B
("A calls B") requires wave(A) <= wave(B). Same-wave placement is valid.

Milestone 2 fix: the prototype only let a group become "ready" after all its
predecessors were placed in an EARLIER wave, so a chain of n servers needed n
waves and every dependency was split. `pack_waves(allow_same_wave=True)` lets a
group join the wave its predecessors are in, while the packer re-checks
readiness after every placement. `allow_same_wave=False` reproduces the old
behaviour for the ablation study.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import networkx as nx


def stable_label(prefix: str, members) -> str:
    """Component label independent of database-internal IDs."""
    return f"{prefix}:{min(members)}"


def build_group_graph(group_of: dict[str, str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(sorted(set(group_of.values())))
    weights: dict[tuple[str, str], int] = defaultdict(int)
    for src, dst in edges:
        a, b = group_of.get(src), group_of.get(dst)
        if a is None or b is None or a == b:
            continue
        weights[(a, b)] += 1
    for (a, b), w in sorted(weights.items()):
        g.add_edge(a, b, weight=w)
    return g


def ensure_dag(group_graph: nx.DiGraph) -> tuple[nx.DiGraph, dict[str, str]]:
    """Collapse any residual cycle between groups into one merged group."""
    if nx.is_directed_acyclic_graph(group_graph):
        return group_graph, {n: n for n in group_graph.nodes}
    cond = nx.condensation(group_graph)
    label = {c: "+".join(sorted(cond.nodes[c]["members"])) for c in cond.nodes}
    mapping = {m: label[c] for c in cond.nodes for m in cond.nodes[c]["members"]}
    dag = nx.DiGraph()
    dag.add_nodes_from(sorted(label.values()))
    for u, v in cond.edges:
        if label[u] != label[v]:
            dag.add_edge(label[u], label[v])
    return dag, mapping


@dataclass
class PackContext:
    wave: int
    members: list[str] = field(default_factory=list)   # groups already in the current wave
    used_gb: float = 0.0


def pack_waves(
    dag: nx.DiGraph,
    group_gb: dict[str, float],
    max_gb: float,
    key_fn: Callable[[str, PackContext], tuple],
    allow_same_wave: bool = True,
    pre_assigned: dict[str, int] | None = None,
    on_place: Callable[[str, int], None] | None = None,
) -> tuple[dict[str, int], list[str]]:
    """Dependency-respecting, capacity-constrained, priority-ordered packing.

    Returns (wave index per group, groups that exceed max_gb on their own).
    An oversized group is indivisible (a real cycle), so it is placed alone in
    its own wave and reported instead of silently violating the limit.
    """
    wave_of: dict[str, int] = dict(pre_assigned or {})
    infeasible: list[str] = []
    preds = {n: set(dag.predecessors(n)) for n in dag.nodes}
    unscheduled = set(dag.nodes) - set(wave_of)
    w = (max(wave_of.values()) + 1) if wave_of else 0

    while unscheduled:
        ctx = PackContext(wave=w)
        dedicated = False
        while not dedicated:
            done = set(wave_of) if allow_same_wave else {g for g, wi in wave_of.items() if wi < w}
            ready = [g for g in unscheduled if preds[g] <= done]
            if not ready:
                break
            ready.sort(key=lambda g: key_fn(g, ctx))
            chosen = None
            for g in ready:
                size = group_gb.get(g, 0.0)
                if size > max_gb:
                    if not ctx.members:
                        chosen, dedicated = g, True
                        infeasible.append(g)
                        break
                    continue
                if ctx.used_gb + size <= max_gb:
                    chosen = g
                    break
            if chosen is None:
                break
            wave_of[chosen] = w
            unscheduled.discard(chosen)
            ctx.members.append(chosen)
            ctx.used_gb += group_gb.get(chosen, 0.0)
            if on_place:
                on_place(chosen, w)
            if not allow_same_wave and dedicated:
                break
        if not ctx.members:
            # Unreachable on a DAG; keep a safe exit instead of looping forever.
            raise RuntimeError(f"Wave packing stalled with {len(unscheduled)} groups unscheduled")
        w += 1
    return wave_of, infeasible


def cross_wave_edges(edges: list[tuple[str, str]], wave_of_server: dict[str, int]) -> int:
    return sum(1 for a, b in edges if a in wave_of_server and b in wave_of_server
               and wave_of_server[a] != wave_of_server[b])


def utilization_pct(wave_totals: dict[int, float], max_gb: float, exclude_waves: set[int] = frozenset()) -> float | None:
    """Mean fill of waves, excluding the final wave (by index) and waves that hold an
    over-limit indivisible group. Milestone 1 target: >= 60%."""
    waves = sorted(w for w in wave_totals if w not in exclude_waves)
    if len(waves) <= 1:
        return None
    body = waves[:-1]
    return round(sum(min(wave_totals[w], max_gb) for w in body) / (len(body) * max_gb) * 100, 1)
