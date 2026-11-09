"""Wave planning policies.

P1 (proposed): SCC groups are the only indivisible unit. Among ready groups the
packer prefers, in order:
  1. confirmed early-migration requests and their prerequisites (ancestors),
  2. groups with the most dependency edges into the wave being filled (affinity:
     fewer cut dependencies),
  3. groups whose community is already partly migrated (finish what was started),
  4. lower risk level (stakeholder: low-risk work first),
  5. higher projected savings, then a stable ID tie-break.
An optional pilot wave (wave 0) holds whole, small, low-risk, isolated apps.

B1 (baseline, Milestone 1 definition): same SCC groups and packer, first-fit
decreasing by size, no early/risk/community/savings signal.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field

import networkx as nx

from p1.core.risk import LEVEL_RANK
from p1.core.scheduler import PackContext, build_group_graph, ensure_dag, pack_waves, stable_label
from p1.state import RunConfig


@dataclass
class PlanResult:
    policy: str
    packing: str
    wave_of_server: dict[str, int]
    group_of_server: dict[str, str]
    group_members: dict[str, list[str]]
    group_gb: dict[str, float]
    wave_totals: dict[int, float]
    infeasible: list[dict]
    pilot: dict = field(default_factory=dict)
    early_related_servers: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["wave_totals"] = {str(k): round(v, 1) for k, v in sorted(self.wave_totals.items())}
        return d


def resolve_early_ids(servers: list[dict], cfg: RunConfig) -> set[str]:
    ids = set()
    if cfg.honor_early_migration_flags:
        ids |= {s["Server_ID"] for s in servers if s.get("Early_Migration_Requested")}
    known = {s["Server_ID"] for s in servers}
    ids |= {i for i in cfg.early_server_ids if i in known}
    apps = {a.strip().lower() for a in cfg.early_app_names}
    if apps:
        ids |= {s["Server_ID"] for s in servers if str(s.get("App_Name", "")).strip().lower() in apps}
    return ids


def _prepare(servers: list[dict], edges: list[tuple[str, str]], group_of: dict[str, str]):
    graph = build_group_graph(group_of, edges)
    dag, merge = ensure_dag(graph)
    final_of = {sid: merge.get(g, g) for sid, g in group_of.items()}
    members: dict[str, list[str]] = defaultdict(list)
    gb: dict[str, float] = defaultdict(float)
    by_id = {s["Server_ID"]: s for s in servers}
    for sid, g in final_of.items():
        members[g].append(sid)
        gb[g] += float(by_id[sid].get("Transfer_GB") or 0.0)
    for g in members:
        members[g].sort()
    return dag, final_of, dict(members), dict(gb)


def _finish(policy, packing, dag, final_of, members, gb, wave_of_group, infeasible_groups,
            pilot=None, early_related=None, excluded=None) -> PlanResult:
    wave_of_server = {sid: wave_of_group[g] for sid, g in final_of.items()}
    totals: dict[int, float] = defaultdict(float)
    for g, w in wave_of_group.items():
        totals[w] += gb[g]
    infeasible = [{"group_id": g, "members": members[g], "total_gb": round(gb[g], 1),
                   "wave": wave_of_group[g]} for g in infeasible_groups]
    return PlanResult(policy, packing, wave_of_server, final_of, members, gb, dict(totals), infeasible,
                      pilot or {}, sorted(early_related or []), sorted(excluded or []))


def plan_p1(servers: list[dict], edges: list[tuple[str, str]], cfg: RunConfig,
            allow_same_wave: bool = True, excluded_ids: set[str] | None = None) -> PlanResult:
    excluded_ids = set(excluded_ids or ())
    servers = [s for s in servers if s["Server_ID"] not in excluded_ids]
    known = {s["Server_ID"] for s in servers}
    edges = [(a, b) for a, b in edges if a in known and b in known]
    by_id = {s["Server_ID"]: s for s in servers}

    group_of = {s["Server_ID"]: s.get("SCC_Id") or stable_label("SCC", [s["Server_ID"]]) for s in servers}
    dag, final_of, members, gb = _prepare(servers, edges, group_of)

    # Per-group aggregates (stable, ID-independent).
    risk = {g: max((by_id[m].get("Risk_Level") or "Low" for m in ms), key=lambda lv: LEVEL_RANK.get(lv, 0))
            for g, ms in members.items()}
    savings = {g: sum(float(by_id[m].get("Savings_Amount") or 0.0) for m in ms) for g, ms in members.items()}
    community = {}
    for g, ms in members.items():
        votes: dict[str, int] = defaultdict(int)
        for m in ms:
            votes[str(by_id[m].get("Community_Id") or by_id[m].get("WCC_Id") or g)] += 1
        community[g] = min(votes, key=lambda c: (-votes[c], c))
    wcc = {g: str(by_id[ms[0]].get("WCC_Id") or g) for g, ms in members.items()}

    early_ids = resolve_early_ids(servers, cfg)
    early_groups = {final_of[s] for s in early_ids if s in final_of}
    early_related = set(early_groups)
    for g in early_groups:
        early_related |= nx.ancestors(dag, g)

    # Optional pilot wave: whole, small, low-risk, isolated apps (connected components).
    pre_assigned: dict[str, int] = {}
    pilot = {"enabled": cfg.pilot_wave, "servers": [], "apps": [], "reason": ""}
    if cfg.pilot_wave:
        comp_groups: dict[str, list[str]] = defaultdict(list)
        for g in members:
            comp_groups[wcc[g]].append(g)
        candidates = []
        for comp, gs in comp_groups.items():
            srv = [m for g in gs for m in members[g]]
            total = sum(gb[g] for g in gs)
            if (all(risk[g] == "Low" for g in gs) and not (set(gs) & early_related)
                    and total <= min(cfg.pilot_max_gb, cfg.max_transfer_gb)
                    and len(srv) <= cfg.pilot_max_servers):
                risk_sum = sum(float(by_id[m].get("Risk_Score") or 0) for m in srv)
                candidates.append((risk_sum, total, len(srv), comp, gs, srv))
        used_gb, used_n = 0.0, 0
        for risk_sum, total, n, comp, gs, srv in sorted(candidates):
            if used_n + n <= cfg.pilot_max_servers and used_gb + total <= min(cfg.pilot_max_gb, cfg.max_transfer_gb):
                for g in gs:
                    pre_assigned[g] = 0
                used_gb += total
                used_n += n
                pilot["servers"].extend(sorted(srv))
                pilot["apps"].extend(sorted({str(by_id[m].get("App_Name")) for m in srv}))
        pilot["total_gb"] = round(used_gb, 1)
        if not pre_assigned:
            pilot["reason"] = "No whole application was small, low-risk, and isolated enough for a pilot."

    # Dynamic priority state.
    neighbors: dict[str, dict[str, int]] = defaultdict(dict)
    for u, v, d in dag.edges(data=True):
        w = int(d.get("weight", 1))
        neighbors[u][v] = neighbors[u].get(v, 0) + w
        neighbors[v][u] = neighbors[v].get(u, 0) + w
    remaining: dict[str, int] = defaultdict(int)
    for g in members:
        remaining[community[g]] += 1
    started: set[str] = set()

    def on_place(g: str, _w: int) -> None:
        remaining[community[g]] -= 1
        started.add(community[g])

    for g in pre_assigned:
        on_place(g, 0)

    def key_fn(g: str, ctx: PackContext) -> tuple:
        affinity = sum(neighbors[g].get(m, 0) for m in ctx.members)
        c = community[g]
        continuing = 0 if (c in started and remaining[c] > 0) else 1
        return (0 if g in early_related else 1, -affinity, continuing,
                LEVEL_RANK.get(risk[g], 0), -savings[g], g)

    wave_of_group, infeasible = pack_waves(dag, gb, cfg.max_transfer_gb, key_fn,
                                           allow_same_wave=allow_same_wave,
                                           pre_assigned=pre_assigned, on_place=on_place)
    early_related_servers = [m for g in early_related for m in members[g]]
    return _finish("P1", "same-wave" if allow_same_wave else "strict-earlier", dag, final_of, members, gb,
                   wave_of_group, infeasible, pilot, early_related_servers, excluded_ids)


def plan_b1(servers: list[dict], edges: list[tuple[str, str]], max_gb: float,
            allow_same_wave: bool = True) -> PlanResult:
    known = {s["Server_ID"] for s in servers}
    edges = [(a, b) for a, b in edges if a in known and b in known]
    g = nx.DiGraph()
    g.add_nodes_from(sorted(known))
    g.add_edges_from(edges)
    group_of = {}
    for comp in nx.strongly_connected_components(g):
        label = stable_label("SCC", comp)
        for sid in comp:
            group_of[sid] = label
    dag, final_of, members, gb = _prepare(servers, edges, group_of)

    def key_fn(grp: str, _ctx: PackContext) -> tuple:
        return (-gb[grp], grp)

    wave_of_group, infeasible = pack_waves(dag, gb, max_gb, key_fn, allow_same_wave=allow_same_wave)
    return _finish("B1", "same-wave" if allow_same_wave else "strict-earlier", dag, final_of, members, gb,
                   wave_of_group, infeasible)
