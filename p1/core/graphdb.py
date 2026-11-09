"""Neo4j + Graph Data Science store (stakeholder baseline technology).

Changes from the prototype:
- Batched UNWIND ingestion (fast at 500+ servers) via the official driver.
- WCC / SCC / Louvain results are streamed back and relabelled by the smallest
  Server_ID in each component, so plans no longer depend on database-internal
  IDs (the prototype produced 30 vs 33 waves for identical input).
- Louvain runs single-threaded for determinism.
- A read-only query path (READ access mode + Cypher guard) for the Q&A agent.
- A file lock serializes runs that share one Neo4j instance.
"""

import fcntl
import threading
from collections import defaultdict
from contextlib import contextmanager

import pandas as pd
from neo4j import GraphDatabase, READ_ACCESS

from p1 import config
from p1.core.cypher_guard import check_read_only
from p1.core.scheduler import stable_label

SCHEMA_DESCRIPTION = """\
Node label :Server (one node per server in the current migration plan). Properties:
  id (e.g. 'SRV-001'), hostname, app (application name), os, cpu (vCPU), ram (GB), storage (GB),
  transfer_gb, business_tags (e.g. 'PCI-DSS'), normalized_cpu (%), current_monthly_cost (USD),
  early_migration_requested (bool), audit_flag (bool: discovery data was incomplete),
  inbound_degree (how many servers call it), outbound_degree,
  wcc_id, scc_id, scc_size (servers in its dependency cycle; 1 = no cycle), community_id,
  target_size (AWS instance type), aws_monthly_cost (USD), savings_amount (USD/month),
  risk_level ('Low' | 'Medium' | 'High'), risk_score (int), risk_factors (list of codes such as
  'OS_EOL', 'OS_EOL_SOON', 'COMPLIANCE', 'HIGH_INBOUND', 'IN_CYCLE', 'LARGE_DATA', 'DATA_GAPS', 'EXTERNAL'),
  wave_id (int, 0 = first wave), group_id, pilot (bool), early_related (bool)
Relationship (:Server)-[:COMMUNICATES_WITH {port, bandwidth}]->(:Server)
  Direction: caller -> callee. "Servers that depend on X" = (s)-[:COMMUNICATES_WITH]->(x).
"""


_LOCAL = threading.local()


@contextmanager
def neo4j_lock():
    """Cross-process lock for the single Neo4j instance (re-entrant within a thread),
    so a run's load-then-query sequence cannot be interleaved with another run's reset."""
    depth = getattr(_LOCAL, "depth", 0)
    if depth:
        _LOCAL.depth = depth + 1
        try:
            yield
        finally:
            _LOCAL.depth -= 1
        return
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RUNS_DIR / ".neo4j.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        _LOCAL.depth = 1
        try:
            yield
        finally:
            _LOCAL.depth = 0
            fcntl.flock(fh, fcntl.LOCK_UN)


class GraphStore:
    def __init__(self, url: str = config.NEO4J_URL, user: str = config.NEO4J_USER,
                 password: str = config.NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(url, auth=(user, password), notifications_min_severity="OFF")

    def close(self) -> None:
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def ping(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:  # noqa: BLE001 - connectivity probe
            return False

    def _run(self, cypher: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(cypher, parameters_=params, database_="neo4j")
        return [r.data() for r in records]

    # ---- ingestion ---------------------------------------------------------
    def reset(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")
        self._run("CREATE CONSTRAINT server_id IF NOT EXISTS FOR (s:Server) REQUIRE s.id IS UNIQUE")

    def ingest(self, servers: pd.DataFrame, edges: list[dict], batch: int = 500) -> dict:
        rows = [{
            "id": r.Server_ID, "hostname": r.Hostname, "os": r.OS_Version, "cpu": float(r.Allocated_vCPU),
            "ram": float(r.Allocated_RAM_GB), "storage": float(r.Storage_GB), "app": r.App_Name,
            "business_tags": r.Business_Tags, "normalized_cpu": float(r.Normalized_CPU_Pct),
            "audit_flag": bool(r.requires_manual_audit), "transfer_gb": float(r.Transfer_GB),
            "current_monthly_cost": float(r.Current_Monthly_Cost),
            "early_migration_requested": bool(r.Early_Migration_Requested),
        } for r in servers.itertuples()]
        for i in range(0, len(rows), batch):
            self._run("UNWIND $rows AS r MERGE (s:Server {id: r.id}) SET s += r", rows=rows[i:i + batch])
        for i in range(0, len(edges), batch):
            self._run(
                "UNWIND $rows AS e MATCH (a:Server {id: e.src}) MATCH (b:Server {id: e.dst}) "
                "MERGE (a)-[r:COMMUNICATES_WITH {port: e.port}]->(b) SET r.bandwidth = e.bandwidth",
                rows=edges[i:i + batch])
        self._run(
            "MATCH (s:Server) "
            "SET s.inbound_degree = COUNT { (:Server)-[:COMMUNICATES_WITH]->(s) }, "
            "    s.outbound_degree = COUNT { (s)-[:COMMUNICATES_WITH]->(:Server) }")
        counts = self._run("MATCH (s:Server) RETURN count(s) AS n")[0]["n"]
        rels = self._run("MATCH ()-[r:COMMUNICATES_WITH]->() RETURN count(r) AS n")[0]["n"]
        return {"nodes": counts, "relationships": rels}

    def degrees(self) -> dict[str, tuple[int, int]]:
        rows = self._run("MATCH (s:Server) RETURN s.id AS id, s.inbound_degree AS i, s.outbound_degree AS o")
        return {r["id"]: (int(r["i"] or 0), int(r["o"] or 0)) for r in rows}

    # ---- graph algorithms (GDS) ------------------------------------------
    def _project(self, name: str, orientation: str) -> None:
        self._run("CALL gds.graph.drop($n, false) YIELD graphName RETURN graphName", n=name)
        self._run(f"CALL gds.graph.project($n, 'Server', {{COMMUNICATES_WITH: {{orientation: '{orientation}'}}}}) "
                  "YIELD graphName RETURN graphName", n=name)

    def _drop(self, name: str) -> None:
        self._run("CALL gds.graph.drop($n, false) YIELD graphName RETURN graphName", n=name)

    def components(self, max_gb: float, transfer: dict[str, float]) -> tuple[dict[str, dict], dict]:
        """WCC (portfolios), SCC (indivisible cycles), Louvain (only inside WCCs
        whose transfer volume exceeds max_gb). Labels are stable (min Server_ID)."""
        und, nat = "p1_undirected", "p1_natural"
        self._project(und, "UNDIRECTED")
        self._project(nat, "NATURAL")
        try:
            wcc_raw = self._run(f"CALL gds.wcc.stream('{und}') YIELD nodeId, componentId "
                                "RETURN gds.util.asNode(nodeId).id AS id, componentId AS c")
            scc_raw = self._run(f"CALL gds.scc.stream('{nat}') YIELD nodeId, componentId "
                                "RETURN gds.util.asNode(nodeId).id AS id, componentId AS c")
            wcc = _relabel(wcc_raw, "WCC")
            scc = _relabel(scc_raw, "SCC")
            wcc_members: dict[str, list[str]] = defaultdict(list)
            for sid, label in wcc.items():
                wcc_members[label].append(sid)
            wcc_gb = {lbl: sum(transfer.get(s, 0.0) for s in mem) for lbl, mem in wcc_members.items()}
            oversized = sorted(lbl for lbl, gb in wcc_gb.items() if gb > max_gb)
            community = {sid: "COM:" + wcc[sid].split(":", 1)[1] for sid in wcc}
            louvain_count = 0
            if oversized:
                lv_raw = self._run(f"CALL gds.louvain.stream('{und}', {{concurrency: 1}}) "
                                   "YIELD nodeId, communityId "
                                   "RETURN gds.util.asNode(nodeId).id AS id, communityId AS c")
                groups: dict[tuple[str, int], list[str]] = defaultdict(list)
                for r in lv_raw:
                    if wcc[r["id"]] in oversized:
                        groups[(wcc[r["id"]], r["c"])].append(r["id"])
                for mem in groups.values():
                    for sid in mem:
                        community[sid] = stable_label("COM", mem)
                louvain_count = len(groups)
        finally:
            self._drop(und)
            self._drop(nat)

        scc_size: dict[str, int] = defaultdict(int)
        for label in scc.values():
            scc_size[label] += 1
        per = {sid: {"WCC_Id": wcc[sid], "SCC_Id": scc[sid], "SCC_Size": scc_size[scc[sid]],
                     "Community_Id": community[sid]} for sid in wcc}
        self._run("UNWIND $rows AS r MATCH (s:Server {id: r.id}) "
                  "SET s.wcc_id = r.wcc, s.scc_id = r.scc, s.scc_size = r.size, s.community_id = r.com",
                  rows=[{"id": s, "wcc": p["WCC_Id"], "scc": p["SCC_Id"], "size": p["SCC_Size"],
                         "com": p["Community_Id"]} for s, p in per.items()])
        cycles = sorted(({"scc_id": lbl, "members": sorted(s for s, l2 in scc.items() if l2 == lbl)}
                         for lbl, n in scc_size.items() if n > 1), key=lambda c: (-len(c["members"]), c["scc_id"]))
        stats = {
            "wcc_count": len(wcc_members),
            "scc_count": len(scc_size),
            "cycles": cycles,
            "oversized_wccs": [{"wcc_id": lbl, "total_gb": round(wcc_gb[lbl], 1),
                                "servers": len(wcc_members[lbl])} for lbl in oversized],
            "louvain_communities_in_oversized_wccs": louvain_count,
            "community_count": len(set(community.values())),
        }
        return per, stats

    # ---- plan write-back + read-only queries ------------------------------
    def write_plan(self, rows: list[dict], run_id: str, batch: int = 500) -> None:
        for i in range(0, len(rows), batch):
            self._run("UNWIND $rows AS r MATCH (s:Server {id: r.id}) SET s += r.props", rows=rows[i:i + batch])
        self._run("MERGE (m:PlanMeta {key: 'current'}) SET m.run_id = $run_id", run_id=run_id)

    def current_plan(self) -> str | None:
        rows = self._run("MATCH (m:PlanMeta {key: 'current'}) RETURN m.run_id AS run_id")
        return rows[0]["run_id"] if rows else None

    def run_read(self, cypher: str, params: dict | None = None, limit: int = 200) -> tuple[list[dict], bool]:
        ok, reason = check_read_only(cypher)
        if not ok:
            raise PermissionError(reason)
        with self.driver.session(database="neo4j", default_access_mode=READ_ACCESS) as session:
            result = session.run(cypher, params or {})
            rows, truncated = [], False
            for i, record in enumerate(result):
                if i >= limit:
                    truncated = True
                    break
                rows.append(record.data())
            result.consume()
        return rows, truncated


def _relabel(rows: list[dict], prefix: str) -> dict[str, str]:
    groups: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        groups[r["c"]].append(r["id"])
    return {sid: stable_label(prefix, mem) for mem in groups.values() for sid in mem}
