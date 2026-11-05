"""
B1 - Greedy Rule-Only Baseline (Milestone 1, Section 5)
===========================================================
A simple deterministic improvement over B0 that is NOT community-aware:
outbound edges plus derived inbound metadata, SCC-safe topological order,
and deterministic first-fit-decreasing packing under the transfer-volume
limit. No LLM, no Neo4j, no community detection, no early-migration
prioritization, and no FinOps savings calculation - exactly the B1
definition from the Milestone 1 report. Used to test whether P1's
community-aware grouping actually improves on simple valid packing.

The packing unit here is the individual SCC (mostly single servers,
except real cycles), not the whole WCC and not a community - so a large
connected component gets chopped purely by topological readiness and
size, with no attempt to keep related servers together within a wave.

Usage:
    python b1_greedy_wave_agent.py --cleaned <cleaned.csv> \
        --max-transfer-gb 3000 --out-plan <plan.csv> --out-summary <summary.json>
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph_utils import build_group_graph, ensure_dag, topological_waves_first_fit  # noqa: E402

AWS_SIZING = [
    (2, 8, "t3.large"),
    (4, 16, "t3.xlarge"),
    (8, 32, "m5.2xlarge"),
]
AWS_DEFAULT = "m5.4xlarge"

LEGACY_MARKERS = ("2008", "RHEL 5", "6.4")
COMPLIANCE_TAGS = ("HIPAA", "PCI", "SOX")


def _pick_size(cpu: float, ram: float) -> str:
    for max_cpu, max_ram, instance in AWS_SIZING:
        if cpu <= max_cpu and ram <= max_ram:
            return instance
    return AWS_DEFAULT


def _risk(os_val: str, tags_val: str) -> str:
    is_legacy = any(m in str(os_val) for m in LEGACY_MARKERS)
    is_compliance = any(t in str(tags_val) for t in COMPLIANCE_TAGS)
    return "High" if (is_legacy or is_compliance) else "Low"


def _parse_edges(df: pd.DataFrame) -> list[tuple[str, str]]:
    edges = []
    for _, row in df.iterrows():
        raw = str(row.get("Outbound_Connections", ""))
        if not raw or raw in ("nan", "None", ""):
            continue
        for conn in raw.split("|"):
            parts = conn.strip().split(":")
            if len(parts) == 3:
                edges.append((row["Server_ID"], parts[0]))
    return edges


def run(cleaned_path: Path, max_transfer_gb: float, out_plan: Path, out_summary: Path) -> None:
    df = pd.read_csv(cleaned_path)
    df["Transfer_GB"] = df.get("Transfer_GB", df["Storage_GB"]).fillna(df["Storage_GB"])

    df["Target_Size"] = df.apply(
        lambda r: _pick_size(r["Allocated_vCPU"], r["Allocated_RAM_GB"]), axis=1
    )
    df["Risk_Score"] = df.apply(
        lambda r: _risk(r["OS_Version"], r.get("Business_Tags", "")), axis=1
    )

    edges = _parse_edges(df)
    known_ids = set(df["Server_ID"])
    edges = [(s, d) for s, d in edges if s in known_ids and d in known_ids]

    import networkx as nx
    server_graph = nx.DiGraph()
    server_graph.add_nodes_from(df["Server_ID"])
    server_graph.add_edges_from(edges)

    scc_of_server: dict[str, str] = {}
    for i, component in enumerate(nx.strongly_connected_components(server_graph)):
        scc_id = f"SCC{i}"
        for node in component:
            scc_of_server[node] = scc_id

    transfer_by_server = dict(zip(df["Server_ID"], df["Transfer_GB"]))
    scc_total_gb: dict[str, float] = {}
    scc_members: dict[str, list[str]] = {}
    for sid, scc_id in scc_of_server.items():
        scc_total_gb[scc_id] = scc_total_gb.get(scc_id, 0.0) + transfer_by_server.get(sid, 0.0)
        scc_members.setdefault(scc_id, []).append(sid)

    group_graph = build_group_graph(scc_of_server, edges)
    dag, merge_map = ensure_dag(group_graph)

    final_total_gb: dict[str, float] = {}
    final_members: dict[str, list[str]] = {}
    for scc_id, total in scc_total_gb.items():
        final_id = merge_map.get(scc_id, scc_id)
        final_total_gb[final_id] = final_total_gb.get(final_id, 0.0) + total
        final_members.setdefault(final_id, []).extend(scc_members[scc_id])

    # Deterministic first-fit-decreasing: larger groups offered first,
    # tie-broken by group id for reproducibility. No risk, savings, or
    # early-migration signal is used - that is the point of this baseline.
    def priority_key(group_id: str):
        return (-final_total_gb.get(group_id, 0.0), group_id)

    wave_of_group, infeasible_groups = topological_waves_first_fit(
        dag, final_total_gb, max_transfer_gb, priority_key
    )

    server_wave: dict[str, int] = {}
    for g, members in final_members.items():
        for m in members:
            server_wave[m] = wave_of_group[g]

    df["Wave_ID"] = df["Server_ID"].map(server_wave)
    df["SCC_Id"] = df["Server_ID"].map(scc_of_server)

    out_cols = [
        "Server_ID", "Hostname", "OS_Version", "Target_Size", "Risk_Score",
        "Transfer_GB", "Current_Monthly_Cost", "SCC_Id", "Wave_ID",
        "Early_Migration_Requested",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].sort_values(["Wave_ID", "Server_ID"]).to_csv(out_plan, index=False)

    wave_totals: dict[int, float] = {}
    for g, wid in wave_of_group.items():
        wave_totals[wid] = wave_totals.get(wid, 0.0) + final_total_gb[g]

    summary = {
        "system": "B1 (greedy rule-only, SCC-granularity, no community detection)",
        "max_transfer_gb": max_transfer_gb,
        "wave_count": len(wave_totals),
        "wave_totals_gb": {str(k): round(v, 1) for k, v in sorted(wave_totals.items())},
        "infeasible_groups": [
            {"group_id": g, "total_gb": final_total_gb[g], "members": final_members[g]}
            for g in infeasible_groups
        ],
        "group_count": len(final_members),
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2))

    print(f"B1 plan written to {out_plan} ({len(df)} rows, {len(wave_totals)} waves)")
    print(f"B1 summary written to {out_summary}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--max-transfer-gb", type=float, default=3000.0)
    parser.add_argument("--out-plan", required=True)
    parser.add_argument("--out-summary", required=True)
    args = parser.parse_args()
    run(Path(args.cleaned), args.max_transfer_gb, Path(args.out_plan), Path(args.out_summary))


if __name__ == "__main__":
    main()
