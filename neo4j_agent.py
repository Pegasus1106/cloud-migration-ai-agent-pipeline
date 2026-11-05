"""
Module 2 - Graph Ingestion Agent (P1)
========================================
Reads cleaned server discovery data and ingests it into Neo4j as a
dependency graph. Extends the baseline by also storing Transfer_GB,
Current_Monthly_Cost, and Early_Migration_Requested as node properties so
downstream agents (graph, FinOps, wave) can use them.
"""

import sys
from pathlib import Path

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from common import get_graph, get_llm

DEFAULT_INPUT = Path("data/cleaned_samples.csv")


@tool
def ingest_graph_data(input_path: str = str(DEFAULT_INPUT)) -> str:
    """Read cleaned server discovery CSV and ingest it into Neo4j as a
    dependency graph with Server nodes and COMMUNICATES_WITH edges.
    Outbound edges are loaded exactly once per recorded connection; no
    reverse edges are created here (inbound adjacency is derived later by
    the Graph Agent via query direction, not by duplicating edges).

    Args:
        input_path: Path to the cleaned discovery CSV.

    Returns:
        A human-readable summary of how many nodes and edges were created.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        return f"ERROR: Input file '{input_path}' not found."

    df = pd.read_csv(input_path)
    total_rows = len(df)

    graph = get_graph()
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Server) REQUIRE s.id IS UNIQUE")

    node_count = 0
    for _, row in df.iterrows():
        graph.query(
            """
            MERGE (s:Server {id: $server_id})
            SET s.hostname       = $hostname,
                s.os             = $os,
                s.cpu            = $cpu,
                s.ram            = $ram,
                s.storage        = $storage,
                s.app            = $app,
                s.business_tags  = $tags,
                s.normalized_cpu = $normalized_cpu,
                s.audit_flag     = $audit_flag,
                s.transfer_gb    = $transfer_gb,
                s.current_monthly_cost = $current_monthly_cost,
                s.early_migration_requested = $early_migration_requested
            """,
            params={
                "server_id": row["Server_ID"],
                "hostname": row["Hostname"],
                "os": str(row["OS_Version"]),
                "cpu": float(row["Allocated_vCPU"]),
                "ram": float(row["Allocated_RAM_GB"]),
                "storage": float(row["Storage_GB"]),
                "app": str(row["App_Name"]),
                "tags": str(row.get("Business_Tags", "")),
                "normalized_cpu": float(row["Normalized_CPU_Pct"]),
                "audit_flag": bool(row.get("requires_manual_audit", False)),
                "transfer_gb": float(row.get("Transfer_GB", row["Storage_GB"])),
                "current_monthly_cost": float(row.get("Current_Monthly_Cost", 0.0)),
                "early_migration_requested": bool(row.get("Early_Migration_Requested", False)),
            },
        )
        node_count += 1

    edge_count = 0
    missing_endpoints = set()
    known_ids = set(df["Server_ID"].astype(str))
    for _, row in df.iterrows():
        raw_connections = str(row.get("Outbound_Connections", ""))
        if not raw_connections or raw_connections in ("", "nan", "None"):
            continue

        source_id = row["Server_ID"]
        for conn in raw_connections.split("|"):
            parts = conn.strip().split(":")
            if len(parts) != 3:
                continue

            target_id, port, bandwidth = parts
            if target_id not in known_ids:
                missing_endpoints.add(target_id)
            graph.query(
                """
                MERGE (source:Server {id: $source_id})
                MERGE (target:Server {id: $target_id})
                MERGE (source)-[r:COMMUNICATES_WITH {port: $port,
                                                      bandwidth: $bandwidth}]->(target)
                """,
                params={
                    "source_id": source_id,
                    "target_id": target_id,
                    "port": port,
                    "bandwidth": bandwidth,
                },
            )
            edge_count += 1

    graph.refresh_schema()

    summary = (
        f"Graph Ingestion Complete\n"
        f"   Rows processed        : {total_rows}\n"
        f"   Server nodes created  : {node_count}\n"
        f"   Edges created         : {edge_count}\n"
        f"   Missing edge endpoints: {sorted(missing_endpoints) or 'none'}\n"
        f"   Neo4j Browser         : http://localhost:7475"
    )
    return summary


def build_agent():
    llm = get_llm()
    tools = [ingest_graph_data]
    system_message = SystemMessage(
        content=(
            "You are a Cloud Migration assistant responsible for graph "
            "ingestion. Use the available tools to load cleaned server "
            "discovery data into a Neo4j dependency graph."
        )
    )
    return create_agent(llm, tools, system_prompt=system_message)


if __name__ == "__main__":
    input_arg = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_INPUT)
    try:
        agent = build_agent()
        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        f"Load the cleaned server discovery data from "
                        f"'{input_arg}' into the local Neo4j database. "
                        "Create Server nodes and COMMUNICATES_WITH edges."
                    )
                )
            ]
        })
        print("\n--- Agent Response ---")
        print(result["messages"][-1].content)
    except Exception as exc:
        print(f"Agent invocation failed ({exc}). Running tool directly...")
        print(ingest_graph_data.invoke({"input_path": input_arg}))
