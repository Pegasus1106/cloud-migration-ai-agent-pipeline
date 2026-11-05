"""
Module 4 – Server Sizing Agent
================================
A LangChain-based agent that fetches the target cloud platform from
Neo4j, applies a sizing matrix to every Server node, and writes the
recommended instance type back to the graph.
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from common import get_graph as _get_graph
from common import get_llm as _get_llm

# ── Sizing matrices ─────────────────────────────────────────────────────────
AWS_SIZING = [
    (2, 8, "t3.large"),
    (4, 16, "t3.xlarge"),
    (8, 32, "m5.2xlarge"),
]
AWS_DEFAULT = "m5.4xlarge"

AZURE_SIZING = [
    (2, 8, "Standard_D2s_v3"),
    (4, 16, "Standard_D4s_v3"),
    (8, 32, "Standard_D8s_v3"),
]
AZURE_DEFAULT = "Standard_D16s_v3"


def _pick_size(cpu: float, ram: float, matrix: list[tuple], default: str) -> str:
    """Walk the sizing matrix and return the first matching instance type."""
    for max_cpu, max_ram, instance in matrix:
        if cpu <= max_cpu and ram <= max_ram:
            return instance
    return default


# ── Tool 1: get_target_cloud ────────────────────────────────────────────────
@tool
def get_target_cloud() -> str:
    """Look up the chosen target cloud platform from the Neo4j Project node.

    Returns:
        The cloud platform name (e.g. 'AWS' or 'AZURE'), or an error
        string if no Project node exists.
    """
    graph = _get_graph()

    results = graph.query(
        "MATCH (p:Project) RETURN p.target_cloud AS cloud"
    )

    if not results or not results[0].get("cloud"):
        return "ERROR: No target cloud found. Run Module 3 first."

    cloud = results[0]["cloud"]
    return f"Target cloud platform: {cloud}"


# ── Tool 2: right_size_servers ──────────────────────────────────────────────
@tool
def right_size_servers(cloud_provider: str) -> str:
    """Apply a sizing matrix to all servers and write the recommended
    instance type back to Neo4j.

    Args:
        cloud_provider: The cloud platform — must be 'AWS' or 'AZURE'.

    Returns:
        A summary string with the sizing breakdown.
    """
    provider = cloud_provider.strip().upper()

    if provider == "AWS":
        matrix, default = AWS_SIZING, AWS_DEFAULT
    elif provider == "AZURE":
        matrix, default = AZURE_SIZING, AZURE_DEFAULT
    else:
        return f"ERROR: Unsupported cloud provider '{cloud_provider}'. Use 'AWS' or 'AZURE'."

    graph = _get_graph()

    # Fetch all servers
    rows = graph.query(
        "MATCH (s:Server) RETURN s.id AS id, s.cpu AS cpu, s.ram AS ram"
    )

    if not rows:
        return "ERROR: No Server nodes found in the graph."

    # Apply sizing matrix
    updates = []
    size_counts: dict[str, int] = {}
    for r in rows:
        cpu = float(r.get("cpu", 0))
        ram = float(r.get("ram", 0))
        size = _pick_size(cpu, ram, matrix, default)
        updates.append({"id": r["id"], "size": size})
        size_counts[size] = size_counts.get(size, 0) + 1

    # Bulk write back to Neo4j
    graph.query(
        "UNWIND $updates AS update "
        "MATCH (s:Server {id: update.id}) "
        "SET s.target_size = update.size",
        params={"updates": updates},
    )

    # Build summary
    breakdown = "\n".join(
        f"  • {size}: {count} servers"
        for size, count in sorted(size_counts.items(), key=lambda x: -x[1])
    )

    return (
        f"✅ Right-sizing complete for {provider}.\n"
        f"   Total servers sized: {len(updates)}\n"
        f"   Breakdown:\n{breakdown}"
    )


# ── Agent system prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are the Capacity Planning Agent. Your job is to right-size "
    "the environment.\n\n"
    "Step 1: Use the `get_target_cloud` tool to find out which cloud "
    "platform was selected.\n"
    "Step 2: Pass that cloud platform name into the `right_size_servers` "
    "tool to size all servers in the database and save the results."
)


# ── LLM & Agent setup ───────────────────────────────────────────────────────
def build_agent():
    """Construct and return the server sizing agent."""

    llm = _get_llm()
    tools = [get_target_cloud, right_size_servers]

    agent = create_agent(
        llm,
        tools,
        system_prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    return agent


# ── Main entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        agent = build_agent()
        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        "Check our target cloud and right-size all "
                        "servers in the database."
                    )
                )
            ]
        })
        print("\n─── Agent Response ───")
        for msg in result["messages"]:
            role = getattr(msg, "type", "unknown")
            if hasattr(msg, "content") and msg.content:
                print(f"\n[{role}] {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[tool_call] {tc['name']}({tc['args']})")
    except Exception as exc:
        print(f"⚠️  Agent invocation failed ({exc}). Running tools directly…")
        print("\n── Step 1: Get target cloud ──")
        cloud_result = get_target_cloud.invoke({})
        print(cloud_result)
        # Extract cloud name from result
        cloud = cloud_result.split(":")[-1].strip() if ":" in cloud_result else "AWS"
        print(f"\n── Step 2: Right-size for {cloud} ──")
        size_result = right_size_servers.invoke({"cloud_provider": cloud})
        print(size_result)
