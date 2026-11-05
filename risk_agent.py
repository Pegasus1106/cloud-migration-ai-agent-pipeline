"""
Module 5 - Risk Analysis Agent (P1, bug fix)
================================================
Same purpose as the baseline risk_agent.py: classify every server as
High/Low risk from legacy OS markers and compliance tags, and write the
result back to Neo4j.

Fix applied here (see Milestone 2, Section 5 failure analysis): the
baseline's `update_risk_scores` tool declared `risk_updates: list[dict]`.
The model sometimes emitted a Python-repr STRING of that list (single
quotes) instead of an actual JSON array, which failed Pydantic validation
twice in a row and caused the baseline run to silently skip the Neo4j
write (risk_score was never set, confirmed by the empty Risk_Score column
in the baseline's exported CSV and a GDS "property key does not exist"
warning later in wave_agent.py).

The fix: accept `risk_updates` as either a real list[dict] or a string
(JSON or Python-repr), and parse it defensively with json.loads first and
ast.literal_eval as a fallback, before writing. This keeps the LLM's
output format unconstrained while making the deterministic write robust.
"""

import ast
import json
import sys

from langchain.agents import create_agent
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool, tool

from common import get_graph, get_llm

LEGACY_MARKERS = ("2008", "RHEL 5", "6.4")
COMPLIANCE_TAGS = ("HIPAA", "PCI", "SOX")


def _build_cypher_chain():
    return GraphCypherQAChain.from_llm(
        llm=get_llm(),
        graph=get_graph(),
        verbose=True,
        allow_dangerous_requests=True,
        top_k=100,
    )


def _run_cypher_reader(query: str) -> str:
    chain = _build_cypher_chain()
    result = chain.invoke({"query": query})
    return result.get("result", "No answer returned.")


database_reader = Tool(
    name="Database_Reader",
    func=_run_cypher_reader,
    description=(
        "Translates natural language into Cypher queries to read the "
        "Neo4j graph. Use this to find server IDs, OS versions, and tags."
    ),
)


def _parse_risk_updates(risk_updates) -> list[dict]:
    """Defensively parse risk_updates regardless of whether the model
    passed a real list, a JSON string, or a Python-repr string."""
    if isinstance(risk_updates, list):
        return risk_updates
    if isinstance(risk_updates, str):
        try:
            parsed = json.loads(risk_updates)
        except (json.JSONDecodeError, TypeError):
            parsed = ast.literal_eval(risk_updates)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Could not parse risk_updates into a list: {risk_updates!r}")


@tool
def update_risk_scores(risk_updates) -> str:
    """Write risk scores back to Neo4j Server nodes.

    Args:
        risk_updates: A list of dicts, each with 'id' (Server_ID) and
                      'risk' ('High' or 'Low'). Also accepts a JSON or
                      Python-repr string of such a list (parsed
                      defensively) to tolerate model formatting drift.
                      Example: [{"id": "SRV-001", "risk": "High"}, ...]

    Returns:
        A confirmation string with counts, or an error string.
    """
    try:
        updates = _parse_risk_updates(risk_updates)
    except Exception as exc:
        return f"ERROR: could not parse risk_updates ({exc}). Resend as a JSON array."

    graph = get_graph()
    graph.query(
        "UNWIND $updates AS update "
        "MATCH (s:Server {id: update.id}) "
        "SET s.risk_score = update.risk",
        params={"updates": updates},
    )

    high = sum(1 for u in updates if u.get("risk") == "High")
    low = sum(1 for u in updates if u.get("risk") == "Low")
    return (
        f"Risk scores updated in Neo4j.\n"
        f"   Total updated : {len(updates)}\n"
        f"   High risk     : {high}\n"
        f"   Low risk      : {low}"
    )


def _run_deterministic_fallback():
    """Fallback that performs the risk analysis without the LLM at all."""
    graph = get_graph()

    print("\n-- Step 1: Fetching servers --")
    rows = graph.query(
        "MATCH (s:Server) RETURN s.id AS id, s.os AS os, s.business_tags AS tags ORDER BY s.id"
    )
    print(f"   Found {len(rows)} servers.")

    print("\n-- Step 2: Analysing risk --")
    updates = []
    for r in rows:
        os_val = str(r.get("os", ""))
        tags_val = str(r.get("tags", ""))
        is_legacy = any(m in os_val for m in LEGACY_MARKERS)
        is_compliance = any(t in tags_val for t in COMPLIANCE_TAGS)
        risk = "High" if (is_legacy or is_compliance) else "Low"
        updates.append({"id": r["id"], "risk": risk})

    high = [u for u in updates if u["risk"] == "High"]
    low = [u for u in updates if u["risk"] == "Low"]
    print(f"   High risk: {len(high)} servers")
    print(f"   Low risk : {len(low)} servers")

    print("\n-- Step 3: Saving to Neo4j --")
    print(update_risk_scores.invoke({"risk_updates": updates}))


SYSTEM_PROMPT = (
    "You are the Security & Risk Analysis Agent.\n\n"
    "Step 1: Use the `Database_Reader` tool to ask for the "
    "'id, os, and business_tags of all servers in the database'.\n"
    "Step 2: Analyze the results. If the OS contains '2008', '5', or "
    "'6.4' (Legacy), OR if the tags contain 'HIPAA', 'PCI', or 'SOX' "
    "(Compliance), assign a risk of 'High'. Otherwise, assign 'Low'.\n"
    "Step 3: Call the `update_risk_scores` tool with a JSON array "
    "argument named risk_updates, e.g. "
    '{"risk_updates": [{"id": "SRV-001", "risk": "High"}, '
    '{"id": "SRV-002", "risk": "Low"}]}. Pass it as a real JSON array, '
    "not as a quoted string."
)


def build_agent():
    llm = get_llm()
    tools = [database_reader, update_risk_scores]
    return create_agent(llm, tools, system_prompt=SystemMessage(content=SYSTEM_PROMPT))


if __name__ == "__main__":
    try:
        agent = build_agent()
        result = agent.invoke({
            "messages": [
                HumanMessage(content="Analyze all servers for legacy and compliance risks, then update their risk scores.")
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

        # Safety net: even if the agent's final message claims success,
        # verify the write actually landed in Neo4j (this is exactly the
        # failure mode observed in the baseline run). If risk_score is
        # still unset for any server, run the deterministic fallback.
        graph = get_graph()
        unset = graph.query(
            "MATCH (s:Server) WHERE s.risk_score IS NULL RETURN count(s) AS n"
        )
        if unset and unset[0]["n"] > 0:
            print(
                f"\nVerification: {unset[0]['n']} servers still missing "
                "risk_score after the agent run. Running deterministic fallback..."
            )
            _run_deterministic_fallback()
    except Exception as exc:
        print(f"Agent invocation failed ({exc}). Running deterministic fallback...")
        _run_deterministic_fallback()
