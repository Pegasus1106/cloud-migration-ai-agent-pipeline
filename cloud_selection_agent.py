"""
Module 3 – Cloud Platform Selection Agent (LLM-Generated Cypher)
=================================================================
A LangChain-based agent that uses GraphCypherQAChain to let the LLM
dynamically generate Cypher queries against the Neo4j dependency graph,
analyse the results, and select the best target cloud platform.
"""

from langchain.agents import create_agent
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from common import get_graph as _get_graph
from common import get_llm as _get_llm


# ── Tool 1: query_neo4j (LLM-generated Cypher) ─────────────────────────────
@tool
def query_neo4j(question: str) -> str:
    """Ask a natural-language question about the server ecosystem.
    The LLM will automatically generate the appropriate Cypher query,
    execute it against Neo4j, and return a summarised answer.

    Args:
        question: A natural-language question about the servers, e.g.
                  'What is the OS distribution?' or
                  'Which servers have the highest CPU usage?'

    Returns:
        The LLM-generated answer based on the Neo4j query results.
    """
    graph = _get_graph()
    llm = _get_llm()

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
    )

    result = chain.invoke({"query": question})

    # Build a detailed response that includes the generated Cypher
    generated_cypher = ""
    if "intermediate_steps" in result:
        steps = result["intermediate_steps"]
        if len(steps) > 0:
            generated_cypher = steps[0].get("query", "")

    answer = result.get("result", "No answer returned.")

    output_parts = []
    if generated_cypher:
        output_parts.append(f"Generated Cypher: {generated_cypher}")
    output_parts.append(f"Answer: {answer}")

    return "\n".join(output_parts)


# ── Tool 2: set_target_cloud (deterministic write) ──────────────────────────
@tool
def set_target_cloud(cloud_name: str, rationale: str) -> str:
    """Save the chosen target cloud platform and rationale to Neo4j as a
    Project node. This is a deterministic write operation.

    Args:
        cloud_name: The target cloud platform (e.g. 'AWS' or 'AZURE').
        rationale:  A 1-sentence explanation for the choice.

    Returns:
        A confirmation string.
    """
    graph = _get_graph()

    graph.query(
        "MERGE (p:Project {id: 'Migration_Plan'}) "
        "SET p.target_cloud = $cloud_name, "
        "    p.rationale = $rationale",
        params={
            "cloud_name": cloud_name,
            "rationale": rationale,
        },
    )

    return (
        f"✅ Target cloud saved to Neo4j.\n"
        f"   Platform  : {cloud_name}\n"
        f"   Rationale : {rationale}"
    )


# ── LLM & Agent setup ───────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an enterprise Cloud Migration Architect. Your job is to "
    "decide whether to migrate this data center to 'AWS' or 'AZURE'.\n\n"
    "Step 1: Use the `query_neo4j` tool with the question "
    "'What is the distribution of operating systems across all servers?' "
    "to discover the OS landscape.\n"
    "Step 2: Analyse the results. If the environment is dominated by "
    "Microsoft technologies (Windows Server, SQL Server), choose 'AZURE' "
    "for hybrid licensing benefits. If it is dominated by Linux/Open "
    "Source (Ubuntu, RHEL, CentOS), choose 'AWS'.\n"
    "Step 3: Use the `set_target_cloud` tool to save your decision and "
    "a 1-sentence rationale."
)


def build_agent():
    """Construct and return the cloud selection agent."""

    llm = _get_llm()
    tools = [query_neo4j, set_target_cloud]

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
                        "Evaluate our server ecosystem and select the "
                        "best target cloud platform."
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
        print("\n── Step 1: Query OS Distribution (LLM-generated Cypher) ──")
        os_result = query_neo4j.invoke({
            "question": "What is the distribution of operating systems across all servers?"
        })
        print(os_result)
        print("\n── Step 2: Making decision ──")
        os_lower = os_result.lower()
        windows_count = os_lower.count("windows") + os_lower.count("sql server")
        linux_count = sum(
            os_lower.count(d)
            for d in ["ubuntu", "rhel", "centos", "oracle linux"]
        )
        if windows_count > linux_count:
            cloud, reason = "AZURE", (
                "Windows Server and SQL Server dominate the environment — "
                "Azure Hybrid Benefit provides significant licensing savings."
            )
        else:
            cloud, reason = "AWS", (
                "Linux/open-source OSes dominate the environment — AWS "
                "offers the broadest Linux ecosystem and pricing flexibility."
            )
        print(f"   Decision: {cloud}")
        print(f"   Rationale: {reason}")
        print("\n── Step 3: Saving to Neo4j ──")
        save_result = set_target_cloud.invoke({
            "cloud_name": cloud,
            "rationale": reason,
        })
        print(save_result)
