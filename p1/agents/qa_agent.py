"""'Ask the plan' Q&A agent (GraphRAG / Text2Cypher).

Agentic retrieval loop: the model writes Cypher, calls a guarded read-only tool,
observes rows or the database error, rewrites the query if needed, and answers
from the rows. Addresses the stakeholder's evaluation item "error rate and
latency of dynamic Cypher generation" and the two prototype retrieval failures:
results are never silently cut to 10 rows (truncation is reported to the model),
and writes are impossible (regex guard + READ_ACCESS session).
"""

import time
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from p1.agents.common import dumps, last_ai_text, run_config
from p1.core.graphdb import SCHEMA_DESCRIPTION, GraphStore
from p1.llm import get_llm
from p1.tracing import Tracer

PROMPT_VERSION = "qa-v2.0"
SYSTEM_PROMPT = f"""You answer questions about a cloud-migration plan stored in Neo4j.
Graph schema:
{SCHEMA_DESCRIPTION}
Procedure:
1. Write ONE read-only Cypher query and call `run_cypher`. Return explicit properties
   (e.g. s.id, s.hostname, s.wave_id), never whole nodes.
2. If the tool returns an error or an empty result that looks wrong, fix the query and try again
   (at most 3 queries in total).
3. Answer in 1-3 sentences using only the returned rows. If rows were truncated, say so.
Examples:
- "How many servers are in wave 2?" -> MATCH (s:Server) WHERE s.wave_id = 2 RETURN count(s) AS servers
- "Which servers depend on hr-db-01?" -> MATCH (c:Server)-[:COMMUNICATES_WITH]->(x:Server {{hostname: 'hr-db-01'}})
  RETURN c.id, c.hostname ORDER BY c.id
- "High-risk servers with end-of-life OS?" -> MATCH (s:Server) WHERE s.risk_level = 'High' AND 'OS_EOL' IN s.risk_factors
  RETURN s.id, s.hostname, s.os ORDER BY s.id
"""
ROW_LIMIT = 200
ROWS_TO_MODEL = 60


@dataclass
class QASession:
    store: GraphStore
    queries: list[dict] = field(default_factory=list)


def _tools(session: QASession):
    @tool
    def run_cypher(query: str) -> str:
        """Run one read-only Cypher query against the migration-plan graph and return rows as JSON."""
        t = time.time()
        try:
            rows, truncated = session.store.run_read(query, limit=ROW_LIMIT)
        except PermissionError as exc:
            session.queries.append({"cypher": query, "status": "rejected", "error": str(exc)})
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - surface DB errors to the model for self-correction
            session.queries.append({"cypher": query, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            return f"ERROR: {type(exc).__name__}: {str(exc)[:500]}"
        session.queries.append({"cypher": query, "status": "ok", "rows": rows, "truncated": truncated,
                                "latency_ms": int((time.time() - t) * 1000)})
        shown = rows[:ROWS_TO_MODEL]
        return dumps({"row_count": len(rows), "truncated_at_limit": truncated,
                      "rows_shown": len(shown), "rows": shown})

    return [run_cypher]


def ask(question: str, store: GraphStore, tracer: Tracer | None = None, model: str | None = None) -> dict:
    session = QASession(store)
    t = time.time()
    answer, error = "", None
    try:
        agent = create_agent(get_llm(model), _tools(session), system_prompt=SYSTEM_PROMPT)
        result = agent.invoke({"messages": [HumanMessage(content=question)]},
                              config=run_config(tracer, "qa_agent", PROMPT_VERSION, recursion_limit=12))
        answer = last_ai_text(result["messages"])
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    last_ok = next((q for q in reversed(session.queries) if q["status"] == "ok"), None)
    return {
        "question": question,
        "answer": answer,
        "cypher": last_ok["cypher"] if last_ok else (session.queries[-1]["cypher"] if session.queries else None),
        "rows": last_ok["rows"] if last_ok else [],
        "attempts": len(session.queries),
        "query_errors": sum(q["status"] != "ok" for q in session.queries),
        "latency_ms": int((time.time() - t) * 1000),
        "error": error,
        "queries": session.queries,
    }
