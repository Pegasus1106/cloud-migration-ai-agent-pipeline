"""'Ask the plan' benchmark: dynamic Cypher generation accuracy, error rate, latency
(the stakeholder proposal's evaluation item for B0's GraphCypherQAChain).

Compares, on the same questions and the same plan in Neo4j:
  B0-style  LangChain GraphCypherQAChain (one query, default top_k=10), as used by the baseline
  P1        Q&A agent: guarded read-only tool, observes errors/empty results, retries, reports truncation
Gold answers are computed from the plan CSV with pandas.

  python eval/eval_qa.py [--model qwen3-235b-a22b]
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import DATASETS, RESULTS  # noqa: E402
from p1 import config  # noqa: E402
from p1.agents.qa_agent import ask  # noqa: E402
from p1.core.data import load_and_normalize  # noqa: E402
from p1.core.graphdb import GraphStore  # noqa: E402
from p1.llm import get_llm  # noqa: E402
from p1.state import RunConfig  # noqa: E402
from p1.supervisor import RunOptions, Supervisor  # noqa: E402
from p1.tracing import Tracer  # noqa: E402

QUESTIONS = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "qa_questions.jsonl"


def build_plan() -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Deterministic (rule-only) D1 plan, written into Neo4j for both systems to query."""
    out = config.RUNS_DIR / "eval" / "qa-plan"
    sup = Supervisor(DATASETS["D1"], RunConfig(confirmed=True, confirmed_by="qa benchmark"),
                     RunOptions(use_llm=False, out_dir=out), run_id="qa-plan")
    state = sup.run()
    assert state["graph_stats"]["engine"] == "neo4j-gds", "Q&A benchmark needs Neo4j running"
    _, edges, _ = load_and_normalize(DATASETS["D1"])
    return pd.read_csv(out / "plan.csv"), [(e["src"], e["dst"]) for e in edges]


def gold_answer(q: dict, plan: pd.DataFrame, edges) -> tuple[str, object]:
    p, t = q["params"], q["template"]
    by_host = plan.set_index("Hostname")
    if t == "count_wave":
        return "number", int((plan["Wave_ID"] == p["wave"]).sum())
    if t == "wave_of_host":
        return "number", int(by_host.loc[p["hostname"], "Wave_ID"])
    if t == "wave_gb":
        return "number", round(float(plan.loc[plan["Wave_ID"] == p["wave"], "Transfer_GB"].sum()), 2)
    if t == "high_risk_app":
        return "set", set(plan[(plan["App_Name"] == p["app"]) & (plan["Risk_Level"] == "High")]["Server_ID"])
    if t == "callers_of":
        target = by_host.loc[p["hostname"], "Server_ID"]
        return "set", {a for a, b in edges if b == target}
    if t == "instance_of":
        return "string", str(by_host.loc[p["hostname"], "Target_Size"])
    if t == "eol_count":
        return "number", int(plan["Risk_Factor_Codes"].fillna("").str.split("|").map(lambda c: "OS_EOL" in c).sum())
    if t == "app_cost":
        return "number", round(float(plan.loc[plan["App_Name"] == p["app"], "AWS_Monthly_Cost"].sum()), 2)
    if t == "early_list":
        return "set", set(plan[plan["Early_Migration_Requested"]]["Server_ID"])
    if t == "wave_count":
        return "number", int(plan["Wave_ID"].nunique())
    if t == "cycle_members":
        return "set", set(plan[plan["SCC_Size"] > 1]["Server_ID"])
    if t == "top_inbound":
        m = plan["Inbound_Degree"].max()
        return "set", set(plan[plan["Inbound_Degree"] == m]["Server_ID"])
    raise ValueError(t)


def _cells(rows) -> list:
    out = []
    for r in rows or []:
        vals = r.values() if isinstance(r, dict) else [r]
        for v in vals:
            out.extend(v if isinstance(v, list) else [v])
    return out


def _num_match(x, gold: float) -> bool:
    try:
        v = float(str(x).replace(",", "").replace("$", ""))
    except ValueError:
        return False
    return abs(v - gold) <= max(0.011, abs(gold) * 0.001)


def score(kind: str, gold, rows, answer: str, host_to_id: dict, all_ids: set) -> dict:
    cells = _cells(rows)
    if kind == "number":
        row_ok = any(_num_match(c, gold) for c in cells if not isinstance(c, bool))
        nums = re.findall(r"\d[\d,]*(?:\.\d+)?", answer or "")
        ans_ok = any(_num_match(n, gold) for n in nums)
        return {"rows_correct": row_ok, "answer_correct": ans_ok}
    if kind == "string":
        return {"rows_correct": any(str(c) == gold for c in cells), "answer_correct": gold in (answer or "")}
    found = {host_to_id.get(str(c), str(c)) for c in cells}
    found &= all_ids
    in_answer = {sid for sid in all_ids if re.search(rf"\b{re.escape(sid)}\b", answer or "")} | \
                {host_to_id[h] for h in host_to_id if re.search(rf"(?<![\w-]){re.escape(h)}(?![\w-])", answer or "")}
    return {"rows_correct": found == gold, "answer_correct": in_answer == gold,
            "recall": round(len(found & gold) / len(gold), 2) if gold else 1.0}


def run_b0_chain(question: str, model: str | None):
    """The baseline's Text2Cypher approach (GraphCypherQAChain, default top_k=10)."""
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
    from langchain_community.graphs import Neo4jGraph
    graph = Neo4jGraph(url=config.NEO4J_URL, username=config.NEO4J_USER, password=config.NEO4J_PASSWORD)
    chain = GraphCypherQAChain.from_llm(llm=get_llm(model), graph=graph, allow_dangerous_requests=True,
                                        return_intermediate_steps=True, verbose=False)
    t = time.time()
    try:
        r = chain.invoke({"query": question})
        steps = r.get("intermediate_steps") or []
        cypher = steps[0].get("query") if steps else None
        rows = steps[1].get("context") if len(steps) > 1 else []
        return {"answer": r.get("result", ""), "cypher": cypher, "rows": rows, "attempts": 1,
                "query_errors": 0, "error": None, "latency_ms": int((time.time() - t) * 1000)}
    except Exception as exc:  # noqa: BLE001 - a failed query is the measured outcome
        return {"answer": "", "cypher": None, "rows": [], "attempts": 1, "query_errors": 1,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}", "latency_ms": int((time.time() - t) * 1000)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--systems", default="B0-style,P1")
    a = ap.parse_args()
    plan, edges = build_plan()
    host_to_id = dict(zip(plan["Hostname"], plan["Server_ID"]))
    all_ids = set(plan["Server_ID"])
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines()]
    results = {s: [] for s in a.systems.split(",")}
    with GraphStore() as store:
        for q in questions:
            kind, gold = gold_answer(q, plan, edges)
            for system in results:
                if system == "P1":
                    r = ask(q["question"], store, Tracer(f"qa-{q['qid']}"), a.model)
                else:
                    r = run_b0_chain(q["question"], a.model)
                s = score(kind, gold, r["rows"], r["answer"], host_to_id, all_ids)
                results[system].append({"qid": q["qid"], "template": q["template"], "question": q["question"],
                                        "gold": sorted(gold) if isinstance(gold, set) else gold, **s,
                                        "executed": r["cypher"] is not None and not r.get("error"),
                                        "attempts": r["attempts"], "query_errors": r["query_errors"],
                                        "latency_ms": r["latency_ms"], "cypher": r["cypher"],
                                        "answer": (r["answer"] or "")[:300], "error": r.get("error")})
                print(f"{q['qid']} {system:<9} rows={'OK ' if s['rows_correct'] else 'BAD'} "
                      f"answer={'OK ' if s['answer_correct'] else 'BAD'} attempts={r['attempts']} "
                      f"{r['latency_ms'] / 1000:.1f}s", flush=True)
    summary = {}
    for system, rs in results.items():
        n = len(rs)
        total_queries = sum(r["attempts"] for r in rs)
        summary[system] = {
            "questions": n,
            "executed_pct": round(100 * sum(r["executed"] for r in rs) / n, 1),
            "rows_correct_pct": round(100 * sum(r["rows_correct"] for r in rs) / n, 1),
            "answer_correct_pct": round(100 * sum(r["answer_correct"] for r in rs) / n, 1),
            "query_error_rate_pct": round(100 * sum(r["query_errors"] for r in rs) / max(1, total_queries), 1),
            "mean_attempts": round(total_queries / n, 2),
            "latency_sec_mean": round(statistics.mean(r["latency_ms"] for r in rs) / 1000, 1),
            "latency_sec_p95": round(sorted(r["latency_ms"] for r in rs)[int(0.95 * (n - 1))] / 1000, 1),
            "by_template_rows_correct": {t: f"{sum(r['rows_correct'] for r in rs if r['template'] == t)}/"
                                            f"{sum(r['template'] == t for r in rs)}"
                                         for t in sorted({r['template'] for r in rs})},
        }
    tag = (a.model or config.AGENT_MODEL).replace("/", "_")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"qa_eval_{tag}.json").write_text(json.dumps({"model": a.model or config.AGENT_MODEL,
                                                             "summary": summary, "results": results}, indent=2,
                                                            default=str))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
