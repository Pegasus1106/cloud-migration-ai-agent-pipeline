"""Constraint-parsing benchmark (Milestone 1 target: >= 85% on >= 20 labeled requests).

Requests were written by gpt-oss-120b from labels fixed in code
(data/benchmark/constraint_prompts.jsonl). The Constraint Agent (a different
model) must either propose the labeled configuration or, for ambiguous
requests, ask a clarifying question instead of guessing.

  python eval/eval_constraints.py [--model qwen3-235b-a22b] [--limit 10]
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import DATASETS, RESULTS  # noqa: E402
from p1 import config  # noqa: E402
from p1.agents.constraint_agent import ConstraintAgent  # noqa: E402
from p1.core.data import load_and_normalize  # noqa: E402
from p1.tracing import Tracer  # noqa: E402

PROMPTS = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "constraint_prompts.jsonl"


def _early_servers(df, app_names, server_ids) -> set[str]:
    apps = {a.lower() for a in app_names}
    return set(df[df["App_Name"].str.lower().isin(apps)]["Server_ID"]) | set(server_ids)


def score(case: dict, turn: dict, df) -> dict:
    if case["kind"] == "ambiguous":
        ok = turn["status"] == "clarification"
        return {"correct": ok, "fields": {"asked_clarification": ok}}
    if turn["status"] != "proposed":
        return {"correct": False, "false_clarification": True,
                "fields": {"max_transfer_gb": False, "target_cloud": False, "early": False, "pilot_wave": False}}
    cfg, gold = turn["proposed"], case["gold"]
    flagged = set(df.loc[df["Early_Migration_Requested"], "Server_ID"])
    got = _early_servers(df, cfg.early_app_names, cfg.early_server_ids) | flagged
    want = _early_servers(df, gold["early_app_names"], []) | flagged
    fields = {"max_transfer_gb": abs(cfg.max_transfer_gb - gold["max_transfer_gb"]) < 0.5,
              "target_cloud": cfg.target_cloud == gold["target_cloud"],
              "early": got == want, "pilot_wave": cfg.pilot_wave == gold["pilot_wave"]}
    return {"correct": all(fields.values()), "false_clarification": False, "fields": fields,
            "early_extra": sorted(got - want), "early_missing": sorted(want - got)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cases = [json.loads(line) for line in PROMPTS.read_text().splitlines()]
    if a.limit:
        cases = cases[:a.limit]
    frames = {ds: load_and_normalize(DATASETS[ds])[0] for ds in {c["dataset"] for c in cases}}
    rows = []
    for c in cases:
        df = frames[c["dataset"]]
        tracer = Tracer(f"constraint-eval-{c['case_id']}")
        t = time.time()
        try:
            turn = ConstraintAgent(df, tracer, a.model).send(c["request"])
            s = score(c, turn, df)
            proposed = turn["proposed"].model_dump(include={"max_transfer_gb", "target_cloud", "early_app_names",
                                                            "early_server_ids", "pilot_wave"}) if turn["proposed"] else None
            err = None
        except Exception as exc:  # noqa: BLE001
            s, proposed, err, turn = {"correct": False, "fields": {}}, None, f"{type(exc).__name__}: {exc}", {"reply": ""}
        summ = tracer.summary()
        rows.append({"case_id": c["case_id"], "kind": c["kind"], "request": c["request"], "gold": c["gold"],
                     "proposed": proposed, "reply": turn.get("reply", "")[:400], **s, "error": err,
                     "latency_sec": round(time.time() - t, 1), "llm_calls": summ["llm_calls"],
                     "tool_calls": summ["tool_calls"], "tool_errors": summ["tool_errors"]})
        print(f"{c['case_id']:<22} {'OK ' if s['correct'] else 'BAD'} {rows[-1]['latency_sec']:>5}s "
              f"{'' if s['correct'] else json.dumps(s.get('fields'))}", flush=True)
    clear = [r for r in rows if r["kind"] == "clear"]
    amb = [r for r in rows if r["kind"] == "ambiguous"]
    field_names = ["max_transfer_gb", "target_cloud", "early", "pilot_wave"]
    summary = {
        "model": a.model or config.AGENT_MODEL, "cases": len(rows), "clear": len(clear), "ambiguous": len(amb),
        "overall_accuracy_pct": round(100 * sum(r["correct"] for r in rows) / len(rows), 1),
        "clear_exact_match_pct": round(100 * sum(r["correct"] for r in clear) / max(1, len(clear)), 1),
        "field_accuracy_pct": {f: round(100 * sum(bool(r["fields"].get(f)) for r in clear) / max(1, len(clear)), 1)
                               for f in field_names},
        "ambiguous_clarified_pct": round(100 * sum(r["correct"] for r in amb) / max(1, len(amb)), 1),
        "false_clarification_pct": round(100 * sum(bool(r.get("false_clarification")) for r in clear)
                                         / max(1, len(clear)), 1),
        "latency_sec_mean": round(statistics.mean(r["latency_sec"] for r in rows), 1),
        "llm_calls_mean": round(statistics.mean(r["llm_calls"] for r in rows), 1),
        "self_corrections_total": sum(r["tool_errors"] for r in rows),
        "errors": sum(bool(r["error"]) for r in rows),
        "milestone1_target_pct": 85.0,
    }
    tag = (a.model or config.AGENT_MODEL).replace("/", "_")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"constraint_eval_{tag}.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
