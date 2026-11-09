"""Milestone 2 evaluation: B0 vs B1 vs P1 (+ ablations) on D1, D2, D3_30/100/500.

Resumable: each run writes its own folder and is skipped if already complete.

  python eval/run_eval.py --datasets D1,D2,D3_30,D3_100,D3_500 --b0-repo ../migration-planning-ai
  python eval/run_eval.py --datasets D1 --p1-repeats 10            # completion-reliability study
  python eval/run_eval.py --summarize-only                         # rebuild tables from saved runs

Systems
  B0           stakeholder baseline, run unmodified (baselines/run_b0.py)
  B1           greedy first-fit-decreasing baseline (Milestone 1 definition)
  P1           agentic system (Constraint config confirmed, Repair Agent, review drafting, explainers)
Ablations
  P1-rules     P1 with every LLM step replaced by its deterministic fallback
  P1-norepair  P1 without the Repair Agent
  P1-strict    P1 with the prototype's packing (no same-wave dependencies)
  B1-strict    B1 with the prototype's packing
  P1-<model>   P1 with a different agent model (--ablation-model)
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import DATASETS, MAX_GB, RESULTS, score_plan  # noqa: E402
from baselines import b1_greedy_wave_agent as b1  # noqa: E402
from baselines.run_b0 import run_b0  # noqa: E402
from p1 import config  # noqa: E402
from p1.state import RunConfig  # noqa: E402
from p1.supervisor import RunOptions, Supervisor  # noqa: E402

RUNS = config.RUNS_DIR / "eval"
PLANS = RESULTS / "plans"


def _done(path: Path) -> bool:
    return (path / "result.json").exists()


def _save(path: Path, row: dict) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.json").write_text(json.dumps(row, indent=2, default=str))
    return row


def run_b0_row(ds: str, b0_repo: Path) -> dict | None:
    folder = RUNS / ds / "B0"
    if _done(folder):
        return json.loads((folder / "result.json").read_text())
    out = PLANS / f"{ds}_B0.csv"
    info = run_b0(b0_repo, DATASETS[ds].resolve(), out)
    row = {"dataset": ds, "system": "B0", "repeat": 1, "latency_sec": info["duration_sec"],
           "completed": bool(info["plan_produced"]), "exit_code": info["exit_code"]}
    if info["plan_produced"]:
        plan = pd.read_csv(out)
        row.update(score_plan(plan, ds, "B0")["metrics"])
    return _save(folder, row)


def run_b1_row(ds: str, strict: bool) -> dict:
    name = "B1-strict" if strict else "B1"
    folder = RUNS / ds / name
    if _done(folder):
        return json.loads((folder / "result.json").read_text())
    t = time.time()
    plan_path, summ_path = PLANS / f"{ds}_{name}.csv", PLANS / f"{ds}_{name}_summary.json"
    summary = b1.run(DATASETS[ds], MAX_GB, plan_path, summ_path, allow_same_wave=not strict)
    row = {"dataset": ds, "system": name, "repeat": 1, "latency_sec": round(time.time() - t, 2), "completed": True}
    row.update(score_plan(pd.read_csv(plan_path), ds, name, summary["infeasible_groups"])["metrics"])
    return _save(folder, row)


def run_p1_row(ds: str, variant: str, repeat: int, model: str | None = None) -> dict:
    folder = RUNS / ds / f"{variant}-r{repeat}"
    if _done(folder):
        return json.loads((folder / "result.json").read_text())
    opts = RunOptions(review_mode="auto", out_dir=folder, model=model,
                      use_llm=variant != "P1-rules", enable_repair=variant != "P1-norepair",
                      allow_same_wave=variant != "P1-strict")
    cfg = RunConfig(max_transfer_gb=MAX_GB, confirmed=True, confirmed_by="evaluation harness (fixed config)")
    t = time.time()
    row = {"dataset": ds, "system": variant, "repeat": repeat, "model": (model or config.AGENT_MODEL)
           if opts.use_llm else None}
    try:
        sup = Supervisor(DATASETS[ds], cfg, opts, run_id=f"eval-{ds}-{variant}-r{repeat}")
        state = sup.run()
        log = json.loads((folder / "run_log.json").read_text())
        plan = pd.read_csv(folder / "plan.csv")
        scored = score_plan(plan, ds, variant, state["plan"]["infeasible"], require_finops=True)["metrics"]
        m = log["metrics"]
        tr = log["agent_trace_summary"]
        ex = m.get("explanations") or {}
        row.update(scored)
        row.update({
            "completed": True, "status": log["status"], "valid_plan": log["passed_validation"],
            # P1's own validator accounts for reviewer-approved over-limit groups; keep both views.
            "blocking_violations": m["blocking_violations"], "over_limit_approved": m["over_limit_approved"],
            "finops_savings_pct": m["finops_savings_pct"], "finops_aws_monthly": m["finops_aws_monthly"],
            "repair_rounds": m["repair_rounds"], "repair_resizes": m["repair_resizes"],
            "repair_rejected_actions": m["repair_rejected_actions"], "repair_escalations": m["repair_escalations"],
            "review_items": m["review_items"], "review_items_drafted_by_llm": m["review_items_drafted_by_llm"],
            "llm_calls": tr["llm_calls"], "llm_errors": tr["llm_errors"], "tokens": tr["tokens_in"] + tr["tokens_out"],
            "tool_calls": tr["tool_calls"], "tool_errors": tr["tool_errors"], "fallbacks": tr["fallbacks"],
            "risk_expl_verif_failures": (ex.get("risk") or {}).get("verification_failures"),
            "risk_expl_fallback": (ex.get("risk") or {}).get("fallback_template"),
            "wave_expl_verif_failures": (ex.get("waves") or {}).get("verification_failures"),
            "wave_expl_fallback": (ex.get("waves") or {}).get("fallback_template"),
            "wave_assignment": dict(zip(plan["Server_ID"], plan["Wave_ID"].astype("Int64").astype(str))),
        })
        if repeat == 1:
            PLANS.mkdir(parents=True, exist_ok=True)
            plan.to_csv(PLANS / f"{ds}_{variant}.csv", index=False)
    except Exception as exc:  # noqa: BLE001 - a failed run is a data point (completion reliability)
        row.update({"completed": False, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    row["latency_sec"] = round(time.time() - t, 1)
    return _save(folder, row)


# ----------------------------------------------------------------------------- summary
TABLE_COLS = [
    ("valid_plan", "Valid plan"), ("dependency_violations", "Dep. viol."),
    ("over_limit_unreported", "Over-limit (unreported)"), ("waves_over_limit", "Over-limit waves"),
    ("undersized", "Undersized"), ("unsupported_sku", "Non-AWS / unsupported sizes"),
    ("target_cloud_in_plan", "Cloud in plan"), ("early_satisfied_pct", "Early on time %"),
    ("early_mean_wave", "Early mean wave"),
    ("cross_wave_edges", "Split deps"), ("cross_wave_edge_pct", "Split deps %"), ("wave_count", "Waves"),
    ("utilization_pct", "Utilization %"), ("risk_completeness_pct", "Risk filled %"),
    ("risk_agreement_with_stakeholder_rule_pct", "Risk = stakeholder rule %"),
    ("finops_savings_pct", "Est. savings %"), ("latency_sec", "Latency s"), ("llm_calls", "LLM calls"),
]


def summarize(rows: list[dict]) -> dict:
    by: dict[tuple, list[dict]] = {}
    for r in rows:
        by.setdefault((r["dataset"], r["system"]), []).append(r)
    summary = {}
    for (ds, sysname), rs in sorted(by.items()):
        ok = [r for r in rs if r.get("completed")]
        first = sorted(rs, key=lambda r: r["repeat"])[0]
        entry = {k: first.get(k) for k, _ in TABLE_COLS}
        entry.update({"runs": len(rs), "completed_runs": len(ok),
                      "completion_rate_pct": round(100 * len(ok) / len(rs), 1),
                      "valid_rate_pct": round(100 * sum(bool(r.get("valid_plan")) for r in rs) / len(rs), 1),
                      "latency_sec_mean": round(statistics.mean(r["latency_sec"] for r in rs), 1)})
        if len(ok) > 1:
            waves = [r["wave_count"] for r in ok]
            cuts = [r["cross_wave_edges"] for r in ok]
            entry["wave_count_std"] = round(statistics.pstdev(waves), 2)
            entry["split_deps_std"] = round(statistics.pstdev(cuts), 2)
            assigns = [r.get("wave_assignment") for r in ok if r.get("wave_assignment")]
            entry["identical_plans_pct"] = round(100 * sum(a == assigns[0] for a in assigns) / len(assigns), 1) \
                if assigns else None
        for k in ("llm_calls", "tokens", "tool_calls", "tool_errors", "fallbacks", "repair_resizes",
                  "repair_rejected_actions", "review_items", "risk_expl_verif_failures", "risk_expl_fallback",
                  "wave_expl_verif_failures", "wave_expl_fallback"):
            vals = [r[k] for r in ok if r.get(k) is not None]
            if vals:
                entry[f"{k}_mean"] = round(statistics.mean(vals), 1)
        summary[f"{ds}|{sysname}"] = entry
    return summary


def markdown(summary: dict) -> str:
    datasets = sorted({k.split("|")[0] for k in summary}, key=list(DATASETS).index)
    out = ["# Evaluation results", "",
           "Generated by `eval/run_eval.py`. Every plan is scored by the same validator with the same input "
           f"data and a {MAX_GB:,.0f} GB wave limit. Values are from the first run; P1 rows list the repeat count.", ""]
    for ds in datasets:
        systems = [k.split("|")[1] for k in summary if k.startswith(ds + "|")]
        order = ["B0", "B1", "P1", "P1-rules", "P1-norepair", "P1-strict", "B1-strict"]
        systems.sort(key=lambda s: order.index(s) if s in order else 99)
        out += [f"## {ds}", "", "| Metric | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
        for key, label in TABLE_COLS + [("runs", "Runs"), ("completion_rate_pct", "Completion %"),
                                        ("identical_plans_pct", "Identical plans %")]:
            vals = []
            for s in systems:
                v = summary[f"{ds}|{s}"].get(key)
                vals.append("—" if v is None else ("yes" if v is True else "no" if v is False else f"{v}"))
            out.append(f"| {label} | " + " | ".join(vals) + " |")
        out.append("")
    return "\n".join(out)


def collect() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob("*/*/result.json"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="D1,D2,D3_30,D3_100,D3_500")
    ap.add_argument("--systems", default="B0,B1,P1")
    ap.add_argument("--ablations", default="P1-rules,P1-norepair,P1-strict,B1-strict")
    ap.add_argument("--ablation-datasets", default="D1,D2,D3_100")
    ap.add_argument("--ablation-model", default=None, help="e.g. qwen3-235b-a22b")
    ap.add_argument("--p1-repeats", type=int, default=3)
    ap.add_argument("--b0-repo", default=str(Path(__file__).resolve().parent.parent.parent / "migration-planning-ai"))
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if not a.summarize_only:
        systems = a.systems.split(",")
        for ds in a.datasets.split(","):
            if "B1" in systems:
                print(ds, "B1", run_b1_row(ds, strict=False).get("wave_count"), flush=True)
            if "B0" in systems and Path(a.b0_repo).exists():
                r = run_b0_row(ds, Path(a.b0_repo))
                print(ds, "B0", r.get("completed"), r.get("latency_sec"), flush=True)
            if "P1" in systems:
                for k in range(1, a.p1_repeats + 1):
                    r = run_p1_row(ds, "P1", k)
                    print(ds, f"P1 r{k}", r.get("status"), r.get("latency_sec"), r.get("error", ""), flush=True)
        for ds in a.ablation_datasets.split(","):
            if ds not in a.datasets.split(","):
                continue
            for ab in [x for x in a.ablations.split(",") if x]:
                r = run_b1_row(ds, strict=True) if ab == "B1-strict" else run_p1_row(ds, ab, 1)
                print(ds, ab, r.get("status", ""), r.get("latency_sec"), flush=True)
            if a.ablation_model:
                r = run_p1_row(ds, f"P1-{a.ablation_model}", 1, model=a.ablation_model)
                print(ds, f"P1-{a.ablation_model}", r.get("status"), r.get("latency_sec"), flush=True)
    rows = collect()
    summary = summarize(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval_runs.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "wave_assignment"} for r in rows], indent=2, default=str))
    (RESULTS / "eval_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (RESULTS / "eval_tables.md").write_text(markdown(summary))
    print(f"\nWrote {RESULTS / 'eval_tables.md'} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
