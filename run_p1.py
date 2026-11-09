"""Command-line entry point for the P1 agentic planner.

Examples:
  python run_p1.py --dataset data/sample.csv --request "Keep waves under 3 TB and move CRM early" --yes
  python run_p1.py --dataset data/synthetic_83.csv --interactive          # chat + human review in the terminal
  python run_p1.py --dataset data/sample.csv --no-llm --yes               # rule-only mode (ablation)

Exit codes: 0 = plan passed validation (ready for approval), 2 = blocked or
rejected (violations remain or reviewer rejected), 1 = error.
"""

import argparse
import json
import sys

from p1 import config
from p1.core.data import load_and_normalize
from p1.llm import llm_available
from p1.state import RunConfig
from p1.supervisor import RunOptions, Supervisor
from p1.tracing import Tracer


def _ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def build_config(args, df) -> RunConfig:
    if args.request:
        if not llm_available():
            sys.exit(f"--request needs the LLM: set {config.LLM_API_KEY_ENV} in .env (or use flags with --no-llm).")
        from p1.agents.constraint_agent import ConstraintAgent
        agent = ConstraintAgent(df, Tracer("constraint-cli"), args.model)
        turn = agent.send(args.request)
        while turn["status"] != "proposed":
            print(f"\nConstraint Agent: {turn['reply']}")
            if not sys.stdin.isatty() or args.yes:
                sys.exit("The Constraint Agent needs clarification; re-run interactively or rephrase --request.")
            turn = agent.send(_ask("You: "))
        print(f"\nConstraint Agent: {turn['reply']}")
        cfg = turn["proposed"]
        for w in turn["warnings"]:
            print(f"  warning: {w}")
    else:
        cfg = RunConfig(max_transfer_gb=args.max_gb, target_cloud=args.target, pilot_wave=args.pilot,
                        source_request="command-line flags")
    print("\nProposed configuration:")
    print(json.dumps(cfg.model_dump(exclude={"confirmed", "confirmed_by"}), indent=2))
    if args.yes:
        cfg.confirmed, cfg.confirmed_by = True, "auto (--yes flag)"
    else:
        ok = _ask("\nConfirm this configuration? [y/N]: ").strip().lower() == "y"
        if not ok:
            sys.exit("Not confirmed; nothing was planned.")
        cfg.confirmed, cfg.confirmed_by = True, "user (terminal)"
    return cfg


def review_in_terminal(items: list[dict]) -> dict:
    decisions = {}
    print(f"\n=== Human review: {len(items)} items ===")
    for item in items:
        print(f"\n[{item['id']}] {item['title']}")
        print(f"  {item.get('summary', '')}")
        if item.get("suggestion"):
            print(f"  Suggestion: {item['suggestion']}")
        print(f"  Agent recommends: {item.get('recommended_action')}  (drafted by {item.get('drafted_by')})")
        opts = item["allowed_actions"]
        for i, a in enumerate(opts, 1):
            print(f"    {i}. {a}")
        choice = _ask(f"  Choose 1-{len(opts)} [Enter = recommended]: ").strip()
        action = opts[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(opts) else item.get(
            "recommended_action", opts[0])
        decisions[item["id"]] = {"action": action, "comment": _ask("  Comment (optional): ").strip()}
    return decisions


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Discovery CSV")
    p.add_argument("--request", help="Free-text planning request for the Constraint Agent")
    p.add_argument("--max-gb", type=float, default=config.DEFAULT_MAX_TRANSFER_GB, help="Wave limit (no --request)")
    p.add_argument("--target", default="AWS", choices=["AWS", "AZURE"])
    p.add_argument("--pilot", action="store_true", help="Reserve wave 0 for a low-risk pilot (no --request)")
    p.add_argument("--yes", action="store_true", help="Auto-confirm the configuration (logged as such)")
    p.add_argument("--interactive", action="store_true", help="Pause for human review decisions in the terminal")
    p.add_argument("--no-repair", action="store_true", help="Disable the Repair Agent (ablation)")
    p.add_argument("--no-llm", action="store_true", help="Rule-only mode: all agents use deterministic fallbacks")
    p.add_argument("--strict-waves", action="store_true", help="Prototype packing: no same-wave dependencies")
    p.add_argument("--no-neo4j", action="store_true", help="Use the networkx graph fallback")
    p.add_argument("--no-explain", action="store_true", help="Skip LLM explanations")
    p.add_argument("--model", default=None, help=f"Agent model (default {config.AGENT_MODEL})")
    p.add_argument("--run-id")
    p.add_argument("--out-dir")
    args = p.parse_args()

    df, _, quality = load_and_normalize(args.dataset)
    print(f"Loaded {quality['servers_after_dedup']} servers, {quality['edge_count']} dependencies from {args.dataset}")
    cfg = build_config(args, df)
    opts = RunOptions(review_mode="interactive" if args.interactive else "auto", enable_repair=not args.no_repair,
                      allow_same_wave=not args.strict_waves, use_llm=not args.no_llm, model=args.model,
                      explain=not args.no_explain, use_neo4j=not args.no_neo4j, out_dir=args.out_dir)
    try:
        sup = Supervisor(args.dataset, cfg, opts, run_id=args.run_id)
        events = sup.stream()
        while True:
            for ev in events:
                print(f"  [{ev['status']:>7}] {ev['label']:<30} {ev['progress']}")
            if sup.interrupt_payload is None:
                break
            events = sup.resume(review_in_terminal(sup.interrupt_payload["items"]))
        state = sup.state
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    m = state["validation"]["metrics"]
    print(f"\nStatus: {state['status']}  {state.get('stop_reason', '')}")
    print(f"Waves {m['wave_count']} | dependencies split {m['cross_wave_edges']}/{m['dependency_edges']} | "
          f"utilization {m['utilization_pct']}% | over-limit waves {m['waves_over_limit']} "
          f"(approved {m['over_limit_approved']}) | undersized {m['undersized']} | early on time {m['early_satisfied_pct']}%")
    f = state["finops"]
    print(f"Cost ${f['total_current_monthly']:,.0f} -> ${f['total_aws_monthly']:,.0f}/month (AWS compute estimate)")
    s = sup.tracer.summary()
    print(f"Agents: {s['llm_calls']} LLM calls, {s['tool_calls']} tool calls ({s['tool_errors']} errors), "
          f"{s['fallbacks']} fallbacks, {s['tokens_in'] + s['tokens_out']:,} tokens")
    print(f"Outputs: {sup.out_dir}")
    return 0 if state["status"] == "ready_for_approval" else 2


if __name__ == "__main__":
    sys.exit(main())
