"""
Supervisor (P1, new in this milestone)
============================================
Orchestrates the full P1 pipeline end to end against a chosen dataset,
maintaining a run log (provenance: stage, start/end time, exit status,
retries) and refusing to proceed to planning if the constraint
configuration was not confirmed. This is the single entry point used by
run_pipeline_p1.sh.

Usage:
    python supervisor.py --input data/sample.csv --run-id stakeholder
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import DEFAULT_RUN_CONFIG_PATH

STAGES = [
    ("data_agent", ["data_agent.py", "{input}", "data/cleaned_samples.csv"]),
    ("neo4j_agent", ["neo4j_agent.py", "data/cleaned_samples.csv"]),
    ("graph_agent", ["graph_agent.py", "{max_transfer_gb}"]),
    ("cloud_selection_agent", ["cloud_selection_agent.py"]),
    ("sizing_agent", ["sizing_agent.py"]),
    ("finops_agent", ["finops_agent.py"]),
    ("risk_agent", ["risk_agent.py"]),
    ("wave_agent_p1", ["wave_agent_p1.py"]),
]


def run_stage(name: str, args: list[str]) -> dict:
    start = time.time()
    result = subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, cwd=Path(__file__).parent
    )
    end = time.time()
    return {
        "stage": name,
        "args": args,
        "exit_code": result.returncode,
        "duration_sec": round(end - start, 2),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the raw discovery CSV.")
    parser.add_argument("--run-id", required=True, help="Label used in the run log filename.")
    parser.add_argument("--clear-graph", action="store_true", help="Wipe Neo4j before this run.")
    args = parser.parse_args()

    if not DEFAULT_RUN_CONFIG_PATH.exists():
        print(f"ERROR: {DEFAULT_RUN_CONFIG_PATH} not found. Run constraint_agent.py first.")
        sys.exit(1)
    run_config = json.loads(DEFAULT_RUN_CONFIG_PATH.read_text())
    if not run_config.get("confirmed", False):
        print("ERROR: run_config.json is not confirmed. Re-run constraint_agent.py and confirm.")
        sys.exit(1)

    max_transfer_gb = str(run_config.get("max_transfer_gb", 3000.0))

    if args.clear_graph:
        from common import get_graph
        get_graph().query("MATCH (n) DETACH DELETE n")
        print("Cleared Neo4j graph for a clean run.")

    run_log = {
        "run_id": args.run_id,
        "input": args.input,
        "run_config": run_config,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": [],
    }

    for name, template_args in STAGES:
        resolved = [
            a.format(input=args.input, max_transfer_gb=max_transfer_gb) for a in template_args
        ]
        print(f"\n=== Stage: {name} ({' '.join(resolved)}) ===")
        stage_result = run_stage(name, resolved)
        print(stage_result["stdout_tail"][-1500:])
        if stage_result["stderr_tail"].strip():
            print("--- stderr (tail) ---")
            print(stage_result["stderr_tail"][-800:])
        run_log["stages"].append(stage_result)

        if stage_result["exit_code"] != 0:
            print(f"\nSupervisor: stage '{name}' failed (exit {stage_result['exit_code']}). Stopping run.")
            run_log["status"] = "failed"
            run_log["failed_stage"] = name
            break
    else:
        run_log["status"] = "completed"

    run_log["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    log_path = Path("data") / f"run_log_{args.run_id}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(run_log, indent=2))
    print(f"\nRun log written to {log_path} (status={run_log['status']})")


if __name__ == "__main__":
    main()
