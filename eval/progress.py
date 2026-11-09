"""Live progress for the Milestone 2 evaluation (no extra packages needed).

  python3 eval/progress.py            # one snapshot
  python3 eval/progress.py --watch    # refresh every 20 s; macOS notification + bell when finished

Reads the run folders written by eval/run_eval.py (runs/eval/<dataset>/<system>/result.json)
and the Q&A benchmark output, compares them with the planned evaluation, and estimates the
finish time from how long finished runs took.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "eval"
RESULTS = ROOT / "results"
SERVERS = {"D1": 50, "D2": 83, "D3_30": 30, "D3_100": 100, "D3_500": 500}
ABLATION_MODEL = "qwen3-235b-a22b"


def planned() -> list[tuple[str, str]]:
    """(dataset, run folder) in the order the background chain executes them."""
    plan = []
    for ds in ("D1", "D2", "D3_30", "D3_100"):
        plan += [(ds, "B1"), (ds, "B0")] + [(ds, f"P1-r{k}") for k in (1, 2, 3)]
    for ds in ("D1", "D2", "D3_100"):
        plan += [(ds, "P1-rules-r1"), (ds, "P1-norepair-r1"), (ds, "P1-strict-r1"), (ds, "B1-strict"),
                 (ds, f"P1-{ABLATION_MODEL}-r1")]
    plan += [("D3_500", "B1"), ("D3_500", "B0"), ("D3_500", "P1-r1")]
    plan += [("D1", f"P1-r{k}") for k in range(4, 11)]
    plan.append(("QA", "Ask-the-plan benchmark (27 questions x 2 systems)"))
    return plan


def family(run: str) -> str:
    return run.rsplit("-r", 1)[0] if run.startswith("P1") else run


def status_of(ds: str, run: str) -> tuple[str, float | None, float | None]:
    """(state, duration_sec if finished, elapsed_sec if running)."""
    if ds == "QA":
        if list(RESULTS.glob("qa_eval_*.json")):
            return "done", None, None
        plan_dir = RUNS / "qa-plan"
        if plan_dir.exists():
            return "running", None, time.time() - plan_dir.stat().st_mtime
        return "pending", None, None
    folder = RUNS / ds / run
    res = folder / "result.json"
    if res.exists():
        r = json.loads(res.read_text())
        state = "done" if r.get("completed") else "failed"
        return state, r.get("latency_sec"), None
    if folder.exists():
        st = folder.stat()
        born = getattr(st, "st_birthtime", st.st_ctime)  # ctime changes whenever a file is written inside
        return "running", None, time.time() - born
    return "pending", None, None


def estimate(ds: str, run: str, finished: dict) -> float:
    if ds == "QA":
        return 420.0
    fam = family(run)
    same = finished.get((fam, ds))
    if same:
        return statistics.mean(same)
    other = [(d, t) for (f, d), ts in finished.items() if f == fam for t in ts]
    if other:
        per_server = statistics.mean(t / SERVERS[d] for d, t in other)
        base = per_server * SERVERS[ds]
        return base if fam in ("B1", "B1-strict", "P1-rules") else max(base, 60.0)
    defaults = {"B1": 1, "B1-strict": 1, "P1-rules": 5, "B0": 150}
    return defaults.get(fam, 30 + 3.0 * SERVERS[ds] * (1.8 if ABLATION_MODEL in fam else 1.0))


def snapshot() -> tuple[str, bool]:
    rows = [(ds, run, *status_of(ds, run)) for ds, run in planned()]
    finished: dict = {}
    for ds, run, state, dur, _ in rows:
        if state in ("done", "failed") and dur is not None:
            finished.setdefault((family(run), ds), []).append(dur)
    done = sum(r[2] in ("done", "failed") for r in rows)
    failed = [f"{r[0]} {r[1]}" for r in rows if r[2] == "failed"]
    total = len(rows)
    remaining = 0.0
    for ds, run, state, _, elapsed in rows:
        if state == "pending":
            remaining += estimate(ds, run, finished)
        elif state == "running":
            remaining += max(estimate(ds, run, finished) - (elapsed or 0), 15)
    width = 40
    filled = int(width * done / total)
    bar = "█" * filled + "░" * (width - filled)
    eta = datetime.now() + timedelta(seconds=remaining)
    lines = [f"Milestone 2 evaluation  [{bar}]  {done}/{total} runs ({100 * done / total:.0f}%)"]
    running = [r for r in rows if r[2] == "running"]
    for ds, run, _, _, elapsed in running:
        lines.append(f"  running : {ds} {run}  ({int(elapsed or 0) // 60}m {int(elapsed or 0) % 60:02d}s so far)")
    nxt = next(((ds, run) for ds, run, st, _, _ in rows if st == "pending"), None)
    if nxt:
        lines.append(f"  next    : {nxt[0]} {nxt[1]}")
    if failed:
        lines.append(f"  failed  : {', '.join(failed)}")
    all_done = done == total
    if all_done:
        lines.append("  ALL DONE. Tables: results/eval_tables.md  Q&A: results/qa_eval_*.json  "
                     "Constraints: results/constraint_eval_*.json")
    else:
        lines.append(f"  estimate: about {int(remaining // 3600)}h {int(remaining % 3600 // 60):02d}m left "
                     f"-> finishes around {eta:%H:%M} (rough; LLM speed varies)")
    return "\n".join(lines), all_done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=20)
    a = ap.parse_args()
    if not a.watch:
        print(snapshot()[0])
        return
    while True:
        text, all_done = snapshot()
        sys.stdout.write("\033[H\033[J" + datetime.now().strftime("%H:%M:%S  ") + text + "\n")
        sys.stdout.flush()
        if all_done:
            print("\a")
            subprocess.run(["osascript", "-e", 'display notification "All evaluation runs finished" '
                            'with title "Cloud Migration P1"'], check=False, capture_output=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
