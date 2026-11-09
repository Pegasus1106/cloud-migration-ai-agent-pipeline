"""Run the stakeholder's original pipeline (B0) unmodified on any dataset.

B0 (https://github.com/ananthsekar007/migration-planning-ai) always reads
data/sample.csv and writes data/final_migration_plan.csv. This runner only
applies the reproducibility adjustments allowed by Milestone 1 (consistent input
filename, readiness polling, API key passed in): it temporarily swaps the input
file, runs `./run_pipeline.sh --clean --run` (fresh Neo4j volume every time),
copies the plan out, and always restores the original sample.csv.

Usage:
  python baselines/run_b0.py --b0-repo ../migration-planning-ai --dataset data/sample.csv \
      --out-plan results/plans/D1_B0.csv
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from p1 import config  # noqa: E402,F401  (loads .env)


def _compose_capable_docker() -> str | None:
    candidates = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    candidates += ["/usr/local/bin", "/Applications/Docker.app/Contents/Resources/bin"]
    for d in candidates:
        exe = Path(d) / "docker"
        if exe.exists():
            ok = subprocess.run([str(exe), "compose", "version"], capture_output=True).returncode == 0
            if ok:
                return str(exe)
    return None


def run_b0(b0_repo: Path, dataset: Path, out_plan: Path, timeout: int = 1800) -> dict:
    b0_repo = b0_repo.resolve()
    sample = b0_repo / "data" / "sample.csv"
    backup = sample.with_suffix(".csv.p1backup")
    docker = _compose_capable_docker()
    if docker is None:
        raise RuntimeError("No docker binary with the compose plugin was found.")
    shim = Path(tempfile.mkdtemp(prefix="b0bin-"))
    (shim / "docker").symlink_to(docker)
    env = dict(os.environ, PATH=f"{shim}{os.pathsep}{os.environ.get('PATH', '')}")
    if not env.get("ASU_RC_API_KEY"):
        raise RuntimeError("ASU_RC_API_KEY is not set (B0's agents need it).")
    shutil.copy2(sample, backup)
    t = time.time()
    try:
        shutil.copy2(dataset, sample)
        (b0_repo / "data" / "final_migration_plan.csv").unlink(missing_ok=True)
        proc = subprocess.run(["bash", "./run_pipeline.sh", "--clean", "--run"], cwd=b0_repo, env=env,
                              capture_output=True, text=True, timeout=timeout)
        produced = b0_repo / "data" / "final_migration_plan.csv"
        out_plan.parent.mkdir(parents=True, exist_ok=True)
        log_path = out_plan.with_suffix(".log.txt")
        log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        result = {"exit_code": proc.returncode, "duration_sec": round(time.time() - t, 1),
                  "plan_produced": produced.exists(), "log": str(log_path)}
        if produced.exists():
            shutil.copy2(produced, out_plan)
            result["plan"] = str(out_plan)
        return result
    finally:
        shutil.move(backup, sample)
        shutil.rmtree(shim, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--b0-repo", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-plan", required=True)
    a = p.parse_args()
    print(json.dumps(run_b0(Path(a.b0_repo), Path(a.dataset).resolve(), Path(a.out_plan)), indent=2))


if __name__ == "__main__":
    main()
