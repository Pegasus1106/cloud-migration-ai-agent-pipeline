"""B1 - Greedy rule-only baseline (Milestone 1, Section 5).

Outbound edges plus derived inbound metadata, SCC-safe topological order, and
deterministic first-fit-decreasing packing under the transfer-volume limit.
No LLM, no Neo4j, no community detection, no early-migration or risk priority,
no FinOps, no repair, no review. Sizing uses the stakeholder matrix and risk the
stakeholder's binary rule, exactly like B0.

It shares the cleaning code and the packer with P1 (p1.core), so P1 and B1
differ only in the packing priority - the comparison Milestone 1 defines.

Usage:
  python baselines/b1_greedy_wave_agent.py --dataset data/sample.csv \
      --out-plan results/plans/D1_B1.csv --out-summary results/plans/D1_B1_summary.json [--strict-waves]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p1.core import catalog as cat  # noqa: E402
from p1.core.data import load_and_normalize  # noqa: E402
from p1.core.planning import plan_b1  # noqa: E402
from p1.core.risk import baseline_risk  # noqa: E402

COLUMNS = ["Server_ID", "Hostname", "App_Name", "OS_Version", "Target_Size", "Risk_Score", "Transfer_GB",
           "Current_Monthly_Cost", "SCC_Id", "Wave_ID", "Early_Migration_Requested"]


def run(dataset: Path, max_gb: float, out_plan: Path, out_summary: Path, allow_same_wave: bool = True) -> dict:
    df, edges, _ = load_and_normalize(dataset)
    df["Target_Size"] = [cat.size_by_matrix(c, r) for c, r in zip(df["Allocated_vCPU"], df["Allocated_RAM_GB"])]
    df["Risk_Score"] = [baseline_risk(o, t) for o, t in zip(df["OS_Version"], df["Business_Tags"])]
    res = plan_b1(df.to_dict("records"), [(e["src"], e["dst"]) for e in edges], max_gb, allow_same_wave)
    df["Wave_ID"] = df["Server_ID"].map(res.wave_of_server)
    df["SCC_Id"] = df["Server_ID"].map(res.group_of_server)
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].sort_values(["Wave_ID", "Server_ID"]).to_csv(out_plan, index=False)
    summary = {"system": "B1 (greedy first-fit-decreasing, SCC granularity, no community/early/risk priority)",
               "packing": res.packing, "max_transfer_gb": max_gb, "wave_count": len(res.wave_totals),
               "wave_totals_gb": {str(k): round(v, 1) for k, v in sorted(res.wave_totals.items())},
               "infeasible_groups": res.infeasible, "group_count": len(res.group_members)}
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", "--cleaned", dest="dataset", required=True, help="Discovery CSV")
    p.add_argument("--max-transfer-gb", type=float, default=3000.0)
    p.add_argument("--out-plan", required=True)
    p.add_argument("--out-summary", required=True)
    p.add_argument("--strict-waves", action="store_true", help="Prototype packing (no same-wave dependencies)")
    a = p.parse_args()
    s = run(Path(a.dataset), a.max_transfer_gb, Path(a.out_plan), Path(a.out_summary), not a.strict_waves)
    print(f"B1: {s['wave_count']} waves, {len(s['infeasible_groups'])} over-limit groups -> {a.out_plan}")


if __name__ == "__main__":
    main()
