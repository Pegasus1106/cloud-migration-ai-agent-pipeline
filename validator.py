"""Validate ANY plan CSV (B0, B1, or P1) against the Milestone 1 rules.

  python validator.py --plan results/plans/D1_B0.csv --dataset data/sample.csv [--max-transfer-gb 3000]
      [--infeasible results/plans/D1_B1_summary.json] [--out report.json]

Exit code 0 if the plan has no blocking violations, 2 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from p1.core.data import load_and_normalize
from p1.core.validation import validate_plan


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True)
    p.add_argument("--dataset", required=True, help="The discovery CSV the plan was made from")
    p.add_argument("--max-transfer-gb", type=float, default=3000.0)
    p.add_argument("--infeasible", help="JSON with an 'infeasible_groups' list reported by the planner")
    p.add_argument("--out")
    a = p.parse_args()
    cleaned, edges, _ = load_and_normalize(a.dataset)
    reported = json.loads(Path(a.infeasible).read_text()).get("infeasible_groups") if a.infeasible else None
    early = set(cleaned.loc[cleaned["Early_Migration_Requested"], "Server_ID"])
    rep = validate_plan(pd.read_csv(a.plan), cleaned, [(e["src"], e["dst"]) for e in edges], a.max_transfer_gb,
                        label=Path(a.plan).stem, reported_infeasible=reported, early_ids=early)
    text = json.dumps(rep, indent=2, default=str)
    if a.out:
        Path(a.out).write_text(text)
    print(json.dumps({"label": rep["label"], "passed": rep["passed"], "blocking": rep["blocking_count"],
                      "metrics": rep["metrics"]}, indent=2))
    for v in rep["violations"][:15]:
        print(f"  [{v['severity']}] {v['id']}: {v['message']}")
    return 0 if rep["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
