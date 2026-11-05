"""
Dataset builder — Stakeholder Sample (D1)
==========================================
Extends the original baseline `data/sample.csv` (50 servers, outbound-only,
no cost/transfer fields) with the three fields the Milestone 1 refinement
requires as inputs: Transfer_GB, Current_Monthly_Cost, and
Early_Migration_Requested.

Documented assumptions (per Milestone 1 "Key assumptions"):
  - Transfer_GB approximates migration data volume. We set it equal to
    Storage_GB (missing Storage_GB defaults to 50, matching data_agent.py).
  - Current_Monthly_Cost is not present in the original discovery CSV, so
    we synthesize a simple on-prem TCO proxy:
        cost = 40*vCPU + 10*RAM_GB + 0.05*Storage_GB + 50 (base overhead)
    This is a documented modeling assumption for the prototype, not a real
    bill. It is held constant across B0/B1/P1 so FinOps comparisons are
    apples-to-apples.
  - Early_Migration_Requested is set True for two independent, low-risk
    application clusters to simulate a stakeholder "move these first"
    request that the scheduler can actually satisfy without violating
    dependency order: CRM (external-facing, customer SLA) and Corp_Blog
    (low-risk pilot candidate).

This project is fully standalone: `data/baseline_raw_sample.csv` is a
local, static copy of the stakeholder's original 50-server discovery CSV
(the unmodified baseline input, kept here so this repository has no
runtime or path dependency on the separate baseline repository).

Run: python make_stakeholder_sample.py
Output: data/sample.csv (overwritten in this directory only).
"""

from pathlib import Path

import pandas as pd

SRC = Path(__file__).parent / "baseline_raw_sample.csv"
OUT = Path(__file__).parent / "sample.csv"

EARLY_MIGRATION_IDS = {"SRV-008", "SRV-009", "SRV-048", "SRV-049"}


def main() -> None:
    df = pd.read_csv(SRC)

    storage = df["Storage_GB"].fillna(50)
    cpu = df["Allocated_vCPU"].fillna(2)
    ram = df["Allocated_RAM_GB"].fillna(4)

    df["Transfer_GB"] = storage.round(1)
    df["Current_Monthly_Cost"] = (40 * cpu + 10 * ram + 0.05 * storage + 50).round(2)
    df["Early_Migration_Requested"] = df["Server_ID"].isin(EARLY_MIGRATION_IDS)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"Early-migration flagged: {sorted(EARLY_MIGRATION_IDS)}")


if __name__ == "__main__":
    main()
