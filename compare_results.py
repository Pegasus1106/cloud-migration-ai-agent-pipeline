"""
Comparison script (P1, new in this milestone)
================================================
Computes the Milestone 2 comparison metrics (dependency cuts, wave
count, utilization, early-migration handling, validator results) across
B0, B1, and P1 outputs for a given dataset, and writes a combined JSON +
prints a markdown table.

Usage:
    python compare_results.py --dataset stakeholder \
        --cleaned data/cleaned_stakeholder.csv \
        --b0-plan baselines/stakeholder_b0_plan.csv \
        --b1-plan baselines/stakeholder_b1_plan.csv \
        --p1-plan data/stakeholder_p1_plan.csv \
        --early-ids SRV-008,SRV-009,SRV-048,SRV-049
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _edges(cleaned: pd.DataFrame) -> list[tuple[str, str]]:
    edges = []
    for _, row in cleaned.iterrows():
        raw = str(row.get("Outbound_Connections", ""))
        if not raw or raw in ("nan", "None", ""):
            continue
        for conn in raw.split("|"):
            parts = conn.strip().split(":")
            if len(parts) == 3:
                edges.append((row["Server_ID"], parts[0]))
    return edges


def cross_wave_edge_count(edges, wave_of: dict) -> int:
    count = 0
    for s, d in edges:
        if s in wave_of and d in wave_of and wave_of[s] != wave_of[d]:
            count += 1
    return count


def utilization(wave_totals: dict, max_transfer_gb: float) -> float:
    non_final = {k: v for k, v in wave_totals.items() if v > 0}
    if len(non_final) <= 1:
        return 0.0
    values = sorted(non_final.values())
    excl_last = values[:-1]  # exclude the final (likely partial) wave
    if not excl_last:
        return 0.0
    return round(sum(excl_last) / (len(excl_last) * max_transfer_gb) * 100, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--b0-plan")
    parser.add_argument("--b1-plan")
    parser.add_argument("--p1-plan")
    parser.add_argument("--early-ids", default="")
    parser.add_argument("--max-transfer-gb", type=float, default=3000.0)
    args = parser.parse_args()

    cleaned = pd.read_csv(args.cleaned)
    edges = _edges(cleaned)
    early_ids = [x for x in args.early_ids.split(",") if x]

    result = {"dataset": args.dataset, "systems": {}}

    for label, path in [("B0", args.b0_plan), ("B1", args.b1_plan), ("P1", args.p1_plan)]:
        if not path or not Path(path).exists():
            continue
        df = pd.read_csv(path)
        wave_of = dict(zip(df["Server_ID"], df["Wave_ID"]))
        cuts = cross_wave_edge_count(edges, wave_of)

        if "Transfer_GB" in df.columns:
            wave_totals = df.groupby("Wave_ID")["Transfer_GB"].sum().to_dict()
        else:
            merged = df.merge(cleaned[["Server_ID", "Transfer_GB"]], on="Server_ID", how="left")
            wave_totals = merged.groupby("Wave_ID")["Transfer_GB"].sum().to_dict()

        util = utilization(wave_totals, args.max_transfer_gb)
        early_waves = {sid: int(wave_of[sid]) for sid in early_ids if sid in wave_of}
        earliest_wave = min(wave_of.values()) if wave_of else None

        result["systems"][label] = {
            "wave_count": int(df["Wave_ID"].nunique()),
            "cross_wave_edges": cuts,
            "wave_utilization_pct_excl_last": util,
            "early_migration_waves": early_waves,
            "early_migration_min_possible_wave": earliest_wave,
            "has_finops": "AWS_Monthly_Cost" in df.columns or "Savings_Amount" in df.columns,
            "has_risk_score": "Risk_Score" in df.columns and df["Risk_Score"].notna().any(),
        }

    def _default(o):
        if isinstance(o, (bool,)):
            return bool(o)
        try:
            return o.item()  # numpy scalar types (bool_, int64, float64, ...)
        except AttributeError:
            return str(o)

    out_path = Path(f"comparison_{args.dataset}.json")
    out_path.write_text(json.dumps(result, indent=2, default=_default))
    print(json.dumps(result, indent=2, default=_default))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
