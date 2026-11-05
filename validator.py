"""
Plan Validator (P1, new in this milestone)
================================================
Deterministic, Neo4j-independent checks against ANY exported migration
plan CSV (B0, B1, or P1), using the matching cleaned discovery CSV as
ground truth for edges, Transfer_GB, and CPU/RAM. This lets the same
validator score all three systems identically for the Milestone 2
comparison table.

Checks:
  - Schema completeness: required columns present, Wave_ID assigned.
  - Dependency validity: no edge (src -> dst) where wave(src) > wave(dst).
  - Transfer-volume validity: no wave's summed Transfer_GB exceeds the
    configured limit, except groups explicitly reported as infeasible
    (indivisible) by the system itself.
  - Sizing safety: assigned instance (Target_Size), if present, must meet
    or exceed each server's CPU/RAM from the static catalog.

Usage:
    python validator.py --plan <plan.csv> --cleaned <cleaned.csv> \
        --label <name> [--max-transfer-gb 3000] [--infeasible path.json]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from common import PRICING_CATALOG_PATH


def _parse_edges(cleaned_df: pd.DataFrame) -> list[tuple[str, str]]:
    edges = []
    for _, row in cleaned_df.iterrows():
        raw = str(row.get("Outbound_Connections", ""))
        if not raw or raw in ("nan", "None", ""):
            continue
        src = row["Server_ID"]
        for conn in raw.split("|"):
            parts = conn.strip().split(":")
            if len(parts) == 3:
                edges.append((src, parts[0]))
    return edges


def validate(plan_path: Path, cleaned_path: Path, label: str,
             max_transfer_gb: float, infeasible_path: Path | None) -> dict:
    plan = pd.read_csv(plan_path)
    cleaned = pd.read_csv(cleaned_path)
    catalog = json.loads(Path(PRICING_CATALOG_PATH).read_text())["instances"]

    report: dict = {"label": label, "plan_path": str(plan_path), "checks": {}}

    required_cols = ["Server_ID", "Wave_ID"]
    missing_cols = [c for c in required_cols if c not in plan.columns]
    report["checks"]["schema_completeness"] = {
        "required_columns_present": not missing_cols,
        "missing_columns": missing_cols,
        "total_rows": len(plan),
        "rows_with_wave_id": int(plan["Wave_ID"].notna().sum()) if "Wave_ID" in plan.columns else 0,
    }
    if missing_cols:
        report["checks"]["note"] = "Stopped further checks: required columns missing."
        return report

    transfer_col = "Transfer_GB" if "Transfer_GB" in plan.columns else None
    if transfer_col is None and "Server_ID" in cleaned.columns and "Transfer_GB" in cleaned.columns:
        plan = plan.merge(cleaned[["Server_ID", "Transfer_GB"]], on="Server_ID", how="left")
        transfer_col = "Transfer_GB"

    edges = _parse_edges(cleaned)
    wave_of = dict(zip(plan["Server_ID"], plan["Wave_ID"]))

    dep_violations = []
    for src, dst in edges:
        if src in wave_of and dst in wave_of and pd.notna(wave_of[src]) and pd.notna(wave_of[dst]):
            if float(wave_of[src]) > float(wave_of[dst]):
                dep_violations.append({"src": src, "dst": dst, "wave_src": wave_of[src], "wave_dst": wave_of[dst]})
    report["checks"]["dependency_validity"] = {
        "edges_checked": len(edges),
        "violations": len(dep_violations),
        "examples": dep_violations[:5],
    }

    infeasible_group_members: set[str] = set()
    if infeasible_path and infeasible_path.exists():
        summary = json.loads(infeasible_path.read_text())
        for g in summary.get("infeasible_groups", []):
            infeasible_group_members.update(g.get("members", []))

    volume_violations = []
    if transfer_col:
        wave_totals = plan.groupby("Wave_ID")[transfer_col].sum()
        wave_has_infeasible = plan[plan["Server_ID"].isin(infeasible_group_members)]["Wave_ID"].unique().tolist()
        for wave_id, total in wave_totals.items():
            if total > max_transfer_gb and wave_id not in wave_has_infeasible:
                volume_violations.append({"wave_id": wave_id, "total_gb": round(float(total), 1)})
        report["checks"]["transfer_volume_validity"] = {
            "max_transfer_gb": max_transfer_gb,
            "waves_checked": int(wave_totals.shape[0]),
            "violations": len(volume_violations),
            "examples": volume_violations[:5],
            "known_infeasible_waves_excused": wave_has_infeasible,
            "wave_totals_gb": {str(k): round(float(v), 1) for k, v in wave_totals.items()},
        }
    else:
        report["checks"]["transfer_volume_validity"] = {"skipped": "no Transfer_GB column available"}

    if "Target_Size" in plan.columns:
        undersized = []
        merged = plan.merge(
            cleaned[["Server_ID", "Allocated_vCPU", "Allocated_RAM_GB"]],
            on="Server_ID", how="left",
        )
        for _, row in merged.iterrows():
            spec = catalog.get(row.get("Target_Size"))
            if spec is None:
                continue
            needed_cpu = row.get("Allocated_vCPU")
            needed_ram = row.get("Allocated_RAM_GB")
            if pd.notna(needed_cpu) and pd.notna(needed_ram):
                if needed_cpu > spec["vcpu"] or needed_ram > spec["ram_gb"]:
                    undersized.append({"server_id": row["Server_ID"], "target_size": row["Target_Size"]})
        report["checks"]["sizing_safety"] = {
            "rows_checked": len(merged),
            "undersized_violations": len(undersized),
            "examples": undersized[:5],
        }
    else:
        report["checks"]["sizing_safety"] = {"skipped": "no Target_Size column available"}

    if "Risk_Score" in plan.columns:
        report["checks"]["risk_score_completeness"] = {
            "rows_with_risk_score": int(plan["Risk_Score"].notna().sum()),
            "total_rows": len(plan),
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-transfer-gb", type=float, default=3000.0)
    parser.add_argument("--infeasible", default=None)
    args = parser.parse_args()

    report = validate(
        Path(args.plan), Path(args.cleaned), args.label,
        args.max_transfer_gb, Path(args.infeasible) if args.infeasible else None,
    )
    out_path = Path(args.plan).parent / f"validation_report_{args.label}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
