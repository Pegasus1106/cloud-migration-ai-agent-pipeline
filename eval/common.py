"""Shared evaluation helpers: dataset registry and plan scoring."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from p1.core.data import load_and_normalize  # noqa: E402
from p1.core.risk import baseline_risk  # noqa: E402
from p1.core.validation import validate_plan  # noqa: E402

DATASETS = {
    "D1": ROOT / "data" / "sample.csv",
    "D2": ROOT / "data" / "synthetic_83.csv",
    "D3_30": ROOT / "data" / "benchmark" / "D3_30.csv",
    "D3_100": ROOT / "data" / "benchmark" / "D3_100.csv",
    "D3_500": ROOT / "data" / "benchmark" / "D3_500.csv",
}
RESULTS = ROOT / "results"
MAX_GB = 3000.0


def score_plan(plan: pd.DataFrame, dataset: str, label: str, reported_infeasible=None,
               require_finops=False) -> dict:
    """Validate any system's plan with the same rules and data (Milestone 1 procedure, step 3)."""
    cleaned, edges, _ = load_and_normalize(DATASETS[dataset])
    pairs = [(e["src"], e["dst"]) for e in edges]
    early = set(cleaned.loc[cleaned["Early_Migration_Requested"], "Server_ID"])
    rep = validate_plan(plan, cleaned, pairs, MAX_GB, label=label, reported_infeasible=reported_infeasible,
                        early_ids=early, require_finops=require_finops)
    m = dict(rep["metrics"])
    m["blocking_violations"] = rep["blocking_count"]
    m["valid_plan"] = rep["passed"]
    # Sizes that cannot be checked against the AWS catalog (e.g. B0 choosing Azure SKUs).
    m["unsupported_sku"] = rep["checks"]["sizing"].get("unsupported_sku")
    if "Target_Size" in plan.columns:
        azure = plan["Target_Size"].astype(str).str.startswith("Standard_").sum()
        m["target_cloud_in_plan"] = "AZURE" if azure == len(plan) else ("mixed" if azure else "AWS")
    # Risk-label agreement with the stakeholder's own rule (catches invented labels).
    if "Risk_Score" in plan.columns or "Risk_Level" in plan.columns:
        truth = {r.Server_ID: baseline_risk(r.OS_Version, r.Business_Tags) for r in cleaned.itertuples()}
        col = "Baseline_Risk" if "Baseline_Risk" in plan.columns else "Risk_Score"
        labels = dict(zip(plan["Server_ID"], plan[col]))
        scored = [s for s in truth if isinstance(labels.get(s), str)]
        m["risk_labels_present"] = len(scored)
        m["risk_agreement_with_stakeholder_rule_pct"] = round(
            100 * sum(labels[s] == truth[s] for s in scored) / len(truth), 1) if truth else None
    return {"metrics": m, "report": rep}
