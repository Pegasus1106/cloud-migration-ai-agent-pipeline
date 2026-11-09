"""Weighted, reproducible risk scoring (Milestone 1: "weighted rules plus
constrained explanation").

Each server gets a list of factor codes with concrete evidence. The score is the
sum of factor weights; the level is High (>= 40), Medium (20-39) or Low (< 20).
The stakeholder baseline's binary rule (legacy OS marker or compliance tag ->
High) is a strict subset: every legacy marker is an end-of-life OS (40) and every
compliance tag scores 40, so nothing the baseline called High becomes Low.
The LLM never assigns scores; it only explains these factors (see
p1/agents/risk_explainer.py), and its explanations are verified against them.
"""

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from p1 import config

FACTORS = {
    "OS_EOL":         (40, "Operating system is past the end of standard vendor support"),
    "OS_EOL_SOON":    (15, "Operating system support ends within 12 months"),
    "OS_UNKNOWN":     (10, "Operating system not found in the lifecycle table"),
    "COMPLIANCE":     (40, "Handles regulated data (PCI-DSS / HIPAA / SOX)"),
    "HIGH_INBOUND":   (15, "Many other servers depend on it (inbound degree >= 3)"),
    "IN_CYCLE":       (10, "Part of a circular dependency that must move as one unit"),
    "LARGE_DATA":     (10, "Large data volume to transfer (>= 2,000 GB)"),
    "DATA_GAPS":      (10, "Discovery record had missing values that were imputed"),
    "EXTERNAL":       (10, "Externally facing service (customer-visible downtime)"),
}
COMPLIANCE_TAGS = ("PCI", "HIPAA", "SOX")
BASELINE_LEGACY_MARKERS = ("2008", "RHEL 5", "6.4")  # stakeholder baseline rule
HIGH_THRESHOLD, MEDIUM_THRESHOLD = 40, 20
LEVEL_RANK = {"Low": 0, "Medium": 1, "High": 2}


@lru_cache(maxsize=4)
def load_lifecycle(path: str = str(config.OS_LIFECYCLE_PATH)) -> dict:
    return json.loads(Path(path).read_text())


def match_os(os_version: str, lifecycle: dict) -> dict | None:
    """Products are listed most-specific first (e.g. '2008 r2' before '2008')."""
    text = " ".join((os_version or "").lower().replace("red hat enterprise linux", "rhel").split())
    for prod in lifecycle["products"]:
        if prod["match"] in text:
            return prod
    return None


def baseline_risk(os_version: str, tags: str) -> str:
    """The stakeholder baseline's binary rule, kept for comparisons."""
    legacy = any(m in str(os_version) for m in BASELINE_LEGACY_MARKERS)
    compliance = any(t in str(tags).upper() for t in COMPLIANCE_TAGS)
    return "High" if (legacy or compliance) else "Low"


def score_server(server: dict, lifecycle: dict | None = None) -> dict:
    lifecycle = lifecycle or load_lifecycle()
    as_of = date.fromisoformat(lifecycle["as_of"])
    soon = as_of + timedelta(days=int(lifecycle.get("eol_soon_days", 365)))
    factors: list[dict] = []

    os_version = str(server.get("OS_Version", ""))
    prod = match_os(os_version, lifecycle)
    if prod is None:
        factors.append({"code": "OS_UNKNOWN", "evidence": f"'{os_version}' has no lifecycle entry"})
    else:
        end = date.fromisoformat(prod["support_end"])
        if end < as_of:
            factors.append({"code": "OS_EOL", "evidence": f"{prod['label']} standard support ended {prod['support_end']}"})
        elif end <= soon:
            factors.append({"code": "OS_EOL_SOON", "evidence": f"{prod['label']} standard support ends {prod['support_end']}"})

    tags = str(server.get("Business_Tags", ""))
    hit = [t for t in COMPLIANCE_TAGS if t in tags.upper()]
    if hit:
        factors.append({"code": "COMPLIANCE", "evidence": f"tagged {tags}"})
    if "EXTERNAL" in tags.upper():
        factors.append({"code": "EXTERNAL", "evidence": f"tagged {tags}"})

    inbound = int(server.get("Inbound_Degree") or 0)
    if inbound >= 3:
        factors.append({"code": "HIGH_INBOUND", "evidence": f"{inbound} servers call it"})
    scc_size = int(server.get("SCC_Size") or 1)
    if scc_size > 1:
        factors.append({"code": "IN_CYCLE", "evidence": f"in a {scc_size}-server dependency cycle"})
    transfer = float(server.get("Transfer_GB") or 0)
    if transfer >= 2000:
        factors.append({"code": "LARGE_DATA", "evidence": f"{transfer:,.0f} GB to transfer"})
    if bool(server.get("requires_manual_audit")):
        factors.append({"code": "DATA_GAPS", "evidence": "missing CPU/RAM/storage/OS values were imputed"})

    score = sum(FACTORS[f["code"]][0] for f in factors)
    level = "High" if score >= HIGH_THRESHOLD else "Medium" if score >= MEDIUM_THRESHOLD else "Low"
    return {"Risk_Score": score, "Risk_Level": level, "Risk_Factors": factors,
            "Baseline_Risk": baseline_risk(os_version, tags)}


def template_explanation(server_id: str, factors: list[dict]) -> str:
    """Deterministic fallback explanation built only from the factors."""
    if not factors:
        return f"{server_id}: no risk factors found; standard rehost risk."
    parts = [f"{FACTORS[f['code']][1].lower()} ({f['evidence']})" for f in factors]
    return f"{server_id}: " + "; ".join(parts) + "."
