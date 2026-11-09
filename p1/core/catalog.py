"""Static AWS instance catalog: sizing and price lookups.

Primary sizing keeps the stakeholder baseline's tier matrix (so B0, B1 and P1
size identically). Servers larger than the matrix's top tier are NOT silently
undersized any more: they fail validation and are routed to the Repair Agent,
which chooses a fitting SKU with `fitting_instances` (catalog-wide lookup).
"""

import json
import math
from functools import lru_cache
from pathlib import Path

from p1 import config

# Stakeholder baseline (B0) sizing matrices: (max vCPU, max RAM GB, instance).
AWS_MATRIX = [(2, 8, "t3.large"), (4, 16, "t3.xlarge"), (8, 32, "m5.2xlarge")]
AWS_MATRIX_TOP = "m5.4xlarge"
AZURE_MATRIX = [(2, 8, "Standard_D2s_v3"), (4, 16, "Standard_D4s_v3"), (8, 32, "Standard_D8s_v3")]
AZURE_MATRIX_TOP = "Standard_D16s_v3"


@lru_cache(maxsize=4)
def load_catalog(path: str = str(config.PRICING_CATALOG_PATH)) -> dict:
    return json.loads(Path(path).read_text())


def size_by_matrix(cpu: float, ram: float, cloud: str = "AWS") -> str:
    matrix, top = (AZURE_MATRIX, AZURE_MATRIX_TOP) if cloud.upper() == "AZURE" else (AWS_MATRIX, AWS_MATRIX_TOP)
    for max_cpu, max_ram, sku in matrix:
        if cpu <= max_cpu and ram <= max_ram:
            return sku
    return top


def spec(sku: str, catalog: dict | None = None) -> dict | None:
    return (catalog or load_catalog())["instances"].get(sku)


def fits(sku: str, cpu: float, ram: float, catalog: dict | None = None) -> bool:
    s = spec(sku, catalog)
    return bool(s) and s["vcpu"] >= cpu and s["ram_gb"] >= ram


def monthly_cost(sku: str, catalog: dict | None = None) -> float | None:
    cat = catalog or load_catalog()
    s = cat["instances"].get(sku)
    return None if s is None else round(s["hourly_usd"] * cat["hours_per_month"], 2)


def fitting_instances(min_vcpu: float, min_ram: float, catalog: dict | None = None,
                      family: str | None = None) -> list[dict]:
    """All catalog SKUs that satisfy the requirement, cheapest first."""
    cat = catalog or load_catalog()
    rows = []
    for sku, s in cat["instances"].items():
        if s["vcpu"] >= min_vcpu and s["ram_gb"] >= min_ram and (family is None or s.get("family") == family):
            rows.append({"instance_type": sku, "vcpu": s["vcpu"], "ram_gb": s["ram_gb"],
                         "family": s.get("family"), "hourly_usd": s["hourly_usd"],
                         "monthly_usd": round(s["hourly_usd"] * cat["hours_per_month"], 2)})
    return sorted(rows, key=lambda r: (r["monthly_usd"], r["instance_type"]))


def rightsized_requirement(alloc_vcpu: float, alloc_ram: float, normalized_cpu_pct: float,
                           headroom: float = 1.3) -> tuple[float, float]:
    """Usage-based requirement (informational, stakeholder's over-provisioning
    question): CPU scaled by observed utilization with 30% headroom, at least
    2 vCPU; memory kept at allocation because no memory telemetry exists."""
    vcpu = max(2.0, math.ceil(alloc_vcpu * min(1.0, normalized_cpu_pct / 100.0 * headroom)))
    return float(vcpu), float(alloc_ram)
