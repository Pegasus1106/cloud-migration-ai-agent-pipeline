"""Static FinOps estimate: current monthly cost vs estimated AWS compute cost."""

from p1.core import catalog as cat


def license_note(os_version: str, catalog: dict) -> str:
    os_l = (os_version or "").lower()
    notes = [text for marker, text in catalog.get("license_not_included_markers", {}).items()
             if marker in os_l]
    return "; ".join(notes)


def price_servers(servers: list[dict], catalog: dict | None = None) -> tuple[dict, dict]:
    """Price every server from its Target_Size. Returns (per-server dict, summary).

    Unknown SKUs are never guessed: the server is left unpriced and reported,
    which the validator treats as an unsupported cost claim.
    """
    catalog = catalog or cat.load_catalog()
    per: dict[str, dict] = {}
    unpriced = []
    tot_cur = tot_aws = tot_right = 0.0
    license_gap = 0
    for s in servers:
        sid = s["Server_ID"]
        sku = s.get("Target_Size")
        cost = cat.monthly_cost(sku, catalog) if sku else None
        current = float(s.get("Current_Monthly_Cost") or 0.0)
        note = license_note(s.get("OS_Version", ""), catalog)
        req_cpu, req_ram = cat.rightsized_requirement(float(s["Allocated_vCPU"]), float(s["Allocated_RAM_GB"]),
                                                      float(s.get("Normalized_CPU_Pct") or 100))
        right = cat.fitting_instances(req_cpu, req_ram, catalog)
        right_sku = right[0]["instance_type"] if right else None
        right_cost = right[0]["monthly_usd"] if right else None
        if cost is None:
            unpriced.append({"server_id": sid, "target_size": sku})
            per[sid] = {"AWS_Monthly_Cost": None, "Savings_Amount": None, "Savings_Pct": None,
                        "License_Note": note, "RightSized_Size": right_sku, "RightSized_Monthly_Cost": right_cost}
            continue
        savings = round(current - cost, 2)
        per[sid] = {
            "AWS_Monthly_Cost": cost,
            "Savings_Amount": savings,
            "Savings_Pct": round(savings / current * 100, 1) if current > 0 else None,
            "License_Note": note,
            "RightSized_Size": right_sku,
            "RightSized_Monthly_Cost": right_cost,
        }
        tot_cur += current
        tot_aws += cost
        tot_right += right_cost if right_cost is not None else cost
        license_gap += bool(note)
    savings_total = round(tot_cur - tot_aws, 2)
    summary = {
        "catalog_version": catalog["catalog_version"],
        "region": catalog["region"],
        "os_assumption": catalog["os_assumption"],
        "hours_per_month": catalog["hours_per_month"],
        "disclaimer": catalog["disclaimer"],
        "servers_priced": len(servers) - len(unpriced),
        "servers_unpriced": unpriced,
        "total_current_monthly": round(tot_cur, 2),
        "total_aws_monthly": round(tot_aws, 2),
        "total_savings_monthly": savings_total,
        "savings_pct": round(savings_total / tot_cur * 100, 1) if tot_cur else None,
        "rightsized_total_aws_monthly": round(tot_right, 2),
        "servers_with_license_not_included": license_gap,
    }
    return per, summary
