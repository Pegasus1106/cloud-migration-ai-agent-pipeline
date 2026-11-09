"""Deterministic plan validator, usable on any plan CSV (B0, B1, or P1).

Implements the Milestone 1 success criteria as checks. Every failed check
becomes a *violation* with a stable ID, a severity, and a routing hint:
  repairable  -> the Repair Agent may fix it with a bounded tool action;
  needs_human -> it goes to the human review queue.
P1's supervisor will not finalize a plan while blocking violations remain.
"""

import math
from collections import defaultdict

import networkx as nx
import pandas as pd

from p1.core import catalog as cat
from p1.core.scheduler import cross_wave_edges, utilization_pct

PRICE_TOLERANCE = 0.011


def _v(vid, category, severity, message, server_ids=(), repairable=False, needs_human=False, **extra):
    return {"id": vid, "category": category, "severity": severity, "message": message,
            "server_ids": list(server_ids), "repairable": repairable, "needs_human": needs_human, **extra}


def early_migration_check(wave_of: dict[str, int], early_ids: set[str], edges: list[tuple[str, str]],
                          transfer: dict[str, float], max_gb: float, first_regular_wave: int = 0) -> dict:
    """An early request is satisfied when the server lands within the minimum
    number of waves needed to move every early request plus its prerequisites
    (servers that call it), plus one wave of packing slack. Requests whose
    prerequisites alone exceed the wave limit are infeasible and must be explained."""
    g = nx.DiGraph()
    g.add_nodes_from(wave_of)
    g.add_edges_from((a, b) for a, b in edges if a in wave_of and b in wave_of)
    closures = {e: (nx.ancestors(g, e) | {e}) for e in early_ids if e in wave_of}
    union = set().union(*closures.values()) if closures else set()
    total = sum(transfer.get(s, 0.0) for s in union)
    bound = first_regular_wave + max(1, math.ceil(total / max_gb)) if closures else first_regular_wave
    rows = []
    for e, clo in sorted(closures.items()):
        clo_gb = sum(transfer.get(s, 0.0) for s in clo)
        feasible = clo_gb <= max_gb
        w = wave_of.get(e)
        rows.append({"server_id": e, "wave": w, "closure_servers": len(clo), "closure_gb": round(clo_gb, 1),
                     "feasible": feasible, "satisfied": bool(feasible and w is not None and w <= bound),
                     "bound_wave": bound,
                     "reason": "" if feasible else f"it and the {len(clo) - 1} servers that call it need "
                                                   f"{clo_gb:,.0f} GB, above the {max_gb:,.0f} GB wave limit"})
    feasible_rows = [r for r in rows if r["feasible"]]
    return {
        "requests": len(rows),
        "feasible": len(feasible_rows),
        "satisfied": sum(r["satisfied"] for r in feasible_rows),
        "satisfied_pct": round(100 * sum(r["satisfied"] for r in feasible_rows) / len(feasible_rows), 1)
        if feasible_rows else None,
        "mean_wave": round(sum(r["wave"] for r in rows if r["wave"] is not None) / len(rows), 2) if rows else None,
        "bound_wave": bound,
        "details": rows,
    }


def validate_plan(
    plan: pd.DataFrame,
    cleaned: pd.DataFrame,
    edges: list[tuple[str, str]],
    max_gb: float,
    label: str = "plan",
    reported_infeasible: list[dict] | None = None,
    approved_exceptions: set[str] | None = None,
    early_ids: set[str] | None = None,
    first_regular_wave: int = 0,
    require_finops: bool = False,
    require_risk: bool = True,
    config_confirmed: bool | None = None,
    excluded_ids: set[str] | None = None,
    catalog: dict | None = None,
) -> dict:
    catalog = catalog or cat.load_catalog()
    approved = set(approved_exceptions or ())
    excluded = set(excluded_ids or ())
    violations: list[dict] = []
    checks: dict = {}

    # ---- schema / completeness ------------------------------------------
    missing_cols = [c for c in ("Server_ID", "Wave_ID") if c not in plan.columns]
    checks["schema"] = {"missing_columns": missing_cols, "rows": len(plan)}
    if missing_cols:
        violations.append(_v("SCHEMA", "schema", "blocking", f"Plan is missing columns {missing_cols}",
                             needs_human=True))
        return {"label": label, "checks": checks, "violations": violations, "passed": False,
                "blocking_count": 1, "metrics": {}}

    plan = plan.copy()
    plan["Server_ID"] = plan["Server_ID"].astype(str)
    active = plan[~plan["Server_ID"].isin(excluded)]
    wave_of = {r.Server_ID: int(r.Wave_ID) for r in active.itertuples() if pd.notna(r.Wave_ID)}
    expected = set(cleaned["Server_ID"].astype(str)) - excluded
    missing = sorted(expected - set(wave_of))
    checks["schema"].update({"servers_expected": len(expected), "servers_with_wave": len(wave_of),
                             "missing_servers": missing, "excluded_servers": sorted(excluded)})
    if missing:
        violations.append(_v("MISSING_SERVERS", "schema", "blocking",
                             f"{len(missing)} servers have no wave", missing, needs_human=True))

    # ---- dependency order -----------------------------------------------
    dep_bad = [(a, b) for a, b in edges if a in wave_of and b in wave_of and wave_of[a] > wave_of[b]]
    checks["dependency"] = {"edges_checked": sum(1 for a, b in edges if a in wave_of and b in wave_of),
                            "violations": len(dep_bad), "examples": dep_bad[:5]}
    for a, b in dep_bad:
        violations.append(_v(f"DEP:{a}->{b}", "dependency", "blocking",
                             f"{a} (wave {wave_of[a]}) calls {b} (wave {wave_of[b]}) but moves later",
                             [a, b], needs_human=True))

    # ---- transfer volume --------------------------------------------------
    transfer = dict(zip(cleaned["Server_ID"].astype(str), cleaned["Transfer_GB"].astype(float)))
    totals: dict[int, float] = defaultdict(float)
    members_of_wave: dict[int, set[str]] = defaultdict(set)
    for sid, w in wave_of.items():
        totals[w] += transfer.get(sid, 0.0)
        members_of_wave[w].add(sid)
    reported = {}
    for grp in reported_infeasible or []:
        reported[grp["group_id"]] = set(grp["members"])
    over = {w: t for w, t in totals.items() if t > max_gb + 1e-6}
    over_rows = []
    for w, t in sorted(over.items()):
        grp_id = next((gid for gid, mem in reported.items() if mem == members_of_wave[w]), None)
        status = "approved" if grp_id in approved else "reported" if grp_id else "unreported"
        over_rows.append({"wave": w, "total_gb": round(t, 1), "status": status, "group_id": grp_id})
        if status == "approved":
            continue
        if status == "reported":
            violations.append(_v(f"VOLUME:{grp_id}", "transfer_volume", "blocking",
                                 f"Indivisible group {grp_id} needs {t:,.0f} GB in wave {w} "
                                 f"(limit {max_gb:,.0f} GB)", sorted(members_of_wave[w]),
                                 needs_human=True, group_id=grp_id, wave=w, total_gb=round(t, 1)))
        else:
            violations.append(_v(f"VOLUME:wave{w}", "transfer_volume", "blocking",
                                 f"Wave {w} carries {t:,.0f} GB, above the {max_gb:,.0f} GB limit, "
                                 "and was not reported as infeasible", sorted(members_of_wave[w]),
                                 needs_human=True, wave=w, total_gb=round(t, 1)))
    checks["transfer_volume"] = {"max_transfer_gb": max_gb, "waves": len(totals), "over_limit": over_rows,
                                 "wave_totals_gb": {str(k): round(v, 1) for k, v in sorted(totals.items())}}

    # ---- sizing safety ------------------------------------------------------
    specs = cleaned.set_index(cleaned["Server_ID"].astype(str))[["Allocated_vCPU", "Allocated_RAM_GB"]]
    if "Target_Size" in plan.columns:
        under, unsupported = [], []
        for r in active.itertuples():
            sku = getattr(r, "Target_Size")
            if r.Server_ID not in specs.index:
                continue
            need_cpu, need_ram = specs.loc[r.Server_ID]
            s = cat.spec(str(sku), catalog) if pd.notna(sku) else None
            if s is None:
                unsupported.append((r.Server_ID, sku))
            elif need_cpu > s["vcpu"] or need_ram > s["ram_gb"]:
                under.append((r.Server_ID, sku, need_cpu, need_ram))
        checks["sizing"] = {"undersized": len(under), "unsupported_sku": len(unsupported),
                            "examples": [{"server_id": a, "target_size": b, "needs_vcpu": c, "needs_ram_gb": d}
                                         for a, b, c, d in under[:5]]}
        for sid, sku, c, r_ in under:
            violations.append(_v(f"SIZING:{sid}", "sizing", "blocking",
                                 f"{sid} needs {c:.0f} vCPU / {r_:.0f} GB but {sku} provides "
                                 f"{cat.spec(sku, catalog)['vcpu']} / {cat.spec(sku, catalog)['ram_gb']}",
                                 [sid], repairable=True, needs_vcpu=c, needs_ram_gb=r_, target_size=sku))
        for sid, sku in unsupported:
            violations.append(_v(f"SKU:{sid}", "sizing", "blocking",
                                 f"{sid} uses '{sku}', which is not in the pricing catalog",
                                 [sid], repairable=True, target_size=sku))
    else:
        checks["sizing"] = {"skipped": "no Target_Size column"}

    # ---- FinOps arithmetic ----------------------------------------------------
    if "AWS_Monthly_Cost" in plan.columns:
        bad = []
        for r in active.itertuples():
            expected_cost = cat.monthly_cost(str(r.Target_Size), catalog) if pd.notna(r.Target_Size) else None
            cost = r.AWS_Monthly_Cost
            if pd.isna(cost) or expected_cost is None or abs(float(cost) - expected_cost) > PRICE_TOLERANCE:
                bad.append(r.Server_ID)
        checks["finops"] = {"servers_checked": len(active), "unsupported_cost_claims": len(bad)}
        if require_finops:
            for sid in bad:
                violations.append(_v(f"PRICE:{sid}", "pricing", "blocking",
                                     f"{sid} has no catalog-backed AWS cost", [sid], repairable=True))
    else:
        checks["finops"] = {"skipped": "no AWS_Monthly_Cost column"}

    # ---- risk completeness ----------------------------------------------------
    risk_col = "Risk_Level" if "Risk_Level" in plan.columns else "Risk_Score" if "Risk_Score" in plan.columns else None
    if risk_col:
        missing_risk = active[active[risk_col].isna()]["Server_ID"].tolist()
        checks["risk"] = {"column": risk_col, "with_risk": len(active) - len(missing_risk), "total": len(active)}
        if require_risk and missing_risk:
            violations.append(_v("RISK_MISSING", "risk", "blocking",
                                 f"{len(missing_risk)} servers have no risk level", missing_risk, needs_human=True))
    else:
        checks["risk"] = {"column": None, "with_risk": 0, "total": len(active)}

    # ---- early migration ----------------------------------------------------
    early = early_migration_check(wave_of, set(early_ids or ()), edges, transfer, max_gb, first_regular_wave)
    checks["early_migration"] = early
    for r in early["details"]:
        if r["feasible"] and not r["satisfied"]:
            violations.append(_v(f"EARLY:{r['server_id']}", "early_migration", "blocking",
                                 f"{r['server_id']} was requested early but is in wave {r['wave']} "
                                 f"(expected by wave {r['bound_wave']})", [r["server_id"]], needs_human=True))
        elif not r["feasible"]:
            violations.append(_v(f"EARLY_INFEASIBLE:{r['server_id']}", "early_migration", "warning",
                                 f"Early request for {r['server_id']} cannot be met: {r['reason']}",
                                 [r["server_id"]], needs_human=True))

    if config_confirmed is False:
        violations.append(_v("CONFIG_UNCONFIRMED", "config", "blocking",
                             "Planning constraints were not confirmed by the user", needs_human=True))

    # Over-limit (indivisible) waves and the deliberately small pilot wave are not
    # capacity-packing outcomes, so they are left out of the utilization metric.
    exclude_waves = {r["wave"] for r in over_rows} | set(range(first_regular_wave))
    metrics = {
        "servers": len(wave_of),
        "wave_count": len(totals),
        "dependency_edges": checks["dependency"]["edges_checked"],
        "cross_wave_edges": cross_wave_edges(edges, wave_of),
        "utilization_pct": utilization_pct(dict(totals), max_gb, exclude_waves),
        "waves_over_limit": len(over_rows),
        "over_limit_unreported": sum(r["status"] == "unreported" for r in over_rows),
        "over_limit_approved": sum(r["status"] == "approved" for r in over_rows),
        "dependency_violations": len(dep_bad),
        "undersized": checks["sizing"].get("undersized"),
        "early_satisfied_pct": early["satisfied_pct"],
        "early_mean_wave": early["mean_wave"],
        "risk_completeness_pct": round(100 * checks["risk"]["with_risk"] / max(1, checks["risk"]["total"]), 1),
    }
    m = metrics["dependency_edges"]
    metrics["cross_wave_edge_pct"] = round(100 * metrics["cross_wave_edges"] / m, 1) if m else None
    blocking = [v for v in violations if v["severity"] == "blocking"]
    return {"label": label, "checks": checks, "violations": violations,
            "blocking_count": len(blocking), "passed": not blocking, "metrics": metrics}
