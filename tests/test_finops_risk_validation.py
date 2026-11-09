"""FinOps arithmetic (Milestone 1 metric: 100%), risk rules, and validator detection."""

import pandas as pd
import pytest

from p1.core import catalog as cat
from p1.core import finops, risk
from p1.core.data import load_and_normalize
from p1.core.validation import validate_plan

from conftest import DATASETS, make_server


def test_monthly_cost_matches_reference_formula_for_every_sku():
    c = cat.load_catalog()
    for sku, s in c["instances"].items():
        assert cat.monthly_cost(sku, c) == round(s["hourly_usd"] * c["hours_per_month"], 2)


def test_price_servers_savings_license_note_and_unknown_sku():
    servers = [make_server("A", Target_Size="m5.2xlarge", Current_Monthly_Cost=700.0, OS_Version="Windows Server 2019"),
               make_server("B", Target_Size="z9.mega", Current_Monthly_Cost=100.0)]
    per, summary = finops.price_servers(servers)
    assert per["A"]["AWS_Monthly_Cost"] == 280.32 and per["A"]["Savings_Amount"] == pytest.approx(419.68)
    assert "Windows Server license not included" in per["A"]["License_Note"]
    assert per["B"]["AWS_Monthly_Cost"] is None and summary["servers_unpriced"] == [{"server_id": "B",
                                                                                     "target_size": "z9.mega"}]


def test_fitting_instances_are_sorted_and_sufficient():
    rows = cat.fitting_instances(32, 256)
    assert rows[0]["instance_type"] == "r5.8xlarge"
    assert all(r["vcpu"] >= 32 and r["ram_gb"] >= 256 for r in rows)
    assert [r["monthly_usd"] for r in rows] == sorted(r["monthly_usd"] for r in rows)


@pytest.mark.parametrize("os_v,label", [("Windows Server 2008 R2", "Windows Server 2008 R2"),
                                        ("Windows Server 2008", "Windows Server 2008"),
                                        ("Oracle Linux 6.4", "Oracle Linux 6"), ("SQL Server 2016", "SQL Server 2016"),
                                        ("RHEL 5", "RHEL 5")])
def test_lifecycle_matching_prefers_specific_versions(os_v, label):
    assert risk.match_os(os_v, risk.load_lifecycle())["label"] == label


def test_risk_factors_and_levels():
    r = risk.score_server(make_server("A", os_v="CentOS 7", tags="PCI-DSS", Inbound_Degree=4, SCC_Size=2))
    assert {f["code"] for f in r["Risk_Factors"]} == {"OS_EOL", "COMPLIANCE", "HIGH_INBOUND", "IN_CYCLE"}
    assert r["Risk_Level"] == "High" and r["Baseline_Risk"] == "High"
    soon = risk.score_server(make_server("B", os_v="Windows Server 2016"))
    assert [f["code"] for f in soon["Risk_Factors"]] == ["OS_EOL_SOON"] and soon["Risk_Level"] == "Low"


@pytest.mark.parametrize("path", DATASETS, ids=lambda p: p.name)
def test_every_baseline_high_server_is_still_high(path):
    df, _, _ = load_and_normalize(path)
    for s in df.to_dict("records"):
        r = risk.score_server(s)
        if r["Baseline_Risk"] == "High":
            assert r["Risk_Level"] == "High", s["Server_ID"]


def _plan(servers, waves):
    df = pd.DataFrame(servers)
    df["Wave_ID"] = df["Server_ID"].map(waves)
    return df


def test_validator_detects_injected_violations():
    servers = [make_server("A", gb=2000, Target_Size="t3.large", cpu=16, ram=64),
               make_server("B", gb=2000, Target_Size="t3.large"), make_server("C", gb=100, Target_Size="t3.large")]
    for s in servers:
        s["AWS_Monthly_Cost"] = cat.monthly_cost("t3.large")
    servers[2]["AWS_Monthly_Cost"] = 1.0  # unsupported cost claim
    plan = _plan(servers, {"A": 1, "B": 0, "C": 0})  # A calls B but moves later; wave 0 = 2100 GB ok
    plan.loc[plan.Server_ID == "C", "Wave_ID"] = 0
    rep = validate_plan(plan, plan, [("A", "B")], 2050, require_finops=True, early_ids={"C"})
    cats = {v["category"] for v in rep["violations"] if v["severity"] == "blocking"}
    assert {"dependency", "transfer_volume", "sizing", "pricing"} <= cats
    assert not rep["passed"]
    assert next(v for v in rep["violations"] if v["category"] == "sizing")["repairable"]


def test_reported_infeasible_group_needs_approval_then_passes():
    servers = [make_server("A", gb=5000, Target_Size="t3.large")]
    plan = _plan(servers, {"A": 0})
    grp = [{"group_id": "SCC:A", "members": ["A"]}]
    rep = validate_plan(plan, plan, [], 3000, reported_infeasible=grp)
    assert rep["violations"][0]["needs_human"] and not rep["passed"]
    rep2 = validate_plan(plan, plan, [], 3000, reported_infeasible=grp, approved_exceptions={"SCC:A"})
    assert rep2["passed"] and rep2["metrics"]["over_limit_approved"] == 1
