"""Wave packing: dependency order, capacity, indivisible groups, determinism, policies."""

import random

from p1.core.planning import plan_b1, plan_p1
from p1.core.validation import validate_plan
from p1.state import RunConfig

import pandas as pd

from conftest import make_server


def _chain(n, gb=100.0):
    servers = [make_server(f"S{i:02d}", gb=gb) for i in range(n)]
    edges = [(f"S{i:02d}", f"S{i + 1:02d}") for i in range(n - 1)]
    return servers, edges


def test_same_wave_packing_keeps_a_fitting_chain_together():
    servers, edges = _chain(10)
    res = plan_b1(servers, edges, 3000)
    assert set(res.wave_of_server.values()) == {0}


def test_strict_packing_reproduces_prototype_one_wave_per_link():
    servers, edges = _chain(10)
    res = plan_b1(servers, edges, 3000, allow_same_wave=False)
    assert sorted(set(res.wave_of_server.values())) == list(range(10))


def test_capacity_dependency_and_oversized_group_reporting():
    servers = [make_server("A", gb=2000), make_server("B", gb=2000), make_server("C", gb=5000), make_server("D", gb=500)]
    edges = [("A", "B"), ("B", "C")]
    res = plan_p1(servers, edges, RunConfig(max_transfer_gb=3000))
    w = res.wave_of_server
    assert w["A"] <= w["B"] <= w["C"]
    assert [g["group_id"] for g in res.infeasible] == ["SCC:C"]
    assert all(t <= 3000 for wave, t in res.wave_totals.items() if wave != w["C"])
    assert res.wave_totals[w["C"]] == 5000  # placed alone


def test_plans_do_not_depend_on_input_order():
    servers = [make_server(f"S{i:02d}", gb=random.Random(i).randint(50, 900), app=f"app{i % 4}") for i in range(40)]
    edges = [(f"S{i:02d}", f"S{(i * 7 + 3) % 40:02d}") for i in range(40) if i != (i * 7 + 3) % 40]
    base = plan_p1(servers, edges, RunConfig()).wave_of_server
    shuffled = servers[:]
    random.Random(1).shuffle(shuffled)
    assert plan_p1(shuffled, list(reversed(edges)), RunConfig()).wave_of_server == base


def test_early_requests_and_their_callers_go_first():
    servers = [make_server(f"S{i}", gb=900) for i in range(6)]
    servers[5]["Early_Migration_Requested"] = True
    edges = [("S4", "S5")]  # S4 calls S5, so S4 must move no later than S5
    res = plan_p1(servers, edges, RunConfig(max_transfer_gb=2000))
    assert res.wave_of_server["S5"] == 0 and res.wave_of_server["S4"] == 0
    b1 = plan_b1(servers, edges, 2000)
    plan = pd.DataFrame(servers).assign(Wave_ID=lambda d: d.Server_ID.map(res.wave_of_server))
    rep = validate_plan(plan, plan, edges, 2000, early_ids={"S5"})
    assert rep["checks"]["early_migration"]["satisfied_pct"] == 100.0
    assert b1.wave_of_server["S5"] >= res.wave_of_server["S5"]


def test_pilot_wave_only_takes_whole_low_risk_isolated_apps():
    servers = [make_server("P1", gb=100, app="pilot"), make_server("P2", gb=100, app="pilot"),
               make_server("H1", gb=100, app="risky", Risk_Level="High"), make_server("X1", gb=100, app="big"),
               make_server("X2", gb=100, app="big")]
    servers[1]["WCC_Id"] = "WCC:P1"
    edges = [("P1", "P2"), ("X1", "X2")]
    for s in servers[3:]:
        s["WCC_Id"] = "WCC:X1"
    cfg = RunConfig(pilot_wave=True, pilot_max_servers=2, pilot_max_gb=500)
    res = plan_p1(servers, edges, cfg)
    assert res.pilot["servers"] == ["P1", "P2"] or res.pilot["servers"] == ["X1", "X2"]
    assert "H1" not in res.pilot["servers"]
    assert min(w for s, w in res.wave_of_server.items() if s not in res.pilot["servers"]) == 1


def test_excluded_servers_are_not_planned():
    servers, edges = _chain(4)
    res = plan_p1(servers, edges, RunConfig(), excluded_ids={"S01"})
    assert "S01" not in res.wave_of_server and res.excluded == ["S01"]
