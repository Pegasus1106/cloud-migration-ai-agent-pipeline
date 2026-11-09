"""Agent guardrails that can be tested without calling a model."""

import pytest

from p1.agents import repair_agent
from p1.core import catalog as cat
from p1.core.cypher_guard import check_read_only
from p1.core.verify import verify_mentions, verify_numbers, verify_risk_explanation
from p1.state import RunConfig
from p1.supervisor import RunOptions, Supervisor

from conftest import ROOT, make_server


@pytest.mark.parametrize("q", ["MATCH (s:Server) RETURN s.id", "MATCH (s:Server) WHERE s.app = 'CREATE' RETURN count(s)",
                               "MATCH (a)-[:COMMUNICATES_WITH]->(b) RETURN a.id, b.id LIMIT 5"])
def test_cypher_guard_accepts_reads(q):
    assert check_read_only(q)[0]


@pytest.mark.parametrize("q", ["MATCH (s) DETACH DELETE s", "MATCH (s) SET s.wave_id = 0 RETURN s",
                               "CREATE (x:Server {id: 'X'})", "MATCH (s) RETURN s; MATCH (t) DELETE t",
                               "CALL gds.wcc.write('g', {writeProperty: 'x'})", "CALL apoc.periodic.iterate('a','b',{})"])
def test_cypher_guard_rejects_writes(q):
    assert not check_read_only(q)[0]


def test_risk_explanation_verifier_catches_hallucinated_factors():
    factors = [{"code": "OS_EOL", "evidence": "RHEL 5 ended 2017"}]
    assert verify_risk_explanation("S1", "RHEL 5 is end-of-life.", ["OS_EOL"], factors) == []
    errs = verify_risk_explanation("S1", "RHEL 5 is end-of-life and holds PCI data.", ["OS_EOL", "COMPLIANCE"], factors)
    assert any("do not apply" in e for e in errs)


def test_number_and_name_verifiers():
    assert verify_numbers("Wave 2 moves 14 servers (2,980 GB).", [2, 14, 2980], []) == []
    assert verify_numbers("Wave 2 moves 15 servers.", [2, 14], []) == ["15"]
    assert verify_numbers("hr-web-01 moves in Wave 0.", [0], ["hr-web-01"]) == []
    assert verify_mentions("Moves hr-web-01 and crm-db-01.", {"hr-web-01"}, {"hr-web-01", "crm-db-01"}) == ["crm-db-01"]


def _session():
    servers = {"BIG": make_server("BIG", cpu=32, ram=256, Target_Size="m5.4xlarge", Normalized_CPU_Pct=70),
               "OK": make_server("OK")}
    viol = [{"id": "SIZING:BIG", "category": "sizing", "message": "", "server_ids": ["BIG"]}]
    return repair_agent.RepairSession(viol, servers, cat.load_catalog())


def test_repair_tools_reject_unsafe_actions():
    s = _session()
    tools = {t.name: t for t in repair_agent._tools(s)}
    assert tools["resize_server"].invoke({"server_id": "BIG", "instance_type": "m5.8xlarge", "reason": "x"}).startswith("REJECTED")
    assert tools["resize_server"].invoke({"server_id": "BIG", "instance_type": "made.up", "reason": "x"}).startswith("REJECTED")
    assert tools["resize_server"].invoke({"server_id": "OK", "instance_type": "t3.large", "reason": "x"}).startswith("REJECTED")
    assert '"resized"' in tools["resize_server"].invoke({"server_id": "BIG", "instance_type": "r5.8xlarge", "reason": "x"})
    assert s.rejected_actions == 3


def test_rule_fallback_repair_policy():
    out = repair_agent.run_repair(_session().violations, _session().servers, cat.load_catalog(), None, use_llm=False)
    assert out["mode"] == "fallback" and out["resizes"][0]["to"] == "r5.8xlarge"


def test_supervisor_refuses_unconfirmed_config():
    with pytest.raises(PermissionError):
        Supervisor(ROOT / "data" / "sample.csv", RunConfig(confirmed=False), RunOptions(use_llm=False))


def test_end_to_end_rule_only_with_interrupt_and_resume(tmp_path):
    cfg = RunConfig(confirmed=True, confirmed_by="test")
    sup = Supervisor(ROOT / "data" / "sample.csv", cfg,
                     RunOptions(use_llm=False, use_neo4j=False, review_mode="interactive", out_dir=tmp_path),
                     run_id="pytest-run")
    events = list(sup.stream())
    assert events[-1]["status"] == "waiting" and sup.interrupt_payload
    items = sup.interrupt_payload["items"]
    decisions = {i["id"]: {"action": "approve_exception" if i["kind"] == "over_limit_group" else "acknowledge"}
                 for i in items}
    list(sup.resume(decisions))
    state = sup.state
    assert state["status"] == "ready_for_approval"
    assert state["validation"]["metrics"]["undersized"] == 0  # fixed by the repair loop (rule fallback)
    assert all(d["decided_by"] == "human" for d in state["review_decisions"])
    assert (tmp_path / "plan.csv").exists() and (tmp_path / "run_log.json").exists()


def test_reject_decision_stops_the_run(tmp_path):
    sup = Supervisor(ROOT / "data" / "sample.csv", RunConfig(confirmed=True),
                     RunOptions(use_llm=False, use_neo4j=False, review_mode="interactive", out_dir=tmp_path))
    list(sup.stream())
    first = sup.interrupt_payload["items"][0]["id"]
    list(sup.resume({first: {"action": "reject_plan"}}))
    assert sup.state["status"] == "rejected"
