"""Discovery contract + inbound derivation (Milestone 1 metric: 100%)."""

import pandas as pd
import pytest

from p1.core.data import DataContractError, normalize, parse_connections, load_and_normalize
from p1.core.graph_local import components_local

from conftest import DATASETS


def test_parse_connections_handles_malformed_and_empty():
    assert parse_connections("SRV-2:8080:150|SRV-3:1433:20") == ([("SRV-2", "8080", "150"), ("SRV-3", "1433", "20")], 0)
    assert parse_connections("SRV-2:8080|") == ([], 1)
    assert parse_connections(float("nan")) == ([], 0)


def _raw(**over):
    row = {"Server_ID": "S1", "Hostname": "h1", "OS_Version": "RHEL 8", "Allocated_vCPU": 2, "Allocated_RAM_GB": 8,
           "Peak_CPU_Pct": 50, "Avg_CPU_Pct": 10, "Storage_GB": 120, "App_Name": "A", "Business_Tags": "SOX",
           "Outbound_Connections": "S2:1433:10|GHOST:80:1"}
    row.update(over)
    other = dict(row, Server_ID="S2", Hostname="h2", Outbound_Connections="")
    return pd.DataFrame([row, other, dict(row)])  # third row duplicates S1


def test_normalize_contract_defaults_and_quality_report():
    df, edges, q = normalize(_raw(Allocated_RAM_GB=None))
    assert len(df) == 2 and q["duplicate_server_ids"] == ["S1"]
    s1 = df.set_index("Server_ID").loc["S1"]
    assert s1["Transfer_GB"] == 120  # transfer defaults to storage, not a constant
    assert bool(s1["requires_manual_audit"]) and q["audit_details"]["S1"] == ["Allocated_RAM_GB"]
    assert s1["Normalized_CPU_Pct"] == pytest.approx(50 * 0.7 + 10 * 0.3)
    assert edges == [{"src": "S1", "dst": "S2", "port": "1433", "bandwidth": 10.0}]
    assert q["unknown_dependency_endpoints"] == ["GHOST"]
    assert set(q["contract_fields_missing_from_source"]) == {"Transfer_GB", "Current_Monthly_Cost",
                                                              "Early_Migration_Requested"}


def test_missing_required_column_is_rejected():
    with pytest.raises(DataContractError):
        normalize(_raw().drop(columns=["Outbound_Connections"]))


@pytest.mark.parametrize("path", DATASETS, ids=lambda p: p.name)
def test_inbound_derivation_matches_outbound_edges_exactly(path):
    df, edges, _ = load_and_normalize(path)
    pairs = [(e["src"], e["dst"]) for e in edges]
    _, _, degrees = components_local(list(df["Server_ID"]), pairs, {}, 3000)
    expected_in = {s: 0 for s in df["Server_ID"]}
    for _, dst in set(pairs):
        expected_in[dst] += 1
    assert {s: d[0] for s, d in degrees.items()} == expected_in
    assert sum(d[0] for d in degrees.values()) == sum(d[1] for d in degrees.values()) == len(set(pairs))


def test_components_have_stable_labels_and_detect_cycles():
    ids = ["B", "A", "C", "D"]
    per, stats, _ = components_local(ids, [("A", "B"), ("B", "A"), ("C", "D")], {s: 10 for s in ids}, 3000)
    assert per["A"]["SCC_Id"] == per["B"]["SCC_Id"] == "SCC:A" and per["A"]["SCC_Size"] == 2
    assert per["C"]["WCC_Id"] == per["D"]["WCC_Id"] == "WCC:C"
    assert stats["cycles"] == [{"scc_id": "SCC:A", "members": ["A", "B"]}]
