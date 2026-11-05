"""
Dataset builder — Synthetic Benchmark (D2)
============================================
Generates a ~100-server synthetic discovery CSV with the same schema as
the stakeholder sample, but deliberately includes one oversized connected
component (the "DataPlatform" cluster) whose combined Transfer_GB exceeds
the 3,000 GB wave limit. This is required to exercise the community-
detection split that B0 and B1 cannot perform. Generation is deterministic
(fixed seed) and documented so the dataset is reproducible.

Run: python generate_synthetic_data.py
Output: data/synthetic_83.csv (~83 servers; see cluster list below for the
exact composition — the name reflects the actual generated count rather
than a round number).
"""

import random
from pathlib import Path

import pandas as pd

random.seed(42)

OUT = Path(__file__).parent / "synthetic_83.csv"

rows = []
edges_by_id: dict[str, list[str]] = {}
counter = 1


def new_id() -> str:
    global counter
    sid = f"SYN-{counter:03d}"
    counter += 1
    return sid


def add_edge(src: str, dst: str, port: int, bandwidth: int) -> None:
    edges_by_id.setdefault(src, []).append(f"{dst}:{port}:{bandwidth}")


def add_server(hostname, os_version, vcpu, ram, peak, avg, storage, app, tags, early=False):
    sid = new_id()
    rows.append({
        "Server_ID": sid,
        "Hostname": hostname,
        "OS_Version": os_version,
        "Allocated_vCPU": vcpu,
        "Allocated_RAM_GB": ram,
        "Peak_CPU_Pct": peak,
        "Avg_CPU_Pct": avg,
        "Storage_GB": storage,
        "App_Name": app,
        "Business_Tags": tags,
        "Transfer_GB": storage,
        "Current_Monthly_Cost": round(40 * vcpu + 10 * ram + 0.05 * storage + 50, 2),
        "Early_Migration_Requested": early,
    })
    return sid


def build_chain(prefix, n, os_version, cpu, ram, storage_range, app, tags, early=False):
    """n servers in a dependency chain: 0 -> 1 -> 2 -> ... -> n-1."""
    ids = []
    for i in range(n):
        storage = random.randint(*storage_range)
        sid = add_server(
            f"{prefix}-{i+1:02d}", os_version, cpu, ram,
            random.randint(40, 90), random.randint(15, 60),
            storage, app, tags, early,
        )
        ids.append(sid)
    for i in range(n - 1):
        add_edge(ids[i], ids[i + 1], random.choice([1433, 5432, 3306, 8080]), random.randint(50, 500))
    return ids


def build_mesh_cluster(prefix, n, os_choices, cpu, ram, storage_range, app, tags, extra_edges_per_node=2):
    """n servers, chained plus forward-only extra edges -> one large dense
    WCC that stays acyclic (SCCs remain singletons). Edges only point to
    a strictly later index so no cycle is introduced; this keeps the
    component splittable at fine granularity while still being one large
    WCC when considered as undirected (which is what WCC/community
    detection operate on)."""
    ids = []
    for i in range(n):
        storage = random.randint(*storage_range)
        sid = add_server(
            f"{prefix}-{i+1:03d}", random.choice(os_choices), cpu, ram,
            random.randint(50, 98), random.randint(20, 85),
            storage, app, tags,
        )
        ids.append(sid)
    for i in range(n - 1):
        add_edge(ids[i], ids[i + 1], 5432, random.randint(200, 900))
    for i, sid in enumerate(ids):
        for _ in range(extra_edges_per_node):
            if i >= n - 1:
                continue
            offset = random.randint(1, min(5, n - 1 - i))
            target = ids[i + offset]
            if target != sid:
                add_edge(sid, target, random.choice([5432, 3306, 9092]), random.randint(100, 700))
    return ids


def build_isolated(prefix, n, os_version, cpu, ram, storage_range, app, tags):
    for i in range(n):
        storage = random.randint(*storage_range)
        add_server(
            f"{prefix}-{i+1:02d}", os_version, cpu, ram,
            random.randint(1, 20), random.randint(0, 8),
            storage, app, tags,
        )


# 1. Finance ERP — SOX, medium chain (~6 servers, ~1,200 GB total: fits one wave)
build_chain("fin-erp", 6, "RHEL 8", 8, 32, (150, 300), "Finance_ERP", "SOX")

# 2. Payment Gateway — PCI-DSS, small chain (3 servers)
build_chain("pay", 3, "Oracle Linux 7", 8, 32, (200, 500), "Payment_Gateway", "PCI-DSS")

# 3. CRM — external facing, early-migration requested (2 servers)
build_chain("crm2", 2, "Ubuntu 22.04", 4, 16, (100, 300), "CRM", "External_Facing", early=True)

# 4. E-commerce — PCI-DSS, modern linux (8 servers)
build_chain("ecom", 8, "Ubuntu 22.04", 4, 16, (80, 200), "E-Commerce", "PCI-DSS")

# 5. DataPlatform mega-cluster — intentionally oversized WCC (30 servers,
#    dense interconnect, ~300-900GB each => total well above 3,000 GB).
build_mesh_cluster(
    "dp", 30,
    ["RHEL 8", "CentOS 7", "Ubuntu 20.04"],
    16, 64, (300, 900), "Data_Platform", "Data_Platform",
    extra_edges_per_node=2,
)

# 6. Active Directory / Core Infra — small, paired (4 servers)
ad = build_chain("ad", 2, "Windows Server 2019", 2, 8, (80, 120), "Active_Directory", "Core_Infra")
dns = build_chain("dns", 2, "CentOS 7", 1, 2, (30, 50), "Core_DNS", "Core_Infra")

# 7. Mail / Collaboration (4 servers)
build_chain("mail", 4, "Windows Server 2016", 4, 16, (150, 400), "Exchange", "Corp_Comms")

# 8. SharePoint (3 servers)
build_chain("sp", 3, "Windows Server 2016", 4, 16, (150, 300), "SharePoint", "Corp_Comms")

# 9. Jira / DevTools — early-migration pilot candidate (3 servers)
build_chain("jira", 3, "Ubuntu 20.04", 4, 16, (100, 250), "Jira", "Internal_Tools", early=True)

# 10. Legacy test / POC junk — isolated singletons, legacy OS, high risk (6 servers)
build_isolated("legacy", 6, "RHEL 5", 1, 2, (20, 60), "POC_Archived", "")

# 11. HIPAA Med Records (5 servers)
build_chain("med", 5, "Windows Server 2008 R2", 4, 16, (150, 400), "Med_Records", "HIPAA")

# 12. Marketing / Blog — early-migration pilot candidate (3 servers)
build_chain("blog2", 3, "Ubuntu 22.04", 1, 2, (30, 60), "Corp_Blog", "Marketing", early=True)

# 13. Misc isolated utility servers to round out ~100 total
build_isolated("util", 6, "Windows Server 2019", 2, 8, (500, 2000), "File_Server", "Internal_Tools")


def main() -> None:
    df = pd.DataFrame(rows)
    df["Outbound_Connections"] = df["Server_ID"].map(
        lambda sid: "|".join(edges_by_id.get(sid, []))
    )
    df = df[
        [
            "Server_ID", "Hostname", "OS_Version", "Allocated_vCPU", "Allocated_RAM_GB",
            "Peak_CPU_Pct", "Avg_CPU_Pct", "Storage_GB", "App_Name", "Business_Tags",
            "Outbound_Connections", "Transfer_GB", "Current_Monthly_Cost",
            "Early_Migration_Requested",
        ]
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    dp_total = df[df["App_Name"] == "Data_Platform"]["Transfer_GB"].sum()
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"DataPlatform cluster total Transfer_GB: {dp_total:.0f} GB (limit is 3,000 GB per wave)")
    print(f"Early-migration flagged: {df[df['Early_Migration_Requested']]['Server_ID'].tolist()}")


if __name__ == "__main__":
    main()
