"""Discovery-data contract, normalization, and data-quality report.

Extends the stakeholder baseline's data agent: same imputation defaults and
Normalized_CPU formula (0.7 * peak + 0.3 * average), plus validation of the P1
input contract (Transfer_GB, Current_Monthly_Cost, Early_Migration_Requested),
duplicate IDs, malformed connection strings, and unknown dependency endpoints.
"""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "Server_ID", "Hostname", "OS_Version", "Allocated_vCPU", "Allocated_RAM_GB",
    "Peak_CPU_Pct", "Avg_CPU_Pct", "Storage_GB", "App_Name", "Business_Tags",
    "Outbound_Connections",
]
CONTRACT_FIELDS = ["Transfer_GB", "Current_Monthly_Cost", "Early_Migration_Requested"]
CRITICAL_FIELDS = ["Allocated_vCPU", "Allocated_RAM_GB", "Storage_GB", "OS_Version"]
DEFAULTS = {
    "OS_Version": "Unknown Legacy OS",
    "Allocated_vCPU": 2,
    "Allocated_RAM_GB": 4,
    "Storage_GB": 50,
    "Peak_CPU_Pct": 20,
    "Avg_CPU_Pct": 5,
    "Current_Monthly_Cost": 0.0,
    "Early_Migration_Requested": False,
}
_TRUE = {"true", "1", "yes", "y", "t"}


class DataContractError(ValueError):
    pass


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip().lower() in _TRUE


def parse_connections(raw) -> tuple[list[tuple[str, str, str]], int]:
    """Parse 'SRV-002:8080:150|SRV-003:1433:20' -> [(dst, port, bandwidth)], malformed count."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return [], 0
    text = str(raw).strip()
    if text in ("", "nan", "None"):
        return [], 0
    out, bad = [], 0
    for part in text.split("|"):
        bits = [b.strip() for b in part.strip().split(":")]
        if len(bits) == 3 and bits[0]:
            out.append((bits[0], bits[1], bits[2]))
        elif part.strip():
            bad += 1
    return out, bad


def normalize(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    """Return (cleaned dataframe, dependency edges, data-quality report)."""
    missing_required = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_required:
        raise DataContractError(f"Discovery CSV is missing required columns: {missing_required}")

    df = raw.copy()
    total_rows = len(df)
    df["Server_ID"] = df["Server_ID"].astype(str).str.strip()
    dupes = df[df["Server_ID"].duplicated()]["Server_ID"].tolist()
    df = df.drop_duplicates("Server_ID", keep="first").reset_index(drop=True)

    contract_missing = [c for c in CONTRACT_FIELDS if c not in df.columns]
    df["requires_manual_audit"] = df[CRITICAL_FIELDS].isna().any(axis=1)
    audit_fields = {
        row["Server_ID"]: [c for c in CRITICAL_FIELDS if pd.isna(row[c])]
        for _, row in df[df["requires_manual_audit"]].iterrows()
    }

    imputed: dict[str, int] = {}
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            continue
        n = int(df[col].isna().sum())
        if n:
            df[col] = df[col].fillna(default)
            imputed[col] = n

    # Transfer volume defaults to storage (documented assumption), not a constant.
    if "Transfer_GB" not in df.columns:
        df["Transfer_GB"] = df["Storage_GB"]
    else:
        n = int(df["Transfer_GB"].isna().sum())
        if n:
            df["Transfer_GB"] = df["Transfer_GB"].fillna(df["Storage_GB"])
            imputed["Transfer_GB"] = n

    for col in ["Allocated_vCPU", "Allocated_RAM_GB", "Storage_GB", "Peak_CPU_Pct",
                "Avg_CPU_Pct", "Transfer_GB", "Current_Monthly_Cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(DEFAULTS.get(col, 0)).astype(float)
    negative = df.loc[(df[["Allocated_vCPU", "Allocated_RAM_GB", "Storage_GB", "Transfer_GB"]] < 0)
                      .any(axis=1), "Server_ID"].tolist()

    df["Early_Migration_Requested"] = df["Early_Migration_Requested"].map(_to_bool)
    df["App_Name"] = df["App_Name"].fillna("Unknown_App").astype(str)
    df["Hostname"] = df["Hostname"].fillna(df["Server_ID"]).astype(str)
    df["OS_Version"] = df["OS_Version"].astype(str)
    df["Business_Tags"] = df["Business_Tags"].fillna("").astype(str).replace({"nan": "", "None": ""})
    df["Outbound_Connections"] = df["Outbound_Connections"].fillna("").astype(str)
    df["Normalized_CPU_Pct"] = (df["Peak_CPU_Pct"] * 0.70 + df["Avg_CPU_Pct"] * 0.30).round(2)

    known = set(df["Server_ID"])
    edges, unknown_endpoints, malformed, self_loops, duplicates = [], set(), 0, 0, 0
    seen_edges = set()
    for _, row in df.iterrows():
        conns, bad = parse_connections(row["Outbound_Connections"])
        malformed += bad
        for dst, port, bw in conns:
            if dst not in known:
                unknown_endpoints.add(dst)
                continue
            if dst == row["Server_ID"]:
                self_loops += 1
                continue
            key = (row["Server_ID"], dst, port)
            if key in seen_edges:
                duplicates += 1
                continue
            seen_edges.add(key)
            try:
                bandwidth = float(bw)
            except ValueError:
                bandwidth = None
            edges.append({"src": row["Server_ID"], "dst": dst, "port": port, "bandwidth": bandwidth})

    report = {
        "total_rows": total_rows,
        "servers_after_dedup": len(df),
        "duplicate_server_ids": dupes,
        "rows_flagged_for_manual_audit": int(df["requires_manual_audit"].sum()),
        "audit_details": audit_fields,
        "contract_fields_missing_from_source": contract_missing,
        "imputed_field_counts": imputed,
        "negative_value_rows": negative,
        "malformed_connection_entries": malformed,
        "self_loops_dropped": self_loops,
        "duplicate_connections_dropped": duplicates,
        "unknown_dependency_endpoints": sorted(unknown_endpoints),
        "edge_count": len(edges),
        "early_migration_requested_count": int(df["Early_Migration_Requested"].sum()),
        "total_transfer_gb": round(float(df["Transfer_GB"].sum()), 1),
        "total_current_monthly_cost": round(float(df["Current_Monthly_Cost"].sum()), 2),
        "app_count": int(df["App_Name"].nunique()),
    }
    return df, edges, report


def load_and_normalize(path: str | Path) -> tuple[pd.DataFrame, list[dict], dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Discovery CSV not found: {path}")
    df, edges, report = normalize(pd.read_csv(path))
    report["input_file"] = str(path)
    return df, edges, report
