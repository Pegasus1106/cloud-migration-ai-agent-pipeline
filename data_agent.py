"""
Module 1 - Discovery / Data Normalization Agent (P1)
======================================================
Extends the stakeholder's baseline data-normalization agent (from the
separate baseline repository this project was originally built on top
of) to also validate and pass through the three fields the P1
refinement needs. This module is self-contained and has no runtime
dependency on that other repository.
Transfer_GB, Current_Monthly_Cost, and Early_Migration_Requested. Produces
a cleaned CSV plus an explicit data-quality report, per the stakeholder's
"show what exists in the current ecosystem" request.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

DEFAULTS = {
    "OS_Version": "Unknown Legacy OS",
    "Allocated_vCPU": 2,
    "Allocated_RAM_GB": 4,
    "Storage_GB": 50,
    "Peak_CPU_Pct": 20,
    "Avg_CPU_Pct": 5,
    "Transfer_GB": 50,
    "Current_Monthly_Cost": 0.0,
    "Early_Migration_Requested": False,
}

REQUIRED_NEW_FIELDS = ["Transfer_GB", "Current_Monthly_Cost", "Early_Migration_Requested"]

DEFAULT_INPUT = Path("data/sample.csv")
DEFAULT_OUTPUT_CSV = Path("data/cleaned_samples.csv")
DEFAULT_QUALITY_REPORT = Path("data/data_quality_report.json")


@tool
def clean_discovery_data(input_path: str = str(DEFAULT_INPUT),
                          output_path: str = str(DEFAULT_OUTPUT_CSV)) -> str:
    """Read raw server discovery CSV, flag & impute missing values,
    compute Normalized_CPU_Pct, validate the P1 input contract
    (Transfer_GB, Current_Monthly_Cost, Early_Migration_Requested), write a
    data-quality report, and save the cleaned result.

    Args:
        input_path:  Path to the raw discovery CSV.
        output_path: Path where the cleaned CSV will be saved.

    Returns:
        A human-readable summary of what was processed.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        return f"ERROR: Input file '{input_path}' not found."

    df = pd.read_csv(input_path)
    total_rows = len(df)

    missing_contract_fields = [c for c in REQUIRED_NEW_FIELDS if c not in df.columns]
    for col in missing_contract_fields:
        df[col] = DEFAULTS[col]

    critical_cols = ["Allocated_vCPU", "Allocated_RAM_GB", "Storage_GB", "OS_Version"]
    df["requires_manual_audit"] = df[critical_cols].isna().any(axis=1)
    flagged_count = int(df["requires_manual_audit"].sum())

    imputation_report: dict[str, int] = {}
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            continue
        missing_before = int(df[col].isna().sum())
        if missing_before > 0:
            df[col] = df[col].fillna(default)
            imputation_report[col] = missing_before

    df["Normalized_CPU_Pct"] = (
        df["Peak_CPU_Pct"].astype(float) * 0.70
        + df["Avg_CPU_Pct"].astype(float) * 0.30
    ).round(2)

    df["Early_Migration_Requested"] = df["Early_Migration_Requested"].astype(bool)
    df["Transfer_GB"] = df["Transfer_GB"].astype(float)
    df["Current_Monthly_Cost"] = df["Current_Monthly_Cost"].astype(float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    quality_report = {
        "input_file": str(input_path),
        "total_rows": total_rows,
        "rows_flagged_for_manual_audit": flagged_count,
        "fields_missing_from_source_and_defaulted": missing_contract_fields,
        "imputed_field_counts": imputation_report,
        "early_migration_requested_count": int(df["Early_Migration_Requested"].sum()),
        "total_transfer_gb": float(df["Transfer_GB"].sum()),
        "total_current_monthly_cost": float(df["Current_Monthly_Cost"].sum()),
    }
    quality_path = output_path.parent / "data_quality_report.json"
    quality_path.write_text(json.dumps(quality_report, indent=2))

    imp_lines = "\n".join(
        f"  - {col}: {cnt} missing -> filled with '{DEFAULTS[col]}'"
        for col, cnt in imputation_report.items()
    )
    summary = (
        f"Data Normalization Complete\n"
        f"   Rows processed : {total_rows}\n"
        f"   Rows flagged   : {flagged_count}\n"
        f"   Contract fields missing from source: {missing_contract_fields or 'none'}\n"
        f"   Imputed fields :\n{imp_lines}\n"
        f"   Output saved to: {output_path}\n"
        f"   Quality report : {quality_path}"
    )
    return summary


def build_agent():
    llm = ChatOpenAI(
        base_url="https://openai.rc.asu.edu/v1",
        api_key=os.environ.get("ASU_RC_API_KEY", "not-set"),
        model="qwen3-coder-30b-a3b-instruct",
        temperature=0.0,
    )
    tools = [clean_discovery_data]
    system_message = SystemMessage(
        content=(
            "You are a Cloud Migration assistant responsible for data "
            "normalization. Use the available tools to clean and prepare "
            "server discovery data for migration planning."
        )
    )
    return create_agent(llm, tools, system_prompt=system_message)


if __name__ == "__main__":
    input_arg = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_INPUT)
    output_arg = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_OUTPUT_CSV)
    try:
        agent = build_agent()
        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        f"Clean the server discovery data in '{input_arg}'. "
                        "Flag any rows with missing critical compute metrics, "
                        "impute safe defaults, compute the Normalized_CPU_Pct, "
                        "validate Transfer_GB / Current_Monthly_Cost / "
                        "Early_Migration_Requested, write a data-quality report, "
                        f"and save the result to '{output_arg}'."
                    )
                )
            ]
        })
        print("\n--- Agent Response ---")
        print(result["messages"][-1].content)
    except Exception as exc:
        print(f"Agent invocation failed ({exc}). Running tool directly...")
        direct_result = clean_discovery_data.invoke({
            "input_path": input_arg,
            "output_path": output_arg,
        })
        print(direct_result)
