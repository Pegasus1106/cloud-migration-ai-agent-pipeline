"""Typed configuration and shared run state for the supervisor graph."""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

from p1 import config


class RunConfig(BaseModel):
    """User-confirmed planning constraints (output of the Constraint Agent)."""

    max_transfer_gb: float = Field(default=config.DEFAULT_MAX_TRANSFER_GB, gt=0, le=1_000_000,
                                   description="Maximum total Transfer_GB in one migration wave.")
    target_cloud: Literal["AWS", "AZURE"] = Field(default="AWS")
    region: str = Field(default="us-east-1")
    honor_early_migration_flags: bool = Field(
        default=True, description="Prioritize servers flagged Early_Migration_Requested in the CSV.")
    early_server_ids: list[str] = Field(
        default_factory=list, description="Extra servers the user asked to move early (resolved IDs).")
    early_app_names: list[str] = Field(default_factory=list,
                                       description="Application names the user asked to move early.")
    pilot_wave: bool = Field(default=False,
                             description="Reserve wave 0 for a small low-risk, isolated pilot.")
    pilot_max_servers: int = Field(default=5, ge=1, le=50)
    pilot_max_gb: float = Field(default=1500.0, gt=0)
    notes: str = ""
    source_request: str = ""
    confirmed: bool = False
    confirmed_by: str = ""

    @field_validator("target_cloud", mode="before")
    @classmethod
    def _upper(cls, v):
        return str(v).strip().upper() if v is not None else v

    @field_validator("early_server_ids", "early_app_names", mode="before")
    @classmethod
    def _dedupe(cls, v):
        if v is None:
            return []
        seen, out = set(), []
        for item in v:
            s = str(item).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out


class P1State(TypedDict, total=False):
    """Shared state passed between supervisor nodes (all JSON-serializable)."""

    run_id: str
    dataset_path: str
    config: dict
    servers: list[dict]          # cleaned discovery rows (+ graph/size/cost/risk columns)
    edges: list[dict]            # {src, dst, port, bandwidth}
    quality: dict
    graph_stats: dict
    cloud: dict
    finops: dict
    risk_summary: dict
    plan: dict                   # wave assignment + infeasible groups + pilot/early info
    validation: dict
    repair_round: int
    repair_log: list[dict]
    review_items: list[dict]
    review_decisions: list[dict]
    approved_exceptions: list[str]
    acknowledged: list[str]
    excluded_ids: list[str]
    explanations: dict
    status: str
    stop_reason: str
    timings: dict
    outputs: dict
    progress: str                # one-line summary of the last node (for the UI/CLI)
    replan: bool                 # set by human review when a server was excluded
