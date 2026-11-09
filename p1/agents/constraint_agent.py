"""Constraint Agent: multi-turn chat -> user-confirmed RunConfig.

Agentic behaviour: the model decides which tools to call to ground the user's
words in the dataset (e.g. "move the CRM and Jira apps early" -> concrete
server IDs), asks a clarifying question when a name is ambiguous or unknown,
and proposes a configuration through a validating tool. It never confirms on
the user's behalf: confirmation is a separate, explicit user action, and there
is no silent fallback to defaults (Milestone 1: "conversation history is not
trusted as configuration until the user confirms the extracted values").
"""

import re
from dataclasses import dataclass, field

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from p1 import config
from p1.agents.common import dumps, last_ai_text, run_config
from p1.llm import get_llm
from p1.state import RunConfig
from p1.tracing import Tracer

PROMPT_VERSION = "constraint-v2.0"
SYSTEM_PROMPT = f"""You are the Constraint Agent of a cloud-migration planning system.
You turn the user's planning request into a structured configuration. You never plan waves
or calculate anything yourself; deterministic tools do that after the user confirms.

How to work:
1. Call `dataset_overview` once to learn which applications, business tags, and totals exist.
2. For every application, hostname, tag, or server the user wants moved early, call
   `find_servers` and use only names/IDs that a tool returned.
3. Units: 1 TB = {config.TB_IN_GB:.0f} GB. If the user states no wave limit, use
   {config.DEFAULT_MAX_TRANSFER_GB:.0f} GB (the stakeholder default) and say so.
4. target_cloud is AWS unless the user explicitly asks for Azure.
5. Words like "pilot", "start small", or "test wave first" mean pilot_wave = true.
6. If a name the user gives matches nothing, or could mean several unrelated applications,
   do NOT guess: ask one short clarifying question and do not call `propose_config`.
7. Once everything is resolved, call `propose_config` exactly once with the final values.
   If it returns an ERROR, fix the values and call it again.
8. Finish with a 2-4 line plain-English summary of the proposal and ask the user to confirm.
"""


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


@dataclass
class _Session:
    df: pd.DataFrame
    proposed: RunConfig | None = None
    proposals: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def apps(self) -> dict[str, list[str]]:
        return self.df.groupby("App_Name")["Server_ID"].apply(lambda s: sorted(s)).to_dict()


def _build_tools(session: _Session):
    df = session.df

    @tool
    def dataset_overview() -> str:
        """Summarize the loaded discovery data: applications (with server counts),
        business tags, total transfer volume, and servers already flagged for early migration."""
        apps = df.groupby("App_Name").agg(servers=("Server_ID", "count"), transfer_gb=("Transfer_GB", "sum"))
        tags = sorted({t for t in df["Business_Tags"] if t})
        return dumps({
            "servers": len(df),
            "total_transfer_gb": round(float(df["Transfer_GB"].sum()), 1),
            "applications": {a: {"servers": int(r.servers), "transfer_gb": round(float(r.transfer_gb), 1)}
                             for a, r in apps.iterrows()},
            "business_tags": tags,
            "already_flagged_early_in_csv": sorted(df.loc[df["Early_Migration_Requested"], "Server_ID"]),
        })

    @tool
    def find_servers(query: str) -> str:
        """Find servers by application name, hostname, business tag, or server ID.
        Matching ignores case, spaces, and punctuation. Returns matches grouped by application."""
        q = _norm(query)
        if not q:
            return "ERROR: empty query"
        hits = df[df["Server_ID"].map(_norm).eq(q) | df["App_Name"].map(_norm).str.contains(q, regex=False)
                  | df["Hostname"].map(_norm).str.contains(q, regex=False)
                  | df["Business_Tags"].map(_norm).str.contains(q, regex=False)]
        if hits.empty:
            return dumps({"query": query, "matches": 0,
                          "hint": "No server matches. Known applications: " + ", ".join(sorted(df["App_Name"].unique()))})
        groups = {}
        for app, g in hits.groupby("App_Name"):
            groups[app] = {"server_ids": sorted(g["Server_ID"]), "hostnames": sorted(g["Hostname"])[:10],
                           "business_tags": sorted({t for t in g["Business_Tags"] if t}),
                           "transfer_gb": round(float(g["Transfer_GB"].sum()), 1)}
        return dumps({"query": query, "matches": len(hits), "applications": groups})

    @tool
    def propose_config(max_transfer_gb: float, target_cloud: str = "AWS", early_app_names: list[str] | None = None,
                       early_server_ids: list[str] | None = None, pilot_wave: bool = False,
                       honor_early_migration_flags: bool = True, notes: str = "") -> str:
        """Validate and store the proposed planning configuration (shown to the user for confirmation).

        Args:
            max_transfer_gb: Maximum total transfer volume per wave, in GB (1 TB = 1000 GB).
            target_cloud: 'AWS' or 'AZURE'.
            early_app_names: Application names (exactly as returned by tools) to migrate early.
            early_server_ids: Individual server IDs to migrate early.
            pilot_wave: Reserve wave 0 for a small low-risk pilot.
            honor_early_migration_flags: Also honor Early_Migration_Requested flags already in the CSV.
            notes: Any other constraint the user mentioned that has no field.
        """
        errors, warnings = [], []
        apps_norm = {_norm(a): a for a in df["App_Name"].unique()}
        resolved_apps = []
        for a in early_app_names or []:
            match = apps_norm.get(_norm(a))
            if match is None:
                errors.append(f"unknown application '{a}' (use find_servers)")
            else:
                resolved_apps.append(match)
        known_ids = set(df["Server_ID"])
        bad_ids = [s for s in early_server_ids or [] if s not in known_ids]
        if bad_ids:
            errors.append(f"unknown server IDs {bad_ids}")
        if not (1 <= float(max_transfer_gb) <= 1_000_000):
            errors.append("max_transfer_gb must be between 1 and 1,000,000 GB")
        cloud = str(target_cloud).strip().upper()
        if cloud not in ("AWS", "AZURE"):
            errors.append("target_cloud must be AWS or AZURE")
        if errors:
            return "ERROR: " + "; ".join(errors)
        if cloud == "AZURE":
            warnings.append("The FinOps price catalog is AWS-only, so Azure costs will not be estimated.")
        cfg = RunConfig(max_transfer_gb=float(max_transfer_gb), target_cloud=cloud,
                        early_app_names=resolved_apps, early_server_ids=list(early_server_ids or []),
                        pilot_wave=bool(pilot_wave), honor_early_migration_flags=bool(honor_early_migration_flags),
                        notes=notes)
        early = set(cfg.early_server_ids) | set(df[df["App_Name"].isin(resolved_apps)]["Server_ID"])
        if honor_early_migration_flags:
            early |= set(df.loc[df["Early_Migration_Requested"], "Server_ID"])
        early_gb = float(df[df["Server_ID"].isin(early)]["Transfer_GB"].sum())
        if early_gb > cfg.max_transfer_gb:
            warnings.append(f"Early servers total {early_gb:,.0f} GB, more than one wave; they will span several waves.")
        session.proposed = cfg
        session.proposals += 1
        session.warnings = warnings
        return dumps({"status": "proposed", "config": cfg.model_dump(exclude={"confirmed", "confirmed_by", "source_request"}),
                      "early_servers_total": len(early), "early_transfer_gb": round(early_gb, 1), "warnings": warnings})

    return [dataset_overview, find_servers, propose_config]


class ConstraintAgent:
    """Stateful chat agent. Call `send()` for each user message."""

    def __init__(self, cleaned: pd.DataFrame, tracer: Tracer | None = None, model: str | None = None):
        self.session = _Session(cleaned)
        self.tracer = tracer
        self.messages: list = []
        self.agent = create_agent(get_llm(model), _build_tools(self.session), system_prompt=SYSTEM_PROMPT)
        self.transcript: list[dict] = []

    def send(self, text: str) -> dict:
        before = self.session.proposals
        self.messages.append(HumanMessage(content=text))
        self.transcript.append({"role": "user", "content": text})
        result = self.agent.invoke({"messages": self.messages},
                                   config=run_config(self.tracer, "constraint_agent", PROMPT_VERSION))
        self.messages = result["messages"]
        reply = last_ai_text(self.messages) or "(no reply)"
        self.transcript.append({"role": "assistant", "content": reply})
        proposed = self.session.proposed if self.session.proposals > before else None
        if proposed is not None:
            proposed.source_request = "\n".join(t["content"] for t in self.transcript if t["role"] == "user")
        return {"reply": reply, "status": "proposed" if proposed else "clarification",
                "proposed": proposed, "warnings": list(self.session.warnings) if proposed else []}


def extract_once(request: str, cleaned: pd.DataFrame, tracer: Tracer | None = None,
                 model: str | None = None) -> dict:
    """Single-turn extraction used by the CLI and the constraint-parsing benchmark."""
    agent = ConstraintAgent(cleaned, tracer, model)
    return agent.send(request)
