"""D3 synthetic benchmark + labeled agent-evaluation sets (Milestone 1, Section 6).

Two-model design so the system under test never writes its own test data:
  * gpt-oss-120b (P1_DATA_MODEL) writes the *realistic* parts: an enterprise
    application portfolio, natural-language planning requests, and paraphrased
    questions about the plan.
  * Seeded Python turns the portfolio into servers and dependencies and owns
    every label, so ground truth is exact (inbound degree, cycles, oversized
    indivisible groups, early requests, servers larger than the sizing matrix).

LLM outputs are cached in this folder (portfolio_spec.json, constraint_prompts.jsonl,
qa_questions.jsonl), so `python data/benchmark/generate_benchmark.py` rebuilds the
same datasets offline. Use --refresh to call the model again.

Outputs:
  D3_30.csv, D3_100.csv, D3_500.csv           discovery CSVs (same schema as the stakeholder sample)
  D3_<n>_truth.json                           ground truth for each CSV
  constraint_prompts.jsonl                    labeled requests for the Constraint Agent
  qa_questions.jsonl                          questions for the Q&A agent (gold computed from the plan at eval time)
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Literal

import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from p1 import config  # noqa: E402
from p1.agents.common import structured_call  # noqa: E402

SEED = 598
SIZES = (30, 100, 500)
BU_FOCUS = [
    "Finance, payments and billing (PCI-DSS and SOX heavy)",
    "Healthcare and patient services (HIPAA heavy)",
    "Retail, e-commerce and customer-facing web",
    "Corporate IT, HR, collaboration and internal tools",
    "Data, analytics and reporting platforms",
]


# ----------------------------------------------------------------------------- portfolio (LLM)
class AppSpec(BaseModel):
    name: str = Field(description="Application name in Title_Snake_Case, e.g. Claims_Processing")
    business_unit: str
    pattern: Literal["three_tier", "two_tier", "single_server", "batch_pipeline", "data_platform"]
    os_family: Literal["windows", "linux", "mixed"]
    compliance: Literal["PCI-DSS", "HIPAA", "SOX", "None"]
    external_facing: bool
    criticality: Literal["low", "medium", "high"]
    data_size: Literal["small", "medium", "large", "very_large"]
    uses_shared_database: bool
    uses_directory_services: bool
    early_migration_requested: bool
    legacy: bool = Field(description="Runs on an old, out-of-support platform version")


class Portfolio(BaseModel):
    applications: list[AppSpec]


PORTFOLIO_SYSTEM = """You are an enterprise architect describing a realistic on-premises application
portfolio for a cloud-migration assessment benchmark. Produce varied, believable applications with
realistic names. Mix architecture patterns, operating systems, compliance scopes, and data sizes the way
a real mid-size company would. Only about 1 in 10 applications should request early migration (typically
low-criticality, customer-requested, or pilot candidates). About 1 in 5 should be legacy."""


def generate_portfolio(refresh: bool) -> list[dict]:
    path = HERE / "portfolio_spec.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())["applications"]
    apps: dict[str, dict] = {}
    for focus in BU_FOCUS:
        out = structured_call(Portfolio, PORTFOLIO_SYSTEM,
                              f"Business-unit focus: {focus}.\nGenerate 26 applications for this business unit.",
                              agent="benchmark_generator", prompt_version="portfolio-v1", tracer=None,
                              model=config.DATA_GEN_MODEL)
        for a in out.applications:
            key = re.sub(r"[^A-Za-z0-9]+", "_", a.name).strip("_")
            if key and key.lower() not in {k.lower() for k in apps}:
                apps[key] = {**a.model_dump(), "name": key}
    data = {"generator_model": config.DATA_GEN_MODEL, "seed": SEED, "applications": list(apps.values())}
    path.write_text(json.dumps(data, indent=2))
    return data["applications"]


# ----------------------------------------------------------------------------- servers (deterministic)
WIN_OS = {"legacy": ["Windows Server 2008 R2", "Windows Server 2012"], "current": ["Windows Server 2016",
          "Windows Server 2019", "Windows Server 2022"]}
WIN_DB = {"legacy": ["SQL Server 2008"], "current": ["SQL Server 2016", "SQL Server 2019", "SQL Server 2022"]}
LIN_OS = {"legacy": ["RHEL 5", "RHEL 6", "CentOS 7", "Ubuntu 18.04"],
          "current": ["RHEL 8", "RHEL 9", "Ubuntu 20.04", "Ubuntu 22.04", "Ubuntu 24.04"]}
LIN_DB = {"legacy": ["Oracle Linux 6.4", "RHEL 7"], "current": ["Oracle Linux 8", "RHEL 8", "Ubuntu 22.04"]}
STORAGE = {"small": (40, 200), "medium": (150, 700), "large": (600, 2200), "very_large": (2500, 9000)}
TIER_SPEC = {"web": [(2, 4), (2, 8), (4, 8), (4, 16)], "app": [(4, 16), (8, 16), (8, 32)],
             "db": [(8, 32), (8, 64), (16, 64), (16, 128)], "data": [(16, 64), (16, 128), (32, 256), (32, 128)],
             "batch": [(4, 16), (8, 32)], "infra": [(2, 8), (4, 16)]}


class Builder:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.rows: list[dict] = []
        self.edges: dict[str, list[str]] = {}
        self.n = 0
        self.shared_db: dict[str, list[str]] = {}

    def server(self, host: str, app: dict | str, tier: str, early: bool = False) -> str:
        self.n += 1
        sid = f"BEN-{self.n:04d}"
        spec = app if isinstance(app, dict) else {"name": app, "os_family": "windows", "compliance": "None",
                                                  "external_facing": False, "criticality": "high",
                                                  "data_size": "small", "legacy": False}
        rng = self.rng
        fam = spec["os_family"] if spec["os_family"] != "mixed" else rng.choice(["windows", "linux"])
        era = "legacy" if spec.get("legacy") and rng.random() < 0.8 else "current"
        pool = (WIN_DB if fam == "windows" else LIN_DB) if tier == "db" else (WIN_OS if fam == "windows" else LIN_OS)
        cpu, ram = rng.choice(TIER_SPEC[tier])
        lo, hi = STORAGE[spec["data_size"]] if tier in ("db", "data") else (30, 250)
        storage = rng.randint(lo, hi)
        peak = rng.randint(25, 95) if spec["criticality"] != "low" else rng.randint(5, 50)
        tags = spec["compliance"] if spec["compliance"] != "None" else ("External_Facing" if spec["external_facing"]
                                                                        else "Internal")
        self.rows.append({"Server_ID": sid, "Hostname": host, "OS_Version": rng.choice(pool[era]),
                          "Allocated_vCPU": cpu, "Allocated_RAM_GB": ram, "Peak_CPU_Pct": peak,
                          "Avg_CPU_Pct": max(1, peak - rng.randint(10, 40)), "Storage_GB": storage,
                          "App_Name": spec["name"], "Business_Tags": tags, "Transfer_GB": storage,
                          "Current_Monthly_Cost": round(40 * cpu + 10 * ram + 0.05 * storage + 50, 2),
                          "Early_Migration_Requested": early})
        return sid

    def edge(self, a: str, b: str) -> None:
        if a != b and b not in self.edges.setdefault(a, []):
            self.edges[a].append(b)

    def infra(self) -> dict[str, str]:
        dc1 = self.server("ad-dc-01", "Active_Directory", "infra")
        dc2 = self.server("ad-dc-02", "Active_Directory", "infra")
        self.edge(dc1, dc2)
        self.edge(dc2, dc1)  # replication: a real 2-node cycle
        dns1 = self.server("dns-01", {"name": "Core_DNS", "os_family": "linux", "compliance": "None",
                                      "external_facing": False, "criticality": "high", "data_size": "small",
                                      "legacy": True}, "infra")
        dns2 = self.server("dns-02", {"name": "Core_DNS", "os_family": "linux", "compliance": "None",
                                      "external_facing": False, "criticality": "high", "data_size": "small",
                                      "legacy": False}, "infra")
        self.edge(dns1, dns2)
        return {"ad": dc1, "dns": dns1}

    def bu_database(self, bu: str) -> str:
        if bu not in self.shared_db:
            slug = re.sub(r"[^a-z0-9]+", "", bu.lower())[:8] or "shared"
            spec = {"name": f"Shared_DB_{slug.title()}", "os_family": "windows", "compliance": "SOX",
                    "external_facing": False, "criticality": "high", "data_size": "large", "legacy": False}
            prim = self.server(f"{slug}-sqlclu-01", spec, "db")
            repl = self.server(f"{slug}-sqlclu-02", spec, "db")
            self.edge(prim, repl)
            self.edge(repl, prim)  # synchronous replication pair = indivisible
            self.shared_db[bu] = [prim, repl]
        return self.shared_db[bu][0]

    def app(self, spec: dict, infra: dict[str, str]) -> None:
        rng, name = self.rng, spec["name"]
        parts = name.split("_")
        slug = "-".join(w[:4].lower() for w in parts[:2])
        if parts[-1] in DR_SUFFIXES:
            slug += f"-{parts[-1].lower()}"
        early = spec["early_migration_requested"]
        p = spec["pattern"]
        front: list[str] = []
        if p == "three_tier":
            webs = [self.server(f"{slug}-web-{i + 1:02d}", spec, "web", early) for i in range(rng.randint(1, 3))]
            apps = [self.server(f"{slug}-app-{i + 1:02d}", spec, "app", early) for i in range(rng.randint(1, 2))]
            db = self.bu_database(spec["business_unit"]) if spec["uses_shared_database"] else \
                self.server(f"{slug}-db-01", spec, "db", early)
            for i, w in enumerate(webs):
                self.edge(w, apps[i % len(apps)])
            for a in apps:
                self.edge(a, db)
            front = webs + apps
        elif p == "two_tier":
            apps = [self.server(f"{slug}-app-{i + 1:02d}", spec, "app", early) for i in range(rng.randint(1, 2))]
            db = self.bu_database(spec["business_unit"]) if spec["uses_shared_database"] else \
                self.server(f"{slug}-db-01", spec, "db", early)
            for a in apps:
                self.edge(a, db)
            front = apps
        elif p == "single_server":
            s = self.server(f"{slug}-srv-01", spec, "app", early)
            if spec["uses_shared_database"]:
                self.edge(s, self.bu_database(spec["business_unit"]))
            front = [s]
        elif p == "batch_pipeline":
            stages = [self.server(f"{slug}-{st}-01", spec, "batch", early) for st in ("ingest", "xform", "load")]
            queue = self.server(f"{slug}-queue-01", spec, "batch", early)
            self.edge(stages[0], stages[1])
            self.edge(stages[1], stages[2])
            self.edge(stages[1], queue)
            self.edge(queue, stages[1])  # worker <-> queue cycle
            if spec["uses_shared_database"]:
                self.edge(stages[2], self.bu_database(spec["business_unit"]))
            front = stages[:1]
        else:  # data_platform: dense forward mesh of large nodes
            nodes = [self.server(f"{slug}-node-{i + 1:02d}", spec, "data", early) for i in range(rng.randint(5, 9))]
            for i, a in enumerate(nodes[:-1]):
                self.edge(a, nodes[i + 1])
                for _ in range(2):
                    j = i + rng.randint(1, min(4, len(nodes) - 1 - i))
                    self.edge(a, nodes[j])
            front = nodes[:2]
        if spec["uses_directory_services"]:
            for s in front[:2]:
                self.edge(s, infra["ad"])
                self.edge(s, infra["dns"])


EARLY_FLOOR = 0.10  # Milestone 1 asks for early-migration preferences in the benchmark


def apply_label_floors(apps: list[dict]) -> list[dict]:
    """The model marked fewer apps for early migration than requested; top up
    deterministically (low/medium criticality first) and record the source."""
    apps = [dict(a, early_source="llm" if a["early_migration_requested"] else "") for a in apps]
    need = int(len(apps) * EARLY_FLOOR) - sum(a["early_migration_requested"] for a in apps)
    if need > 0:
        rng = random.Random(SEED)
        pool = [a for a in apps if not a["early_migration_requested"] and a["criticality"] != "high"]
        for a in rng.sample(pool, min(need, len(pool))):
            a["early_migration_requested"], a["early_source"] = True, "generator_floor"
    return apps


DR_SUFFIXES = ["DR", "EU", "APAC", "LATAM"]  # regional / disaster-recovery instances for large sizes


def build_dataset(apps: list[dict], size: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = random.Random(seed + size)
    b = Builder(rng)
    infra = b.infra()
    order = apps[:]
    rng.shuffle(order)
    for spec in order:
        if b.n >= size:
            break
        b.app(spec, infra)
    # Large sizes: real estates run several instances of the same application.
    for suffix in DR_SUFFIXES:
        for spec in order:
            if b.n >= size:
                break
            b.app(dict(spec, name=f"{spec['name']}_{suffix}"), infra)
    df = pd.DataFrame(b.rows)
    keep = set(df["Server_ID"].iloc[:size])
    df = df[df["Server_ID"].isin(keep)].reset_index(drop=True)
    df["Outbound_Connections"] = [
        "|".join(f"{d}:{rng.choice([1433, 5432, 3306, 8080, 443, 389, 53])}:{rng.randint(20, 900)}"
                 for d in b.edges.get(s, []) if d in keep) for s in df["Server_ID"]]
    # Realistic data-quality defects (deterministic): ~3% of rows lose one critical field.
    audit = rng.sample(list(df.index), max(1, len(df) * 3 // 100))
    for i, idx in enumerate(audit):
        col = ["Allocated_vCPU", "Allocated_RAM_GB", "Storage_GB", "OS_Version"][i % 4]
        df.loc[idx, col] = None
    df = df[["Server_ID", "Hostname", "OS_Version", "Allocated_vCPU", "Allocated_RAM_GB", "Peak_CPU_Pct",
             "Avg_CPU_Pct", "Storage_GB", "App_Name", "Business_Tags", "Outbound_Connections", "Transfer_GB",
             "Current_Monthly_Cost", "Early_Migration_Requested"]]
    return df, _truth(df, b, keep)


def _truth(df: pd.DataFrame, b: Builder, keep: set[str]) -> dict:
    edges = [(a, d) for a, ds in b.edges.items() if a in keep for d in ds if d in keep]
    g = nx.DiGraph()
    g.add_nodes_from(df["Server_ID"])
    g.add_edges_from(edges)
    transfer = dict(zip(df["Server_ID"], df["Transfer_GB"].fillna(0)))
    cycles = [sorted(c) for c in nx.strongly_connected_components(g) if len(c) > 1]
    big_scc = [c for c in nx.strongly_connected_components(g) if sum(transfer[s] for s in c) > 3000]
    return {
        "servers": len(df), "edges": len(edges), "apps": int(df["App_Name"].nunique()),
        "inbound_degree": {s: g.in_degree(s) for s in g.nodes},
        "cycles": sorted(cycles),
        "indivisible_groups_over_3000gb": sorted(sorted(c) for c in big_scc),
        "early_ids": sorted(df.loc[df["Early_Migration_Requested"], "Server_ID"]),
        "audit_ids": sorted(df.loc[df[["Allocated_vCPU", "Allocated_RAM_GB", "Storage_GB", "OS_Version"]]
                                   .isna().any(axis=1), "Server_ID"]),
        "larger_than_sizing_matrix": sorted(df.loc[(df["Allocated_vCPU"] > 16) | (df["Allocated_RAM_GB"] > 64),
                                                   "Server_ID"]),
        "wcc_count": nx.number_weakly_connected_components(g),
        "total_transfer_gb": round(float(df["Transfer_GB"].fillna(0).sum()), 1),
    }


# ----------------------------------------------------------------------------- constraint prompts (LLM text, code labels)
class Request(BaseModel):
    case_id: str
    request: str


class Requests(BaseModel):
    items: list[Request]


REQUEST_SYSTEM = """You write realistic messages that a cloud-migration architect would type into a planning
assistant to request a migration plan. Each case lists exactly what the message must ask for. Mention every
listed item and nothing else. Write it as a request (not a question asking to confirm something). Vary tone
and length (terse, formal, chatty, bullet-like). Refer to applications the way a person would say them
(e.g. 'the payment gateway', 'HR portal'), not with underscores. Express the wave limit exactly with the
given wording. If the case says the target cloud is not mentioned, do not mention any cloud. Only mention a
pilot wave if the case includes start_with_a_pilot_wave; otherwise never use the word 'pilot'."""


def _spoken(app: str) -> str:
    return app.replace("_", " ").replace("-", " ")


def _gb_text(gb: float, rng: random.Random) -> str:
    if gb % 500 == 0 and rng.random() < 0.5:
        return f"{gb / 1000:g} TB"
    return rng.choice([f"{gb:,.0f} GB", f"{gb:.0f}GB", f"{gb:,.0f} gigabytes"])


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build_prompt_cases(datasets: dict[str, pd.DataFrame], rng: random.Random) -> list[dict]:
    """Clear cases: every label is fixed by code before the model writes the text.
    Ambiguous cases name something that matches no application, or a word shared
    by several different applications, so the correct behaviour is to ask."""
    cases = []
    for ds_name, df in datasets.items():
        apps = sorted(df["App_Name"].dropna().unique())
        norm = {a: _norm(a) for a in apps}
        clear_apps = [a for a in apps if not a.lower().startswith(("unknown", "shared_db"))
                      and not any(norm[a] in norm[b] for b in apps if b != a)]
        for k in range(12):
            gb = rng.choice([1500, 2000, 2500, 3000, 3500, 4000, 5000])
            n_apps = rng.choice([0, 1, 1, 2, 2, 3])
            early = sorted(rng.sample(clear_apps, n_apps))
            cloud = rng.choice(["AWS", "AWS", "not mentioned", "not mentioned", "AZURE"])
            pilot = rng.random() < 0.35
            spec = {"wave_limit": _gb_text(gb, rng), "target_cloud": cloud, "move_early": [_spoken(a) for a in early]}
            if pilot:
                spec["start_with_a_pilot_wave"] = True
            cases.append({"case_id": f"{ds_name}-clear-{k:02d}", "dataset": ds_name, "kind": "clear", "spec": spec,
                          "gold": {"max_transfer_gb": float(gb), "target_cloud": "AWS" if cloud != "AZURE" else "AZURE",
                                   "early_app_names": early, "pilot_wave": pilot}})
    vague = {
        "D1": ["the billing system", "ERP", "the mainframe connector"],
        "D3_100": ["the payment app", "the clinical system", "the mainframe connector",
                   "the warehouse management system"],
    }
    for ds_name, refs in vague.items():
        for k, ref in enumerate(refs):
            cases.append({"case_id": f"{ds_name}-ambiguous-{k:02d}", "dataset": ds_name, "kind": "ambiguous",
                          "spec": {"wave_limit": "3000 GB", "target_cloud": "not mentioned", "move_early": [ref]},
                          "gold": {"needs_clarification": True}})
    return cases


def _mentions_ok(text: str, case: dict) -> bool:
    low = text.lower()
    nums = re.sub(r"[,\s]", "", low)
    gold_gb = case["gold"].get("max_transfer_gb")
    if gold_gb and not (f"{gold_gb:.0f}" in nums or f"{gold_gb / 1000:g}tb" in nums or
                        f"{gold_gb / 1000:g}terabyte" in nums):
        return False
    for app in case["spec"]["move_early"]:
        words = [w for w in re.split(r"\W+", app.lower()) if len(w) > 2]
        if words and not any(w in low for w in words):
            return False
    if case["kind"] == "clear" and case["gold"]["target_cloud"] == "AZURE" and "azure" not in low:
        return False
    if case["spec"]["target_cloud"] == "not mentioned" and ("aws" in low or "azure" in low or "amazon" in low):
        return False
    wants_pilot = case["spec"].get("start_with_a_pilot_wave", False)
    if ("pilot" in low) != wants_pilot:
        return False
    return True


def generate_prompts(datasets: dict[str, pd.DataFrame], refresh: bool) -> list[dict]:
    path = HERE / "constraint_prompts.jsonl"
    if path.exists() and not refresh:
        return [json.loads(line) for line in path.read_text().splitlines()]
    rng = random.Random(SEED)
    cases = build_prompt_cases(datasets, rng)
    done: dict[str, str] = {}
    for attempt in range(3):
        todo = [c for c in cases if c["case_id"] not in done]
        for i in range(0, len(todo), 8):
            batch = todo[i:i + 8]
            out = structured_call(Requests, REQUEST_SYSTEM,
                                  "Cases:\n" + json.dumps([{"case_id": c["case_id"], **c["spec"]} for c in batch]),
                                  agent="benchmark_generator", prompt_version="requests-v1", tracer=None,
                                  model=config.DATA_GEN_MODEL)
            by_id = {c["case_id"]: c for c in batch}
            for r in out.items:
                if r.case_id in by_id and _mentions_ok(r.request, by_id[r.case_id]):
                    done[r.case_id] = r.request
    rows = [{**c, "request": done[c["case_id"]], "generator_model": config.DATA_GEN_MODEL}
            for c in cases if c["case_id"] in done]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


# ----------------------------------------------------------------------------- Q&A questions (LLM paraphrase, code gold)
QA_TEMPLATES = {
    "count_wave": "How many servers are in wave {wave}?",
    "wave_of_host": "Which wave is {hostname} scheduled in?",
    "wave_gb": "What is the total transfer volume of wave {wave} in GB?",
    "high_risk_app": "Which servers of the {app} application are high risk?",
    "callers_of": "Which servers depend on {hostname} (call it)?",
    "instance_of": "Which AWS instance type is recommended for {hostname}?",
    "eol_count": "How many servers run an operating system that is past its end of support?",
    "app_cost": "What is the estimated monthly AWS cost of the {app} application?",
    "early_list": "Which servers were requested for early migration?",
    "wave_count": "How many migration waves are in the plan?",
    "cycle_members": "Which servers are part of a circular dependency?",
    "top_inbound": "Which server has the most other servers depending on it?",
}


class Paraphrase(BaseModel):
    qid: str
    question: str


class Paraphrases(BaseModel):
    items: list[Paraphrase]


def generate_questions(df: pd.DataFrame, refresh: bool) -> list[dict]:
    path = HERE / "qa_questions.jsonl"
    if path.exists() and not refresh:
        return [json.loads(line) for line in path.read_text().splitlines()]
    rng = random.Random(SEED)
    hosts = sorted(df["Hostname"])
    called = sorted({d.split(":")[0] for c in df["Outbound_Connections"].dropna() for d in str(c).split("|") if d})
    called_hosts = sorted(df[df["Server_ID"].isin(called)]["Hostname"])
    apps = sorted(a for a in df["App_Name"].unique() if a != "Unknown")
    qs = []

    def add(t, **params):
        qs.append({"qid": f"q{len(qs) + 1:02d}", "template": t, "params": params,
                   "template_text": QA_TEMPLATES[t].format(**params)})

    for w in (0, 1, 3):
        add("count_wave", wave=w)
    for w in (0, 2):
        add("wave_gb", wave=w)
    for h in rng.sample(hosts, 4):
        add("wave_of_host", hostname=h)
    for h in rng.sample(hosts, 3):
        add("instance_of", hostname=h)
    for h in rng.sample(called_hosts, 4):
        add("callers_of", hostname=h)
    for a in rng.sample(apps, 3):
        add("high_risk_app", app=a)
    for a in rng.sample(apps, 3):
        add("app_cost", app=a)
    for t in ("eol_count", "early_list", "wave_count", "cycle_members", "top_inbound"):
        add(t)
    out = structured_call(
        Paraphrases,
        "Rewrite each question the way a busy migration manager would ask it in chat. Keep every hostname, "
        "application name, and wave number exactly as written. Do not add new constraints.",
        "Questions:\n" + json.dumps([{"qid": q["qid"], "question": q["template_text"]} for q in qs]),
        agent="benchmark_generator", prompt_version="qa-paraphrase-v1", tracer=None, model=config.DATA_GEN_MODEL)
    para = {p.qid: p.question for p in out.items}
    for q in qs:
        text = para.get(q["qid"], "")
        keep = all(str(v) in text for v in q["params"].values() if not isinstance(v, int)) and \
            all(re.search(rf"\b{v}\b", text) for v in q["params"].values() if isinstance(v, int))
        q["question"] = text if (text and keep) else q["template_text"]
        q["paraphrased"] = bool(text and keep)
        q["generator_model"] = config.DATA_GEN_MODEL
    path.write_text("\n".join(json.dumps(q) for q in qs) + "\n")
    return qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="Call the data-generation model for everything again")
    ap.add_argument("--refresh-prompts", action="store_true", help="Regenerate only the constraint requests")
    ap.add_argument("--refresh-questions", action="store_true", help="Regenerate only the Q&A paraphrases")
    a = ap.parse_args()
    apps = apply_label_floors(generate_portfolio(a.refresh))
    print(f"Portfolio: {len(apps)} applications from {config.DATA_GEN_MODEL} "
          f"({sum(x['early_source'] == 'llm' for x in apps)} early by model, "
          f"{sum(x['early_source'] == 'generator_floor' for x in apps)} by generator floor)")
    built = {}
    for size in SIZES:
        df, truth = build_dataset(apps, size, SEED)
        df.to_csv(HERE / f"D3_{size}.csv", index=False)
        (HERE / f"D3_{size}_truth.json").write_text(json.dumps(truth, indent=2))
        built[f"D3_{size}"] = df
        print(f"D3_{size}: {truth['servers']} servers, {truth['edges']} edges, {truth['apps']} apps, "
              f"{len(truth['cycles'])} cycles, {len(truth['indivisible_groups_over_3000gb'])} indivisible >3TB, "
              f"{len(truth['early_ids'])} early, {len(truth['larger_than_sizing_matrix'])} > sizing matrix")
    d1 = pd.read_csv(HERE.parent / "sample.csv")
    prompts = generate_prompts({"D1": d1, "D3_100": built["D3_100"]}, a.refresh or a.refresh_prompts)
    print(f"Constraint prompts: {len(prompts)} ({sum(p['kind'] == 'ambiguous' for p in prompts)} ambiguous)")
    qs = generate_questions(d1, a.refresh or a.refresh_questions)
    print(f"Q&A questions: {len(qs)} ({sum(q['paraphrased'] for q in qs)} paraphrased by {config.DATA_GEN_MODEL})")


if __name__ == "__main__":
    main()
