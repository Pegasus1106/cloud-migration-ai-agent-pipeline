"""Streamlit web UI for the P1 agentic migration planner.

    streamlit run app.py

Flow: pick a dataset -> chat with the Constraint Agent -> confirm the configuration ->
run the supervisor (live node progress) -> decide human-review items -> inspect the plan,
the dependency graph, ask questions about the plan, and audit every agent action.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from p1 import config
from p1.core.data import load_and_normalize
from p1.core.graphdb import GraphStore, neo4j_lock
from p1.export import load_plan_into_neo4j
from p1.llm import llm_available
from p1.state import RunConfig
from p1.supervisor import NODE_LABELS, RunOptions, Supervisor
from p1.tracing import Tracer

# Chart tokens (dataviz reference palette): slot 1 for the single series, the reserved
# "critical" status color only for over-limit waves (always with a text label), muted ink for rules.
SERIES_1, CRITICAL, INK_2, ORANGE = "#2a78d6", "#d03b3b", "#52514e", "#eb6834"
DATASETS = {
    "D1 · Stakeholder sample (50 servers)": config.DATA_DIR / "sample.csv",
    "D2 · Synthetic chains (83 servers)": config.DATA_DIR / "synthetic_83.csv",
    "D3 · Benchmark 30": config.DATA_DIR / "benchmark" / "D3_30.csv",
    "D3 · Benchmark 100": config.DATA_DIR / "benchmark" / "D3_100.csv",
    "D3 · Benchmark 500": config.DATA_DIR / "benchmark" / "D3_500.csv",
}
MODELS = [config.AGENT_MODEL, "qwen3-235b-a22b", "gpt-oss-120b"]

st.set_page_config(page_title="Cloud Migration Planner", page_icon="☁️", layout="wide")
ss = st.session_state


def reset_run_state() -> None:
    for k in ("agent", "chat", "proposed", "confirmed", "sup", "events", "warnings", "qa", "approved"):
        ss.pop(k, None)


@st.cache_data(show_spinner=False)
def neo4j_ok() -> bool:
    with GraphStore() as gs:
        return gs.ping()


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Setup")
    choice = st.selectbox("Discovery dataset", list(DATASETS) + ["Upload a CSV"], key="dataset_choice",
                          on_change=reset_run_state)
    if choice == "Upload a CSV":
        up = st.file_uploader("Discovery CSV (same columns as data/sample.csv)", type="csv")
        if up is None:
            st.stop()
        tmp = Path(tempfile.gettempdir()) / f"p1-upload-{up.name}"
        tmp.write_bytes(up.getvalue())
        dataset_path = tmp
    else:
        dataset_path = DATASETS[choice]
    model = st.selectbox("Agent model (ASU Voyager)", MODELS, help="Default matches the stakeholder baseline")
    use_llm = st.toggle("Use LLM agents", value=llm_available(), disabled=not llm_available(),
                        help="Off = every agent uses its deterministic fallback (ablation).")
    st.caption(("🟢 " if llm_available() else "🔴 ") + f"LLM key ({config.LLM_API_KEY_ENV})")
    st.caption(("🟢 " if neo4j_ok() else "🟠 ") + f"Neo4j at {config.NEO4J_URL}"
               + ("" if neo4j_ok() else " — networkx fallback will be used"))
    if st.button("Start over", width="stretch"):
        reset_run_state()
        st.rerun()

try:
    df, edges, quality = load_and_normalize(dataset_path)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

st.title("Cloud Migration Planner")
st.caption("P1 agentic system — the language model interprets intent, chooses bounded actions, and explains; "
           "deterministic tools compute graph structure, sizes, prices, risk, waves, and validation.")
c = st.columns(5)
c[0].metric("Servers", quality["servers_after_dedup"])
c[1].metric("Dependencies", quality["edge_count"])
c[2].metric("Applications", quality["app_count"])
c[3].metric("Transfer volume", f"{quality['total_transfer_gb'] / 1000:,.1f} TB")
c[4].metric("Incomplete records", quality["rows_flagged_for_manual_audit"])

tabs = st.tabs(["① Constraints", "② Run & review", "③ Plan", "④ Dependency graph", "⑤ Ask the plan", "⑥ Agent trace"])

# ----------------------------------------------------------------------------- ① constraints
with tabs[0]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Tell the Constraint Agent what you need")
        st.caption("Example: “Keep each wave under 2.5 TB, move the CRM and payment gateway apps early, "
                   "and start with a small pilot wave.”")
        ss.setdefault("chat", [])
        for msg in ss["chat"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if use_llm:
            prompt = st.chat_input("Describe your migration constraints…")
            if prompt:
                if "agent" not in ss:
                    from p1.agents.constraint_agent import ConstraintAgent
                    ss["agent"] = ConstraintAgent(df, Tracer("ui-constraints"), model)
                ss["chat"].append({"role": "user", "content": prompt})
                with st.spinner("Constraint Agent is grounding your request in the data…"):
                    turn = ss["agent"].send(prompt)
                ss["chat"].append({"role": "assistant", "content": turn["reply"]})
                if turn["proposed"] is not None:
                    ss["proposed"], ss["warnings"] = turn["proposed"], turn["warnings"]
                    ss.pop("confirmed", None)
                st.rerun()
        else:
            st.info("LLM agents are off: set the constraints with the form on the right.")
    with right:
        st.subheader("Configuration")
        base = ss.get("proposed") or RunConfig()
        with st.form("cfg"):
            max_gb = st.number_input("Wave limit (GB)", 100.0, 1_000_000.0, float(base.max_transfer_gb), step=100.0)
            cloud = st.selectbox("Target cloud", ["AWS", "AZURE"], index=0 if base.target_cloud == "AWS" else 1)
            apps = st.multiselect("Move these applications early", sorted(df["App_Name"].unique()),
                                  default=[a for a in base.early_app_names if a in set(df["App_Name"])])
            ids = st.multiselect("…and these servers", sorted(df["Server_ID"]),
                                 default=[i for i in base.early_server_ids if i in set(df["Server_ID"])])
            honor = st.checkbox("Honor Early_Migration_Requested flags from the CSV", base.honor_early_migration_flags)
            pilot = st.checkbox("Start with a low-risk pilot wave", base.pilot_wave)
            confirm = st.form_submit_button("✅ Confirm configuration", type="primary", width="stretch")
        for w in ss.get("warnings") or []:
            st.warning(w)
        if confirm:
            source = "Constraint Agent proposal" if ss.get("proposed") else "form"
            ss["confirmed"] = RunConfig(max_transfer_gb=max_gb, target_cloud=cloud, early_app_names=apps,
                                        early_server_ids=ids, honor_early_migration_flags=honor, pilot_wave=pilot,
                                        source_request=base.source_request or source, confirmed=True,
                                        confirmed_by=f"user (web UI, {source}) at {datetime.now():%H:%M:%S}")
            ss.pop("sup", None)
        if ss.get("confirmed"):
            st.success(f"Confirmed by {ss['confirmed'].confirmed_by}. Go to **② Run & review**.")
        else:
            st.info("Nothing is planned until you confirm (the agent never confirms for you).")

# ----------------------------------------------------------------------------- ② run & review
with tabs[1]:
    if not ss.get("confirmed"):
        st.info("Confirm a configuration in ① first.")
    else:
        opts_cols = st.columns(3)
        enable_repair = opts_cols[0].checkbox("Repair Agent", True)
        explain = opts_cols[1].checkbox("Explanations", True)
        same_wave = opts_cols[2].checkbox("Same-wave dependencies", True, help="Off = prototype packing")
        if st.button("▶ Run planner", type="primary", disabled="sup" in ss and ss["sup"].interrupt_payload is not None):
            opts = RunOptions(review_mode="interactive", enable_repair=enable_repair, allow_same_wave=same_wave,
                              use_llm=use_llm, model=model, explain=explain)
            ss["sup"] = Supervisor(dataset_path, ss["confirmed"], opts)
            ss["events"] = []
            if "agent" in ss:  # keep the Constraint Agent's chat trace with the run it configured
                (ss["sup"].out_dir / "constraint_trace.jsonl").write_text(
                    "\n".join(json.dumps(e, default=str) for e in ss["agent"].tracer.events) + "\n")
                (ss["sup"].out_dir / "constraint_chat.json").write_text(json.dumps(ss["chat"], indent=2))
            with st.status("Running supervisor…", expanded=True) as status:
                for ev in ss["sup"].stream():
                    ss["events"].append(ev)
                    status.write(f"**{ev['label']}** — {ev['progress']}")
                status.update(label="Waiting for your review" if ss["sup"].interrupt_payload else "Run complete",
                              state="running" if ss["sup"].interrupt_payload else "complete")
        sup = ss.get("sup")
        if sup is not None:
            with st.expander("Node log", expanded=False):
                st.dataframe(pd.DataFrame(ss.get("events", [])), hide_index=True, width="stretch")
            if sup.interrupt_payload:
                st.subheader("Human review")
                st.caption("Code decided which items need a person and which actions are allowed; the agent drafted "
                           "the summary and a recommendation; you decide.")
                with st.form("review"):
                    decisions = {}
                    for item in sup.interrupt_payload["items"]:
                        st.markdown(f"**{item['title']}**  \n{item.get('summary', '')}")
                        if item.get("suggestion"):
                            st.caption(f"💡 {item['suggestion']}  ·  drafted by {item.get('drafted_by')}")
                        opts = item["allowed_actions"]
                        rec = item.get("recommended_action")
                        action = st.radio("Decision", opts, index=opts.index(rec) if rec in opts else 0,
                                          key=f"d-{item['id']}", horizontal=True)
                        note = st.text_input("Comment", key=f"c-{item['id']}")
                        decisions[item["id"]] = {"action": action, "comment": note}
                        st.divider()
                    submit = st.form_submit_button("Submit decisions", type="primary")
                if submit:
                    with st.status("Resuming…", expanded=True) as status:
                        for ev in sup.resume(decisions):
                            ss["events"].append(ev)
                            status.write(f"**{ev['label']}** — {ev['progress']}")
                    st.rerun()
            elif sup.state.get("status"):
                stt = sup.state
                if stt["status"] == "ready_for_approval":
                    st.success("Plan passed validation and is ready for your approval.")
                    if st.button("Approve plan", type="primary", disabled=ss.get("approved") == sup.run_id):
                        log_path = sup.out_dir / "run_log.json"
                        log = json.loads(log_path.read_text())
                        log.update(status="approved", approved_by="user (web UI)",
                                   approved_at=datetime.now().isoformat(timespec="seconds"))
                        log_path.write_text(json.dumps(log, indent=2))
                        ss["approved"] = sup.run_id
                    if ss.get("approved") == sup.run_id:
                        st.info(f"Approved. Files: `{sup.out_dir}`")
                else:
                    st.error(f"Status: {stt['status']} — {stt.get('stop_reason', '')}")

# ----------------------------------------------------------------------------- ③ plan
sup = ss.get("sup")
done = sup is not None and sup.interrupt_payload is None and bool(sup.state.get("outputs"))
with tabs[2]:
    if not done:
        st.info("Run the planner first.")
    else:
        stt = sup.state
        m, f = stt["validation"]["metrics"], stt["finops"]
        k = st.columns(6)
        k[0].metric("Waves", m["wave_count"])
        k[1].metric("Dependencies kept together", f"{m['dependency_edges'] - m['cross_wave_edges']}/{m['dependency_edges']}")
        k[2].metric("Wave utilization", f"{m['utilization_pct']}%" if m["utilization_pct"] is not None else "—")
        k[3].metric("Early requests on time", f"{m['early_satisfied_pct']}%" if m["early_satisfied_pct"] is not None else "—")
        k[4].metric("Est. AWS / month", f"${f['total_aws_monthly']:,.0f}",
                    delta=f"-${f['total_savings_monthly']:,.0f} vs today", delta_color="inverse")
        k[5].metric("Validation", "passed" if stt["validation"]["passed"] else "blocked")
        st.caption(f"Costs: {f['disclaimer']}")

        plan = pd.read_csv(stt["outputs"]["plan_csv"])
        plan = plan[plan["Wave_ID"].notna()].copy()
        plan["Wave_ID"] = plan["Wave_ID"].astype(int)
        limit = stt["config"]["max_transfer_gb"]
        waves = plan.groupby("Wave_ID").agg(servers=("Server_ID", "count"), gb=("Transfer_GB", "sum"),
                                            apps=("App_Name", lambda s: ", ".join(sorted(set(s))[:6]))).reset_index()
        waves["state"] = waves["gb"].map(lambda g: "Over limit (approved exception)" if g > limit else "Within limit")
        st.subheader("Transfer volume per wave")
        base = alt.Chart(waves).encode(x=alt.X("Wave_ID:O", title="Wave", axis=alt.Axis(labelAngle=0)))
        bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=18).encode(
            y=alt.Y("gb:Q", title="Transfer volume (GB)", axis=alt.Axis(gridColor="#e9e8e4")),
            color=alt.Color("state:N", scale=alt.Scale(domain=["Within limit", "Over limit (approved exception)"],
                                                       range=[SERIES_1, CRITICAL]), legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("Wave_ID:O", title="Wave"), alt.Tooltip("servers:Q", title="Servers"),
                     alt.Tooltip("gb:Q", title="GB", format=",.0f"), alt.Tooltip("apps:N", title="Applications"),
                     alt.Tooltip("state:N", title="State")])
        rule = alt.Chart(pd.DataFrame({"y": [limit]})).mark_rule(color=INK_2, strokeWidth=1).encode(y="y:Q")
        rule_label = alt.Chart(pd.DataFrame({"y": [limit], "t": [f"{limit:,.0f} GB limit"]})).mark_text(
            align="left", dx=4, dy=-6, color=INK_2, fontSize=11).encode(y="y:Q", text="t:N", x=alt.value(0))
        st.altair_chart((bars + rule + rule_label).properties(height=280), width="stretch")

        st.subheader("Why each wave looks the way it does")
        for w, info in sorted((stt.get("explanations") or {}).get("waves", {}).items(), key=lambda kv: int(kv[0])):
            with st.expander(f"Wave {w} · {info['servers']} servers · {info['total_transfer_gb']:,.0f} GB"
                             f"{' · pilot' if info['pilot'] else ''}"):
                st.write(info["text"])
                st.caption(f"Explanation source: {info['source']} (numbers and names checked against the plan)")

        st.subheader("Server plan")
        fc = st.columns(3)
        wsel = fc[0].multiselect("Waves", sorted(plan["Wave_ID"].unique()))
        rsel = fc[1].multiselect("Risk level", ["High", "Medium", "Low"])
        asel = fc[2].multiselect("Applications", sorted(plan["App_Name"].unique()))
        view = plan
        if wsel:
            view = view[view["Wave_ID"].isin(wsel)]
        if rsel:
            view = view[view["Risk_Level"].isin(rsel)]
        if asel:
            view = view[view["App_Name"].isin(asel)]
        st.dataframe(view[["Wave_ID", "Server_ID", "Hostname", "App_Name", "OS_Version", "Target_Size", "Size_Source",
                           "AWS_Monthly_Cost", "Current_Monthly_Cost", "Risk_Level", "Risk_Factor_Codes",
                           "Risk_Explanation", "Transfer_GB", "Inbound_Degree", "Early_Related", "Pilot"]],
                     hide_index=True, width="stretch", height=360)
        d = st.columns(3)
        d[0].download_button("Download plan.csv", Path(stt["outputs"]["plan_csv"]).read_bytes(), "plan.csv")
        d[1].download_button("Download explanations.md", Path(stt["outputs"]["explanations_md"]).read_bytes(),
                             "explanations.md")
        d[2].download_button("Download run_log.json", Path(stt["outputs"]["run_log"]).read_bytes(), "run_log.json")

# ----------------------------------------------------------------------------- ④ graph
with tabs[3]:
    if not done:
        st.info("Run the planner first.")
    else:
        stt = sup.state
        servers = {s["Server_ID"]: s for s in stt["servers"] if s.get("Wave_ID") is not None}
        apps_all = sorted({s["App_Name"] for s in servers.values()})
        split_apps = sorted({servers[x]["App_Name"] for e in stt["edges"] for x in (e["src"], e["dst"])
                             if e["src"] in servers and e["dst"] in servers
                             and servers[e["src"]]["Wave_ID"] != servers[e["dst"]]["Wave_ID"]})
        # Default view: applications with dependencies split across waves (the part a reviewer checks).
        default_apps = apps_all if len(servers) <= 25 else (split_apps[:8] or apps_all[:4])
        pick = st.multiselect("Applications to draw (servers they call are included)", apps_all, default=default_apps)
        nodes = {sid for sid, s in servers.items() if s["App_Name"] in pick}
        edge_list = [(e["src"], e["dst"]) for e in stt["edges"]]
        nodes |= {b for a, b in edge_list if a in nodes and b in servers}
        st.caption("Boxes are waves. Gray arrows stay inside a wave; orange arrows are dependencies split across "
                   "waves (caller → callee). Bold outline = high risk.")
        lines = ["digraph G {", 'rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" '
                 'fillcolor="#fcfcfb" color="#c9c8c3" fontname="Helvetica" fontsize=10]; edge [color="#9c9b96"];']
        for w in sorted({servers[n]["Wave_ID"] for n in nodes}):
            lines.append(f'subgraph cluster_{w} {{ label="Wave {w}"; color="#c9c8c3"; fontname="Helvetica";')
            for n in sorted(n for n in nodes if servers[n]["Wave_ID"] == w):
                s = servers[n]
                pen = ' penwidth=2.5 color="#52514e"' if s["Risk_Level"] == "High" else ""
                lines.append(f'"{n}" [label="{s["Hostname"]}\\n{s["App_Name"]}"{pen}];')
            lines.append("}")
        for a, b in edge_list:
            if a in nodes and b in nodes:
                cut = servers[a]["Wave_ID"] != servers[b]["Wave_ID"]
                lines.append(f'"{a}" -> "{b}"' + (f' [color="{ORANGE}" penwidth=1.6]' if cut else "") + ";")
        lines.append("}")
        if len(nodes) > 150:
            st.warning("Too many servers to draw clearly; pick fewer applications.")
        else:
            st.graphviz_chart("\n".join(lines), width="stretch")

# ----------------------------------------------------------------------------- ⑤ ask
with tabs[4]:
    if not done:
        st.info("Run the planner first.")
    elif sup.state["graph_stats"].get("engine") != "neo4j-gds":
        st.warning("“Ask the plan” needs Neo4j (this run used the networkx fallback).")
    elif not use_llm:
        st.info("Turn on LLM agents to ask questions.")
    else:
        st.caption("The Q&A agent writes read-only Cypher, runs it, fixes its own query on errors, and answers from "
                   "the returned rows. Try: “Which servers depend on hr-db-01?” or “Which high-risk servers are in wave 1?”")
        ss.setdefault("qa", [])
        q = st.chat_input("Ask about this plan…", key="qa_input")
        if q:
            from p1.agents.qa_agent import ask
            with st.spinner("Querying Neo4j…"):
                with neo4j_lock():  # no other run can reset Neo4j between loading and querying
                    if load_plan_into_neo4j(sup.state):
                        st.toast("Reloaded this run's plan into Neo4j")
                    with GraphStore() as store:
                        ss["qa"].append(ask(q, store, sup.tracer, model))
        for r in reversed(ss["qa"]):
            with st.chat_message("user"):
                st.write(r["question"])
            with st.chat_message("assistant"):
                st.write(r["answer"] or f"(no answer) {r.get('error') or ''}")
                st.caption(f"{r['attempts']} quer{'y' if r['attempts'] == 1 else 'ies'}, "
                           f"{r['query_errors']} self-corrected, {r['latency_ms'] / 1000:.1f}s")
                with st.expander("Cypher and rows"):
                    st.code(r["cypher"] or "", language="cypher")
                    st.dataframe(pd.DataFrame(r["rows"]), width="stretch")

# ----------------------------------------------------------------------------- ⑥ trace
with tabs[5]:
    if sup is None:
        st.info("Run the planner first.")
    else:
        s = sup.tracer.summary()
        k = st.columns(5)
        k[0].metric("LLM calls", s["llm_calls"])
        k[1].metric("Tokens", f"{s['tokens_in'] + s['tokens_out']:,}")
        k[2].metric("Tool calls", s["tool_calls"], delta=f"{s['tool_errors']} rejected/errors", delta_color="off")
        k[3].metric("Verifier rejections", sum(1 for e in sup.tracer.events if e["kind"] == "verification"))
        k[4].metric("Fallbacks", s["fallbacks"])
        timings = sup.state.get("timings") or {}
        if timings:
            tdf = pd.DataFrame({"node": [NODE_LABELS.get(n, n) for n in timings], "seconds": [v / 1000 for v in timings.values()]})
            st.altair_chart(alt.Chart(tdf).mark_bar(color=SERIES_1, cornerRadiusEnd=4, size=14).encode(
                x=alt.X("seconds:Q", title="Seconds (all rounds)"), y=alt.Y("node:N", sort="-x", title=None),
                tooltip=["node", alt.Tooltip("seconds:Q", format=".1f")]).properties(height=28 * len(tdf) + 20),
                width="stretch")
        if sup.state.get("repair_log"):
            st.subheader("Repair Agent actions")
            st.dataframe(pd.DataFrame([{**r, "round": rnd["round"]} for rnd in sup.state["repair_log"]
                                       for r in rnd["resizes"]]), hide_index=True, width="stretch")
        st.subheader("Event log")
        ev = pd.DataFrame(sup.tracer.events)
        kinds = st.multiselect("Kinds", sorted(ev["kind"].unique()), default=[k for k in ("llm", "tool", "verification",
                                                                                     "fallback") if k in set(ev["kind"])])
        cols = [c for c in ["kind", "name", "agent", "status", "model", "latency_ms", "tokens_in", "tokens_out",
                            "input", "output", "errors", "error", "reason", "progress"] if c in ev.columns]
        st.dataframe(ev[ev["kind"].isin(kinds)][cols], hide_index=True, width="stretch", height=380)
