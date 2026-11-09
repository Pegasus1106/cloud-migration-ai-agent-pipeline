"""Wave Explainer: grounded, fact-checked rationale for every wave.

The model receives each wave's facts (computed by code) and writes a short
rationale. A verifier rejects any number that is not in the facts and any
hostname/app that is not in that wave. Rejected waves are retried once with the
verifier's errors, then replaced by a deterministic template.
"""

from pydantic import BaseModel, Field

from p1.agents.common import dumps, structured_call
from p1.core.verify import verify_mentions, verify_numbers
from p1.tracing import Tracer

PROMPT_VERSION = "wave-explainer-v2.0"
SYSTEM_PROMPT = """You explain a migration wave plan to a migration architect.
For each wave you receive facts computed by the planner. Write 2-3 sentences per wave explaining what
moves in it and why it is scheduled there. Rules:
- The "why" must come ONLY from that wave's scheduling_reasons; do not invent other reasons.
- Refer to the wave exactly as "Wave <wave>" using the given index (wave 0 is the first wave).
- Use ONLY numbers that appear in that wave's facts. Do not compute new numbers (no sums, no percentages
  other than utilization_pct).
- Only name applications and hostnames listed in that wave's facts.
- Be concrete and brief."""
BATCH = 6


class WaveExplanation(BaseModel):
    wave: int
    explanation: str


class WaveExplanations(BaseModel):
    items: list[WaveExplanation] = Field(default_factory=list)


def wave_facts(plan_rows: list[dict], edges: list[tuple[str, str]], max_gb: float,
               infeasible: list[dict], approved: set[str]) -> list[dict]:
    by_wave: dict[int, list[dict]] = {}
    for r in plan_rows:
        if r.get("Wave_ID") is not None:
            by_wave.setdefault(int(r["Wave_ID"]), []).append(r)
    wave_of = {r["Server_ID"]: int(r["Wave_ID"]) for r in plan_rows if r.get("Wave_ID") is not None}
    facts = []
    for w in sorted(by_wave):
        rows = by_wave[w]
        ids = {r["Server_ID"] for r in rows}
        apps: dict[str, int] = {}
        for r in rows:
            apps[r["App_Name"]] = apps.get(r["App_Name"], 0) + 1
        total = round(sum(float(r["Transfer_GB"]) for r in rows), 1)
        over = next((g for g in infeasible if set(g["members"]) == ids), None)
        internal = sum(1 for a, b in edges if a in ids and b in ids)
        to_later = sum(1 for a, b in edges if a in ids and b in wave_of and wave_of[b] > w)
        from_earlier = sum(1 for a, b in edges if b in ids and a in wave_of and wave_of[a] < w)
        early_n = sum(bool(r.get("Early_Related")) for r in rows)
        is_pilot = any(bool(r.get("Pilot")) for r in rows)
        # The planner's actual reasons (from the priority rules), so the model explains, not invents.
        notes = []
        if is_pilot:
            notes.append("pilot wave: whole small, low-risk, isolated applications moved first to prove the process")
        if early_n:
            notes.append("holds early-migration requests and the servers that call them (highest priority)")
        if over:
            notes.append("one indivisible dependency group larger than the wave limit, so it is placed alone")
        if from_earlier:
            notes.append("continues applications whose calling servers moved in earlier waves")
        if internal:
            notes.append("keeps directly dependent servers together in the same wave")
        if not over and not is_pilot:
            notes.append("remaining capacity filled by priority: lower risk first, then higher projected savings")
        facts.append({
            "wave": w,
            "servers": len(rows),
            "total_transfer_gb": total,
            "limit_gb": max_gb,
            "utilization_pct": round(min(total, max_gb) / max_gb * 100),
            "applications": dict(sorted(apps.items(), key=lambda kv: -kv[1])),
            "hostnames": sorted(r["Hostname"] for r in rows)[:12],
            "early_requested_servers": early_n,
            "pilot": is_pilot,
            "scheduling_reasons": notes,
            "high_risk_servers": sum(r.get("Risk_Level") == "High" for r in rows),
            "dependencies_inside_wave": internal,
            "dependencies_to_later_waves": to_later,
            "dependencies_from_earlier_waves": from_earlier,
            "over_limit_indivisible_group": bool(over),
            "exception_approved": bool(over and over["group_id"] in approved),
        })
    return facts


def _numbers(f: dict) -> list[float]:
    nums = [f["wave"], f["servers"], f["total_transfer_gb"], f["limit_gb"], f["utilization_pct"],
            f["early_requested_servers"], f["high_risk_servers"], f["dependencies_inside_wave"],
            f["dependencies_to_later_waves"], f["dependencies_from_earlier_waves"], len(f["applications"])]
    nums += list(f["applications"].values())
    return nums


def _template(f: dict) -> str:
    apps = ", ".join(f"{a} ({n})" for a, n in list(f["applications"].items())[:4])
    parts = [f"Wave {f['wave']} moves {f['servers']} servers ({f['total_transfer_gb']:,.0f} GB of the "
             f"{f['limit_gb']:,.0f} GB limit): {apps}."]
    if f["pilot"]:
        parts.append("It is the low-risk pilot wave.")
    if f["early_requested_servers"]:
        parts.append(f"{f['early_requested_servers']} servers are early-migration requests or their prerequisites.")
    if f["over_limit_indivisible_group"]:
        parts.append("It holds one indivisible dependency group that exceeds the limit"
                     + (" (exception approved by reviewer)." if f["exception_approved"] else "."))
    parts.append(f"{f['dependencies_inside_wave']} dependencies stay inside the wave; "
                 f"{f['dependencies_to_later_waves']} point to later waves.")
    return " ".join(parts)


def explain_waves(facts: list[dict], all_hostnames: set[str], all_apps: set[str], tracer: Tracer | None,
                  use_llm: bool = True, model: str | None = None) -> tuple[dict[int, dict], dict]:
    out: dict[int, dict] = {}
    stats = {"waves": len(facts), "llm_first_pass_ok": 0, "llm_retry_ok": 0, "fallback_template": 0,
             "verification_failures": 0}
    if not use_llm:
        for f in facts:
            out[f["wave"]] = {"text": _template(f), "source": "template"}
        stats["fallback_template"] = len(facts)
        return out, stats
    by_wave = {f["wave"]: f for f in facts}
    for i in range(0, len(facts), BATCH):
        batch = facts[i:i + BATCH]
        failures: dict[int, list[str]] = {}
        for attempt in (1, 2):
            targets = batch if attempt == 1 else [by_wave[w] for w in failures]
            if not targets:
                break
            user = "Waves:\n" + dumps(targets)
            if attempt == 2:
                user += "\n\nYour previous text for these waves was rejected:\n" + dumps(failures)
            try:
                got = {it.wave: it.explanation for it in structured_call(
                    WaveExplanations, SYSTEM_PROMPT, user, agent="wave_explainer",
                    prompt_version=PROMPT_VERSION, tracer=tracer, model=model).items}
            except Exception as exc:  # noqa: BLE001
                if tracer:
                    tracer.event("fallback", "wave_explainer", reason=f"{type(exc).__name__}: {exc}")
                got = {}
            new_failures = {}
            for f in targets:
                text = got.get(f["wave"])
                if text is None:
                    errors = ["no explanation returned"]
                else:
                    names = list(f["applications"]) + f["hostnames"]
                    bad_nums = verify_numbers(text, _numbers(f), list(all_hostnames) + list(all_apps))
                    bad_names = verify_mentions(text, set(names), all_hostnames | all_apps)
                    errors = ([f"numbers not in facts: {bad_nums}"] if bad_nums else []) + \
                             ([f"names not in this wave: {bad_names}"] if bad_names else [])
                if errors:
                    new_failures[f["wave"]] = errors
                    stats["verification_failures"] += 1
                    if tracer:
                        tracer.event("verification", "wave_explanation", status="rejected", wave=f["wave"],
                                     attempt=attempt, errors=errors)
                else:
                    out[f["wave"]] = {"text": text.strip(), "source": "llm"}
                    stats["llm_first_pass_ok" if attempt == 1 else "llm_retry_ok"] += 1
            failures = new_failures
        for w in failures:
            out[w] = {"text": _template(by_wave[w]), "source": "template"}
            stats["fallback_template"] += 1
    return out, stats
