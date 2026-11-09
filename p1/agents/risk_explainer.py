"""Risk Explainer: constrained explanations for deterministic risk scores.

Fixes the prototype failure where the LLM *assigned* risk labels without seeing
server IDs (22 of 40 labels wrong). Now code computes the score and factors;
the LLM only explains them, must cite exactly the factor codes it was given,
and every explanation is verified. Failures are retried once with the
verifier's error messages, then replaced by a deterministic template.
"""

from pydantic import BaseModel, Field

from p1.agents.common import dumps, structured_call
from p1.core.risk import FACTORS, template_explanation
from p1.core.verify import verify_risk_explanation
from p1.tracing import Tracer

PROMPT_VERSION = "risk-explainer-v2.0"
SYSTEM_PROMPT = f"""You explain migration risk scores that were already computed by rules.
For each server you receive its ID, OS, tags, risk level, and the exact list of risk factors
(codes with evidence). Write ONE or TWO short sentences per server that explain the risk using ONLY
those factors and their evidence. Rules:
- cited_factors must list exactly the factor codes given for that server, no more, no fewer.
- Do not mention any risk reason that is not in the server's factor list.
- Plain English for a migration architect; under 300 characters.
Factor meanings: {dumps({k: v[1] for k, v in FACTORS.items()})}"""
BATCH = 15


class RiskExplanation(BaseModel):
    server_id: str
    explanation: str
    cited_factors: list[str] = Field(default_factory=list)


class RiskExplanations(BaseModel):
    items: list[RiskExplanation] = Field(default_factory=list)


def explain_risks(servers: list[dict], tracer: Tracer | None, use_llm: bool = True,
                  model: str | None = None) -> tuple[dict[str, dict], dict]:
    """Returns ({server_id: {text, source}}, stats)."""
    out: dict[str, dict] = {}
    stats = {"servers": len(servers), "template_no_factors": 0, "llm_first_pass_ok": 0,
             "llm_retry_ok": 0, "fallback_template": 0, "verification_failures": 0, "llm_batches": 0}
    todo = []
    for s in servers:
        if not s["Risk_Factors"]:
            out[s["Server_ID"]] = {"text": template_explanation(s["Server_ID"], []), "source": "template"}
            stats["template_no_factors"] += 1
        else:
            todo.append(s)
    if not use_llm:
        for s in todo:
            out[s["Server_ID"]] = {"text": template_explanation(s["Server_ID"], s["Risk_Factors"]), "source": "template"}
            stats["fallback_template"] += 1
        return out, stats

    by_id = {s["Server_ID"]: s for s in todo}
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        failures: dict[str, list[str]] = {}
        for attempt in (1, 2):
            targets = batch if attempt == 1 else [by_id[sid] for sid in failures]
            if not targets:
                break
            payload = [{"server_id": s["Server_ID"], "os": s["OS_Version"], "tags": s["Business_Tags"],
                        "risk_level": s["Risk_Level"], "factors": s["Risk_Factors"]} for s in targets]
            user = "Servers:\n" + dumps(payload)
            if attempt == 2:
                user += "\n\nYour previous explanations for these servers were rejected:\n" + dumps(failures)
            stats["llm_batches"] += 1
            try:
                result = structured_call(RiskExplanations, SYSTEM_PROMPT, user, agent="risk_explainer",
                                         prompt_version=PROMPT_VERSION, tracer=tracer, model=model)
                got = {it.server_id: it for it in result.items}
            except Exception as exc:  # noqa: BLE001
                if tracer:
                    tracer.event("fallback", "risk_explainer", reason=f"{type(exc).__name__}: {exc}")
                got = {}
            new_failures = {}
            for s in targets:
                sid = s["Server_ID"]
                it = got.get(sid)
                errors = (["no explanation returned for this server"] if it is None
                          else verify_risk_explanation(sid, it.explanation, it.cited_factors, s["Risk_Factors"]))
                if errors:
                    new_failures[sid] = errors
                    stats["verification_failures"] += 1
                    if tracer:
                        tracer.event("verification", "risk_explanation", status="rejected", server_id=sid,
                                     attempt=attempt, errors=errors)
                else:
                    out[sid] = {"text": it.explanation.strip(), "source": "llm"}
                    stats["llm_first_pass_ok" if attempt == 1 else "llm_retry_ok"] += 1
            failures = new_failures
        for sid in failures:
            out[sid] = {"text": template_explanation(sid, by_id[sid]["Risk_Factors"]), "source": "template"}
            stats["fallback_template"] += 1
    return out, stats
