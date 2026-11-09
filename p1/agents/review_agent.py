"""Review drafting for the human review queue.

Deterministic code decides WHICH items need a human and WHICH actions are
allowed for each (approve an over-limit indivisible group, acknowledge an
imputed record, exclude a server, reject the plan). The LLM writes the
reviewer-facing summary, suggests practical mitigations, and recommends one of
the allowed actions. Its text is number-checked against the item's facts; the
human makes the decision.
"""

from typing import Literal

from pydantic import BaseModel, Field

from p1.agents.common import dumps, structured_call
from p1.core.verify import verify_numbers
from p1.tracing import Tracer

PROMPT_VERSION = "review-v2.0"
SYSTEM_PROMPT = """You prepare items for a human migration architect's review queue.
For each item you get its facts and the allowed actions. Write:
- summary: 1-2 plain-English sentences explaining the issue and why it matters. Only use numbers
  that appear in the facts.
- suggestion: one practical mitigation an architect would consider (for very large data, e.g. pre-copying
  data with replication or an offline transfer device before the cutover window).
- recommended_action: exactly one of the item's allowed actions.
Do not invent servers, numbers, or actions."""

ACTIONS = Literal["approve_exception", "acknowledge", "exclude_server", "reject_plan"]
BATCH = 15


class Draft(BaseModel):
    item_id: str
    summary: str
    suggestion: str
    recommended_action: ACTIONS


class Drafts(BaseModel):
    items: list[Draft] = Field(default_factory=list)


def _template(item: dict) -> dict:
    f = item["facts"]
    kind = item["kind"]
    if kind == "over_limit_group":
        s = (f"{f['servers']} servers form an indivisible dependency group needing {f['total_gb']:,.0f} GB, "
             f"above the {f['limit_gb']:,.0f} GB wave limit.")
        g = "Pre-copy the data before the cutover window, or approve a dedicated over-limit wave."
        rec = "approve_exception"
    elif kind == "data_gap":
        s = f"{item['server_ids'][0]} had missing {', '.join(f['missing_fields'])}; defaults were used."
        g = "Confirm the values with the server owner before the wave is scheduled."
        rec = "acknowledge"
    elif kind == "early_infeasible":
        s = f"The early request for {item['server_ids'][0]} cannot be met: {f['reason']}."
        g = "Ask the requester whether the servers that call it can also move early."
        rec = "acknowledge"
    else:
        s = item.get("title", "Review required.")
        g = "Review the details and decide."
        rec = item["allowed_actions"][0]
    return {"summary": s, "suggestion": g, "recommended_action": rec, "drafted_by": "template"}


def draft_items(items: list[dict], tracer: Tracer | None, use_llm: bool = True, model: str | None = None) -> list[dict]:
    pending = [i for i in items if not i.get("summary")]
    if not pending:
        return items
    drafts: dict[str, dict] = {}
    by_id = {i["id"]: i for i in pending}
    for start in range(0, len(pending) if use_llm else 0, BATCH):
        chunk = pending[start:start + BATCH]
        try:
            payload = [{"item_id": i["id"], "kind": i["kind"], "title": i["title"], "facts": i["facts"],
                        "allowed_actions": i["allowed_actions"]} for i in chunk]
            out = structured_call(Drafts, SYSTEM_PROMPT, "Items:\n" + dumps(payload), agent="review_agent",
                                  prompt_version=PROMPT_VERSION, tracer=tracer, model=model)
            for d in out.items:
                item = by_id.get(d.item_id)
                if item is None or d.recommended_action not in item["allowed_actions"]:
                    continue
                numbers = [float(x) for x in _fact_numbers(item["facts"])]
                bad = verify_numbers(d.summary + " " + d.suggestion, numbers, item["server_ids"] + [item["id"]])
                if bad:
                    if tracer:
                        tracer.event("verification", "review_draft", status="rejected", item=item["id"],
                                     unsupported_numbers=bad)
                    continue
                drafts[d.item_id] = {"summary": d.summary, "suggestion": d.suggestion,
                                     "recommended_action": d.recommended_action, "drafted_by": "llm"}
        except Exception as exc:  # noqa: BLE001
            if tracer:
                tracer.event("fallback", "review_agent", reason=f"{type(exc).__name__}: {exc}")
    for item in pending:
        item.update(drafts.get(item["id"]) or _template(item))
    return items


def _fact_numbers(facts: dict) -> list:
    out = []
    for v in facts.values():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append(v)
        elif isinstance(v, list):
            out.extend(x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool))
    return out
