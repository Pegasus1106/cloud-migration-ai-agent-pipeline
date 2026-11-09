"""Small helpers shared by the agents."""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from p1.llm import get_llm
from p1.tracing import Tracer


def run_config(tracer: Tracer | None, agent: str, prompt_version: str, recursion_limit: int = 25) -> dict:
    cfg: dict = {"recursion_limit": recursion_limit}
    if tracer is not None:
        cfg["callbacks"] = tracer.callbacks(agent, prompt_version)
    return cfg


def structured_call(schema, system: str, user: str, *, agent: str, prompt_version: str,
                    tracer: Tracer | None, model: str | None = None):
    llm = get_llm(model).with_structured_output(schema)
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=user)],
                      config=run_config(tracer, agent, prompt_version))


_ARTIFACTS = re.compile(r"</?tool_call>|<\|im_end\|>")


def last_ai_text(messages) -> str:
    """Final assistant text, without tool-call markup some models leak into content."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str):
            text = _ARTIFACTS.sub("", m.content).strip()
            if text:
                return text
    return ""


def dumps(obj) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
