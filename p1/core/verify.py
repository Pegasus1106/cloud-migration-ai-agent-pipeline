"""Verifiers for LLM-written text. The LLM may phrase things freely, but every
factual claim it makes must be traceable to tool output; otherwise the text is
rejected, the model is retried once with the error, and then a deterministic
template is used (Milestone 1: "retry once with the validation error, then use
a rule-based fallback")."""

import re

# Words that signal a risk factor. If an explanation uses one of these words,
# the matching factor must actually apply to that server.
FACTOR_KEYWORDS = {
    "OS_EOL": [r"end[- ]of[- ](life|support)", r"\beol\b", r"unsupported", r"out of support", r"no longer supported"],
    "OS_EOL_SOON": [r"approaching (its )?end", r"nearing (its )?end", r"support ends (in|on|within)"],
    "COMPLIANCE": [r"\bpci\b", r"pci-dss", r"hipaa", r"\bsox\b", r"complian", r"regulat"],
    "HIGH_INBOUND": [r"depend(ed|s)? on (it|by)", r"many (other )?servers", r"inbound", r"shared service"],
    "IN_CYCLE": [r"cycl", r"circular"],
    "LARGE_DATA": [r"large (data|volume)", r"\btb\b", r"data volume"],
    "DATA_GAPS": [r"missing", r"imputed", r"incomplete"],
    "EXTERNAL": [r"external", r"customer[- ]facing", r"public"],
}
def unsupported_factor_claims(text: str, factor_codes: set[str]) -> list[str]:
    """Return factor codes the text alludes to that are NOT in factor_codes.
    The two lifecycle factors share vocabulary, so either one licenses both."""
    low = text.lower()
    allowed = set(factor_codes)
    if allowed & {"OS_EOL", "OS_EOL_SOON"}:
        allowed |= {"OS_EOL", "OS_EOL_SOON"}
    return [code for code, patterns in FACTOR_KEYWORDS.items()
            if code not in allowed and any(re.search(p, low) for p in patterns)]


def verify_risk_explanation(server_id: str, text: str, cited: list[str], factors: list[dict]) -> list[str]:
    codes = {f["code"] for f in factors}
    errors = []
    if not text or len(text) > 400:
        errors.append("explanation must be 1-2 sentences (max 400 characters)")
    bad_cites = [c for c in cited if c not in codes]
    if bad_cites:
        errors.append(f"cited factors {bad_cites} do not apply to {server_id}; allowed: {sorted(codes) or 'none'}")
    missing = sorted(codes - set(cited))
    if missing:
        errors.append(f"must cite every applicable factor; missing {missing}")
    extra = unsupported_factor_claims(text, codes)
    if extra:
        errors.append(f"text mentions factors that do not apply: {extra}")
    return errors


_NUM = re.compile(r"(?<![\w-])\d[\d,]*(?:\.\d+)?(?![\w-])")


def _number_variants(x: float) -> set[str]:
    out = {f"{x:.0f}", f"{x:,.0f}", f"{x:.1f}", f"{x:,.1f}"}
    if x >= 1000:
        out |= {f"{x / 1000:.1f}", f"{x / 1000:.0f}"}  # "2.9 TB" style
    return {v.replace(",", "") for v in out}


def verify_numbers(text: str, allowed_numbers: list[float], strip_tokens: list[str]) -> list[str]:
    """Every number in the text must equal (after rounding) a fact number.
    Hostnames, server IDs, and app names are removed first because they contain digits."""
    cleaned = text
    for tok in sorted(set(strip_tokens), key=len, reverse=True):
        if tok:
            cleaned = cleaned.replace(tok, " ")
    allowed: set[str] = set()
    for x in allowed_numbers:
        allowed |= _number_variants(float(x))
    bad = []
    for m in _NUM.finditer(cleaned):
        val = m.group(0).replace(",", "")
        if val not in allowed:
            bad.append(m.group(0))
    return bad


def verify_mentions(text: str, allowed: set[str], universe: set[str]) -> list[str]:
    """Names from `universe` (e.g. all hostnames) that appear in text but are not in `allowed`."""
    return sorted(n for n in universe if n and n not in allowed and re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text))
