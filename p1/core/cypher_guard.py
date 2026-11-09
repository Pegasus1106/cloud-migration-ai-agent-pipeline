"""Read-only guard for LLM-generated Cypher (defence in depth: the query also
runs in a READ_ACCESS session, which Neo4j enforces server-side)."""

import re

_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|CALL\s*\{|"
    r"apoc\.(create|merge|refactor|periodic|load|export|do|cypher\.run)|gds\.[\w.]*\.(write|mutate)|"
    r"dbms\.|db\.create|gds\.graph\.(project|drop))\b",
    re.IGNORECASE,
)
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def check_read_only(cypher: str) -> tuple[bool, str]:
    if not cypher or not cypher.strip():
        return False, "REJECTED: empty query"
    stripped = _STRING.sub("''", _COMMENT.sub(" ", cypher))
    if ";" in stripped.strip().rstrip(";"):
        return False, "REJECTED: multiple statements are not allowed"
    m = _WRITE.search(stripped)
    if m:
        return False, f"REJECTED: write or admin operation '{m.group(0)}' is not allowed (read-only access)"
    if not re.search(r"\b(MATCH|RETURN|WITH|UNWIND|CALL\s+db\.(labels|schema))\b", stripped, re.IGNORECASE):
        return False, "REJECTED: query must read with MATCH ... RETURN"
    return True, "ok"
