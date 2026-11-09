"""Central configuration: paths, Neo4j connection, LLM endpoint and models.

Everything can be overridden with environment variables (or a local .env file)
so the same code runs on a laptop, in CI, or in the Streamlit app.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RUNS_DIR = Path(os.environ.get("P1_RUNS_DIR", ROOT / "runs"))
RESULTS_DIR = ROOT / "results"
PRICING_CATALOG_PATH = DATA_DIR / "aws_pricing_catalog.json"
OS_LIFECYCLE_PATH = DATA_DIR / "os_lifecycle.json"

NEO4J_URL = os.environ.get("P1_NEO4J_URL", "bolt://localhost:7688")
NEO4J_USER = os.environ.get("P1_NEO4J_USER", "neo4j")
# Local-only development default that matches compose.yml; override in .env.
NEO4J_PASSWORD = os.environ.get("P1_NEO4J_PASSWORD", "password123")

LLM_BASE_URL = os.environ.get("P1_LLM_BASE_URL", "https://openai.rc.asu.edu/v1")
LLM_API_KEY_ENV = "ASU_RC_API_KEY"
# Same model as the stakeholder baseline (B0) so comparisons are fair.
AGENT_MODEL = os.environ.get("P1_AGENT_MODEL", "qwen3-coder-30b-a3b-instruct")
# A different model family generates synthetic benchmark text, so the
# system under test never writes its own test questions.
DATA_GEN_MODEL = os.environ.get("P1_DATA_MODEL", "gpt-oss-120b")
# Per-attempt timeout and an output cap: a runaway generation on the shared ASU endpoint once
# blocked a run for 6+ minutes; no legitimate agent answer here needs more than a few thousand tokens.
LLM_TIMEOUT_SEC = float(os.environ.get("P1_LLM_TIMEOUT", "60"))
LLM_MAX_TOKENS = int(os.environ.get("P1_LLM_MAX_TOKENS", "6000"))

DEFAULT_MAX_TRANSFER_GB = 3000.0
REPAIR_MAX_ROUNDS = 3
TB_IN_GB = 1000.0
