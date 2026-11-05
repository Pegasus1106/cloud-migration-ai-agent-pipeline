"""
Shared configuration and helpers for the P1 prototype.
========================================================
Centralizes the Neo4j connection (this project runs its own container on
offset ports 7475/7688 so it can coexist with the baseline's 7474/7687)
and the ASU RC LLM client, so every agent module uses identical settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv()

NEO4J_URL = os.environ.get("P1_NEO4J_URL", "bolt://localhost:7688")
NEO4J_USER = os.environ.get("P1_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("P1_NEO4J_PASSWORD", "password123")

MODEL_NAME = "qwen3-coder-30b-a3b-instruct"
LLM_BASE_URL = "https://openai.rc.asu.edu/v1"

PRICING_CATALOG_PATH = Path(__file__).parent / "data" / "aws_pricing_catalog.json"
DEFAULT_RUN_CONFIG_PATH = Path(__file__).parent / "data" / "run_config.json"
DEFAULT_MAX_TRANSFER_GB = 3000.0


def get_graph() -> Neo4jGraph:
    """Return a Neo4jGraph instance pointed at this project's container."""
    return Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASSWORD)


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointing to the ASU Voyager endpoint."""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=os.environ.get("ASU_RC_API_KEY", "not-set"),
        model=MODEL_NAME,
        temperature=temperature,
    )
