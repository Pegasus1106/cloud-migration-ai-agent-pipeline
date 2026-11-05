"""
Constraint Agent (P1, new in this milestone)
================================================
Translates a free-text stakeholder request (e.g. "keep each wave under
3000 GB and move the CRM and marketing blog servers early") into a typed,
user-confirmed configuration. The LLM only extracts intent; it never
executes planning itself. The extracted configuration is always shown to
the user (or auto-confirmed in non-interactive batch mode, which is
logged as such) before the rest of the pipeline reads it, per the
Milestone 1 design: "conversation history is not trusted as configuration
until the user confirms the extracted values."

Usage:
    python constraint_agent.py "<free text request>" [--yes]

If no request is given, a sensible default is used so the pipeline
remains runnable non-interactively for reproducibility.
"""

import json
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from common import DEFAULT_MAX_TRANSFER_GB, DEFAULT_RUN_CONFIG_PATH, get_llm

DEFAULT_REQUEST = (
    "Keep each migration wave under 3000 GB of transfer volume, target "
    "AWS, and move any server flagged Early_Migration_Requested as early "
    "as dependency order allows."
)


class RunConfig(BaseModel):
    max_transfer_gb: float = Field(
        default=DEFAULT_MAX_TRANSFER_GB,
        description="Maximum total Transfer_GB allowed in a single migration wave.",
    )
    target_cloud: str = Field(
        default="AWS", description="Target cloud platform, e.g. AWS or AZURE."
    )
    honor_early_migration_flags: bool = Field(
        default=True,
        description="Whether to prioritize servers flagged Early_Migration_Requested.",
    )
    notes: str = Field(
        default="", description="Any other constraint the user mentioned that is not yet structured."
    )


def extract_config(request_text: str) -> RunConfig:
    """Ask the LLM to extract a structured RunConfig from free text."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(RunConfig)
    system = SystemMessage(
        content=(
            "Extract migration planning constraints from the user's request "
            "into the given schema. Use the schema defaults for anything "
            "not mentioned. Do not invent numeric limits the user did not "
            "state unless the schema default already covers it."
        )
    )
    result = structured_llm.invoke([system, HumanMessage(content=request_text)])
    if isinstance(result, RunConfig):
        return result
    return RunConfig(**result)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--yes"]
    auto_confirm = "--yes" in sys.argv[1:]
    request_text = " ".join(args) if args else DEFAULT_REQUEST

    print(f"User request: {request_text!r}")
    try:
        config = extract_config(request_text)
    except Exception as exc:
        print(f"LLM extraction failed ({exc}). Falling back to schema defaults.")
        config = RunConfig()

    config_dict = config.model_dump()
    config_dict["source_request"] = request_text
    config_dict["auto_confirmed"] = auto_confirm

    print("\nExtracted configuration (requires confirmation before planning runs):")
    print(json.dumps(config_dict, indent=2))

    if auto_confirm:
        print("\n--yes passed: auto-confirmed for non-interactive/reproducible runs.")
        confirmed = True
    else:
        answer = input("\nConfirm this configuration? [y/N]: ").strip().lower()
        confirmed = answer == "y"

    config_dict["confirmed"] = confirmed
    DEFAULT_RUN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_RUN_CONFIG_PATH.write_text(json.dumps(config_dict, indent=2))

    if not confirmed:
        print(f"\nNot confirmed. {DEFAULT_RUN_CONFIG_PATH} written with confirmed=false; "
              "the supervisor will refuse to plan with an unconfirmed configuration.")
        sys.exit(1)

    print(f"\nConfirmed. Configuration saved to {DEFAULT_RUN_CONFIG_PATH}")


if __name__ == "__main__":
    main()
