"""LLM agents. Each agent reads tool output, decides an action, and is checked
by deterministic code. Every agent has a rule-based fallback so the pipeline
still completes (and records the fallback) when the model is unavailable."""
