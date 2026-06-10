"""
LLM Guardrails — Complete Practical Guide
==========================================
Guardrails are validation and safety layers that wrap LLM calls.
They ensure what goes IN and comes OUT of a model is safe, structured,
and within policy — without relying on the model to police itself.

Files in this project (run them in order):
  01_intro_to_guardrails.py   — Core concepts, taxonomy, minimal working demo
  02_input_guardrails.py      — PII detection, injection blocking, topic filters
  03_output_guardrails.py     — Pydantic schemas, structured outputs, retry loops
  04_guardrails_ai_library.py — The official guardrails-ai Guard class
  05_langchain_guardrails.py  — LangChain output parsers as production guardrails

Setup:
  cp .env.example .env        # then fill in your ANTHROPIC_API_KEY
  uv run 01_intro_to_guardrails.py
  uv run 02_input_guardrails.py
  ... and so on
"""

if __name__ == "__main__":
    print(__doc__)
