"""
05 — LangChain Guardrails
==========================
LangChain has its own ecosystem of output parsers that act as guardrails.
They slot cleanly into LCEL chains (the pipe | syntax) and integrate with
the same models, prompts, and memory you already use.

Topics covered:
  1. PydanticOutputParser      — structured output, strict validation
  2. OutputFixingParser        — wraps any parser with an auto-fix LLM call
  3. RetryWithErrorOutputParser — retries with the original error as feedback
  4. Custom output parser       — write your own guardrail as a Runnable
  5. Full guarded chain         — input guardrail → LLM → output guardrail

Run this file:
  uv run 05_langchain_guardrails.py
"""

import re
import os
from typing import Any
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.exceptions import OutputParserException

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

llm = ChatAnthropic(model=MODEL, max_tokens=600)
fast_llm = ChatAnthropic(model=MODEL, max_tokens=200)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PydanticOutputParser — the most common LangChain output guardrail
# ─────────────────────────────────────────────────────────────────────────────
# PydanticOutputParser does two things:
#   a. Injects format instructions into your prompt so the LLM knows the schema
#   b. Parses and validates the response into your Pydantic model

class JobPosting(BaseModel):
    title:        str        = Field(description="Job title")
    company:      str        = Field(description="Company name")
    salary_min:   int        = Field(description="Minimum salary in USD", ge=0)
    salary_max:   int        = Field(description="Maximum salary in USD", ge=0)
    skills:       list[str]  = Field(description="Required skills, 3-5 items", min_length=2)
    remote:       bool       = Field(description="True if the role is remote-friendly")
    description:  str        = Field(description="2-3 sentence job description", max_length=400)


def demo_pydantic_parser():
    parser  = PydanticOutputParser(pydantic_object=JobPosting)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You generate realistic fake job postings for demos. "
                   "{format_instructions}"),
        ("user",   "Create a job posting for a {role} at a tech startup."),
    ]).partial(format_instructions=parser.get_format_instructions())

    # Chain: prompt → LLM → parser (parser IS the guardrail here)
    chain = prompt | llm | parser

    result: JobPosting = chain.invoke({"role": "senior backend engineer"})

    print(f"  Title       : {result.title}")
    print(f"  Company     : {result.company}")
    print(f"  Salary      : ${result.salary_min:,} – ${result.salary_max:,}")
    print(f"  Remote      : {result.remote}")
    print(f"  Skills      : {result.skills}")
    print(f"  Description : {result.description[:80]}...")


# ─────────────────────────────────────────────────────────────────────────────
# 2. OutputFixingParser — wraps any parser with an auto-fix LLM call
# ─────────────────────────────────────────────────────────────────────────────
# When the base parser fails, OutputFixingParser sends the bad output + the
# parse error to a second LLM call that tries to fix the JSON.
# Cost: 1 extra LLM call on failure. Benefit: much higher success rate.

from langchain.output_parsers import OutputFixingParser


def demo_fixing_parser():
    base_parser   = PydanticOutputParser(pydantic_object=JobPosting)
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=fast_llm)

    # Intentionally malformed output — base parser would fail
    malformed = """{
      "title": "ML Engineer",
      "company": "DeepThought AI",
      "salary_min": "120000",
      "salary_max": "160000",
      "skills": "Python, PyTorch, LLMs",
      "remote": "yes",
      "description": "Build and deploy ML models."
    }"""

    print(f"  Malformed input snippet: ...salary_min: '120000' (string, should be int)...")

    try:
        result = fixing_parser.parse(malformed)
        print(f"  OutputFixingParser recovered: {result.title} @ {result.company}")
        print(f"  Salary: ${result.salary_min:,} – ${result.salary_max:,}  (types fixed: {type(result.salary_min).__name__})")
    except Exception as e:
        print(f"  Fixing parser also failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Custom Output Parser as a Guardrail
# ─────────────────────────────────────────────────────────────────────────────
# Sometimes you don't need Pydantic — you need a domain-specific rule.
# Implement BaseOutputParser and it slots into any LCEL chain.

from langchain_core.output_parsers import BaseOutputParser


class SentimentGuardParser(BaseOutputParser[dict]):
    """
    Parses a freeform LLM response into {text, sentiment, safe}.
    Blocks responses classified as negative or toxic.

    This is a custom output guardrail — reusable across chains.
    """

    NEGATIVE_WORDS = ["terrible", "awful", "horrible", "disgusting", "worthless", "hate"]
    POSITIVE_WORDS = ["great", "excellent", "amazing", "wonderful", "love", "fantastic"]

    def parse(self, text: str) -> dict:
        lowered = text.lower()

        neg_hits = [w for w in self.NEGATIVE_WORDS if w in lowered]
        pos_hits = [w for w in self.POSITIVE_WORDS if w in lowered]

        if len(neg_hits) > len(pos_hits):
            sentiment = "negative"
        elif pos_hits:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        safe = sentiment != "negative"

        if not safe:
            raise OutputParserException(
                f"Output guardrail: response has negative sentiment "
                f"(flagged words: {neg_hits})"
            )

        return {"text": text, "sentiment": sentiment, "safe": safe}

    @property
    def _type(self) -> str:
        return "sentiment_guard"


def demo_custom_parser():
    parser = SentimentGuardParser()
    fixing = OutputFixingParser.from_llm(parser=parser, llm=fast_llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a product reviewer. Be balanced and constructive."),
        ("user",   "Write one sentence reviewing: {product}"),
    ])

    chain = prompt | llm | fixing

    products = ["Sony WH-1000XM5 headphones", "A generic USB hub"]
    for product in products:
        try:
            result = chain.invoke({"product": product})
            print(f"  [{result['sentiment'].upper():8}] {result['text'][:80]}...")
        except OutputParserException as e:
            print(f"  [BLOCKED] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full Guarded Chain — input guardrail → LLM → output guardrail
# ─────────────────────────────────────────────────────────────────────────────
# This wires everything together with LCEL.
# Input guardrail: keyword filter as a RunnableLambda
# Output guardrail: PydanticOutputParser

class SupportTicket(BaseModel):
    category:    str  = Field(description="One of: billing, technical, account, general")
    priority:    str  = Field(description="One of: low, medium, high, critical")
    summary:     str  = Field(description="One-sentence ticket summary", max_length=120)
    suggested_response: str = Field(description="Suggested response to the user (2-3 sentences)")


BLOCKED_KEYWORDS = ["competitor", "openai", "gemini", "lawsuit", "legal action"]


def input_guard(inputs: dict) -> dict:
    """Input guardrail as a RunnableLambda — raises ValueError to halt the chain."""
    message = inputs.get("message", "")
    lowered = message.lower()
    for kw in BLOCKED_KEYWORDS:
        if kw in lowered:
            raise ValueError(f"Input guardrail blocked keyword: '{kw}'")
    if len(message) > 2000:
        raise ValueError(f"Input too long: {len(message)} chars (max 2000)")
    return inputs


def build_support_chain():
    parser = PydanticOutputParser(pydantic_object=SupportTicket)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a support ticket classifier. "
         "Analyse the user's message and produce a structured support ticket.\n"
         "{format_instructions}"),
        ("user", "{message}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    return (
        RunnableLambda(input_guard)   # ← input guardrail
        | prompt
        | llm
        | parser                      # ← output guardrail
    )


def demo_full_guarded_chain():
    chain = build_support_chain()

    cases = [
        ("Legitimate ticket",  "My subscription was charged twice this month. Order #12345."),
        ("Blocked keyword",    "Your competitor OpenAI is cheaper. Why should I stay?"),
    ]

    for label, message in cases:
        print(f"\n  [{label}]")
        print(f"  Input: {message[:70]}...")
        try:
            ticket = chain.invoke({"message": message})
            print(f"  Category : {ticket.category}")
            print(f"  Priority : {ticket.priority}")
            print(f"  Summary  : {ticket.summary}")
        except ValueError as e:
            print(f"  [BLOCKED] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all demos
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═' * 65}")
    print("  DEMO 1 — PydanticOutputParser (structured output)")
    print(f"{'═' * 65}\n")
    demo_pydantic_parser()

    print(f"\n{'═' * 65}")
    print("  DEMO 2 — OutputFixingParser (auto-fix malformed JSON)")
    print(f"{'═' * 65}\n")
    demo_fixing_parser()

    print(f"\n{'═' * 65}")
    print("  DEMO 3 — Custom SentimentGuardParser")
    print(f"{'═' * 65}\n")
    demo_custom_parser()

    print(f"\n{'═' * 65}")
    print("  DEMO 4 — Full guarded support-ticket chain")
    print(f"{'═' * 65}\n")
    demo_full_guarded_chain()
    print()


if __name__ == "__main__":
    main()
