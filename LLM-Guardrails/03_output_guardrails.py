"""
03 — Output Guardrails
=======================
Everything you can validate AFTER the LLM responds.

The core problem: LLMs return free-form text. Your application needs
structured, predictable, policy-compliant data.

Topics covered:
  1. The raw-output problem — why you can't trust unvalidated LLM responses
  2. Pydantic structured output — define the shape you expect
  3. Retry loop — ask the LLM to fix its own mistakes
  4. Output sanitization — strip HTML/markdown, normalise whitespace
  5. Content policy check — flag responses that violate policy
  6. Full pipeline: parse → validate → retry → sanitize

Run this file:
  uv run 03_output_guardrails.py
"""

import re
import json
from typing import TypeVar, Type
from pydantic import BaseModel, Field, ValidationError
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
MODEL  = "claude-haiku-4-5-20251001"

T = TypeVar("T", bound=BaseModel)


# ─────────────────────────────────────────────────────────────────────────────
# 1. The Raw-Output Problem
# ─────────────────────────────────────────────────────────────────────────────

def demo_raw_output_problem():
    """
    Ask the LLM for structured data without any validation.
    Notice how the output format can vary across calls.
    """
    prompt = "Give me the top 3 programming languages with their year of creation."
    response = client.messages.create(
        model=MODEL, max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    print("Raw output (unpredictable structure):")
    print(text)
    print()
    # Try to JSON-parse it — this will likely fail because the LLM added prose
    try:
        parsed = json.loads(text)
        print("Happened to be valid JSON:", parsed)
    except json.JSONDecodeError:
        print("Cannot parse as JSON — this is the problem output guardrails solve.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pydantic Structured Output
# ─────────────────────────────────────────────────────────────────────────────

class Language(BaseModel):
    name:         str  = Field(description="Name of the programming language")
    year_created: int  = Field(description="Year the language was created", ge=1940, le=2030)
    paradigm:     str  = Field(description="Main paradigm: functional, OOP, procedural, etc.")


class LanguageList(BaseModel):
    languages: list[Language] = Field(description="List of programming languages", min_length=1)


def extract_json_block(text: str) -> str:
    """
    Pulls the first JSON object or array out of a string.
    LLMs often wrap JSON in prose or markdown code fences.
    """
    # Try to find a JSON block inside markdown fences first
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
    if fence_match:
        return fence_match.group(1)

    # Fall back to finding the first { or [ and extracting to its closing bracket
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start=start):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
    return text  # return as-is if nothing found


def parse_structured_output(text: str, model_class: Type[T]) -> T:
    """
    Attempts to parse LLM output into a Pydantic model.
    Raises ValidationError if parsing fails.
    """
    json_str = extract_json_block(text)
    data = json.loads(json_str)
    return model_class.model_validate(data)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Retry Loop — Ask the LLM to Fix Its Own Mistakes
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a data extraction assistant.
Always respond with valid JSON only — no prose, no markdown fences, no extra text.
Your JSON must exactly match the requested schema."""


def call_with_schema(prompt: str, schema: dict, max_tokens: int = 500) -> str:
    """Calls the LLM with explicit schema instructions in the prompt."""
    full_prompt = f"""{prompt}

Respond ONLY with valid JSON matching this exact schema:
{json.dumps(schema, indent=2)}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": full_prompt}],
    )
    return response.content[0].text


def call_with_retry(
    prompt: str,
    model_class: Type[T],
    max_retries: int = 3,
) -> T:
    """
    Calls the LLM and retries up to max_retries times if output fails validation.
    On each retry, the LLM is told exactly what went wrong.

    This is the core pattern behind libraries like guardrails-ai and instructor.
    """
    schema = model_class.model_json_schema()
    messages = [{"role": "user", "content": f"{prompt}\n\nRespond with JSON matching: {json.dumps(schema)}"}]

    for attempt in range(1, max_retries + 1):
        response = client.messages.create(
            model=MODEL, max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        raw = response.content[0].text

        try:
            result = parse_structured_output(raw, model_class)
            if attempt > 1:
                print(f"  Succeeded on attempt {attempt}")
            return result

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt == max_retries:
                raise

            # Feed the error back to the LLM so it can self-correct
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user",      "content": (
                    f"Your response failed validation with this error:\n{e}\n\n"
                    f"Please fix your response. Return ONLY valid JSON."
                )},
            ]

    raise RuntimeError("Unreachable")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Output Sanitization
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_output(text: str) -> str:
    """
    Cleans up common LLM output artifacts for display or storage.
    Strips markdown, HTML tags, and normalises whitespace.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove markdown bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove markdown links — keep the display text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Content Policy Check on Output
# ─────────────────────────────────────────────────────────────────────────────

POLICY_VIOLATIONS = [
    (r"\b(kill|murder|assassinate)\b",          "violent language"),
    (r"\b(n-word|racial slur)\b",               "hate speech"),
    (r"\b(password|api.?key|secret.?key)\b",    "credential exposure"),
    (r"\b(http[s]?://(?:phish|malware|hack))",  "malicious URL"),
]


def check_output_policy(text: str) -> list[str]:
    """Returns list of policy violation labels found in the output."""
    lowered = text.lower()
    return [
        label
        for pattern, label in POLICY_VIOLATIONS
        if re.search(pattern, lowered)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ProductReview(BaseModel):
    product_name: str  = Field(description="Name of the product")
    rating:       int  = Field(description="Rating from 1 to 5", ge=1, le=5)
    pros:         list[str] = Field(description="List of 2-3 positive points")
    cons:         list[str] = Field(description="List of 1-2 negative points")
    summary:      str  = Field(description="One-sentence summary", max_length=150)


def get_product_review(product_name: str) -> ProductReview:
    """
    Full output-guardrailed pipeline:
    1. Call LLM with schema instructions
    2. Retry on failure (up to 3 attempts)
    3. Validate with Pydantic
    4. Check content policy
    """
    prompt = f"Write a balanced review for: {product_name}"
    review = call_with_retry(prompt, ProductReview, max_retries=3)

    violations = check_output_policy(review.summary)
    if violations:
        raise ValueError(f"Output policy violated: {violations}")

    return review


# ─────────────────────────────────────────────────────────────────────────────
# Run demos
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═' * 65}")
    print("  DEMO 1 — Raw output problem")
    print(f"{'═' * 65}\n")
    demo_raw_output_problem()

    print(f"\n{'═' * 65}")
    print("  DEMO 2 — Structured output with Pydantic + retry")
    print(f"{'═' * 65}\n")

    review = get_product_review("MacBook Pro M3")
    print(f"  Product : {review.product_name}")
    print(f"  Rating  : {review.rating}/5")
    print(f"  Pros    : {review.pros}")
    print(f"  Cons    : {review.cons}")
    print(f"  Summary : {review.summary}")

    print(f"\n{'═' * 65}")
    print("  DEMO 3 — Output sanitization")
    print(f"{'═' * 65}\n")

    dirty = """### Result
**Key point**: The answer is *really* important.
[Click here](https://example.com) for more.

<b>Bold HTML</b> and   extra   spaces.
"""
    clean = sanitize_output(dirty)
    print(f"  Before: {repr(dirty[:80])}")
    print(f"  After : {repr(clean)}\n")


if __name__ == "__main__":
    main()
