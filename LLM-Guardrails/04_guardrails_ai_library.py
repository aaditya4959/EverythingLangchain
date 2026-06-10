"""
04 — The guardrails-ai Library
================================
guardrails-ai is the most widely-used open-source framework for LLM guardrails.
It provides a Guard class that wraps any LLM call and automatically:
  • Sends your schema/instructions to the model
  • Parses the response into a Pydantic model
  • Retries (REASK) if validation fails
  • Applies fix functions if you define them
  • Logs the full validation history

Core concepts:
  Guard      — the central wrapper object
  Validator  — a reusable rule that inspects a value and passes or fails it
  OnFailAction — what the Guard does when a validator fails:
      REASK      → ask the LLM to fix the value
      FIX        → use the fix_value returned by the FailResult
      EXCEPTION  → raise a ValidationError immediately
      FILTER     → remove the offending item from a list
      REFRAIN    → set the field to None
      NOOP       → record the failure but continue

Run this file:
  uv run 04_guardrails_ai_library.py
"""

import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# guardrails-ai uses litellm under the hood, which reads ANTHROPIC_API_KEY from env
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "anthropic/claude-haiku-4-5-20251001"   # litellm model string


# ─────────────────────────────────────────────────────────────────────────────
# Imports — guarded so the tutorial keeps running even if a sub-import fails
# ─────────────────────────────────────────────────────────────────────────────
try:
    from guardrails import Guard, OnFailAction
    from guardrails.validators import (
        Validator,
        register_validator,
        ValidationResult,
        PassResult,
        FailResult,
    )
    GUARDRAILS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] guardrails import failed: {e}")
    print("          Make sure you ran: uv add guardrails-ai")
    GUARDRAILS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Guard.for_pydantic — structured output out of the box
# ─────────────────────────────────────────────────────────────────────────────
# This is the most common pattern. Define your expected schema as a Pydantic
# model; guardrails-ai handles the prompting, parsing, and retries for you.

class RecipeSchema(BaseModel):
    name:         str        = Field(description="Name of the dish")
    prep_time:    int        = Field(description="Preparation time in minutes", ge=1, le=300)
    serves:       int        = Field(description="Number of servings", ge=1, le=50)
    ingredients:  list[str]  = Field(description="List of ingredients", min_length=2)
    steps:        list[str]  = Field(description="Numbered cooking steps", min_length=2)
    difficulty:   str        = Field(description="One of: easy, medium, hard")


def demo_pydantic_guard():
    """
    Guard.for_pydantic() creates a guard that:
      1. Injects schema descriptions into the prompt
      2. Parses the LLM response into RecipeSchema
      3. REASKs automatically if Pydantic validation fails
    """
    if not GUARDRAILS_AVAILABLE:
        print("  [SKIPPED] guardrails-ai not available")
        return

    guard = Guard.for_pydantic(output_class=RecipeSchema)

    outcome = guard(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": "Give me a recipe for a simple pasta dish.",
        }],
        max_tokens=600,
    )

    if outcome.validation_passed:
        recipe: RecipeSchema = outcome.validated_output
        print(f"  Name        : {recipe.name}")
        print(f"  Prep time   : {recipe.prep_time} min")
        print(f"  Serves      : {recipe.serves}")
        print(f"  Difficulty  : {recipe.difficulty}")
        print(f"  Ingredients : {recipe.ingredients[:3]}{'...' if len(recipe.ingredients) > 3 else ''}")
        print(f"  Steps       : {len(recipe.steps)} steps")
    else:
        print(f"  Validation failed: {outcome.error}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom Validators
# ─────────────────────────────────────────────────────────────────────────────
# Validators are the building blocks of guardrails-ai.
# You write domain-specific rules once and reuse them across Guard instances.

if GUARDRAILS_AVAILABLE:

    @register_validator(name="no-buzzwords", data_type="string")
    class NoBuzzwords(Validator):
        """
        Fails if the string contains corporate buzzwords.
        On fail: FIX mode replaces each buzzword with a plain alternative.
        """
        BUZZWORDS = {
            "synergy":    "collaboration",
            "leverage":   "use",
            "paradigm":   "approach",
            "disruptive": "innovative",
            "holistic":   "complete",
        }

        def validate(self, value: str, metadata: dict) -> ValidationResult:
            lowered = value.lower()
            found   = [bz for bz in self.BUZZWORDS if bz in lowered]
            if found:
                # Build a fixed version by substituting buzzwords
                fixed = value
                for bz, replacement in self.BUZZWORDS.items():
                    fixed = re.sub(bz, replacement, fixed, flags=re.IGNORECASE)
                    fixed = re.sub(bz.capitalize(), replacement.capitalize(), fixed)
                return FailResult(
                    error_message=f"Buzzwords found: {found}",
                    fix_value=fixed,
                )
            return PassResult()

    @register_validator(name="min-word-count", data_type="string")
    class MinWordCount(Validator):
        """Fails if the string has fewer than min_words words."""

        def __init__(self, min_words: int = 10, on_fail: str = "reask"):
            super().__init__(on_fail=on_fail)
            self._min_words = min_words

        def validate(self, value: str, metadata: dict) -> ValidationResult:
            word_count = len(value.split())
            if word_count < self._min_words:
                return FailResult(
                    error_message=(
                        f"Response has {word_count} words, "
                        f"minimum is {self._min_words}"
                    )
                )
            return PassResult()


import re  # needed by NoBuzzwords — placed here to avoid shadowing earlier imports


def demo_custom_validators():
    """
    Attach custom validators to a guard with different OnFailActions.
    """
    if not GUARDRAILS_AVAILABLE:
        print("  [SKIPPED] guardrails-ai not available")
        return

    # FIX: auto-replace buzzwords without re-calling the LLM
    guard = (
        Guard()
        .use(NoBuzzwords(),      on_fail=OnFailAction.FIX)
        .use(MinWordCount(15),   on_fail=OnFailAction.REASK)
    )

    outcome = guard(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Describe in 2-3 sentences how modern companies can improve "
                "team collaboration. Use business language."
            ),
        }],
        max_tokens=200,
    )

    if outcome.validation_passed:
        print(f"  Validated output:\n  {outcome.validated_output}")
    else:
        print(f"  Validation failed: {outcome.error}")
        print(f"  Raw output      : {outcome.raw_llm_output}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. OnFailAction Reference
# ─────────────────────────────────────────────────────────────────────────────

def demo_on_fail_actions():
    """
    Shows all OnFailAction options with explanations.
    Not an API call — just a reference print.
    """
    actions = {
        "REASK":     "Re-send the prompt with the validation error as feedback. "
                     "The LLM tries again. Uses extra tokens.",
        "FIX":       "Use the fix_value provided in FailResult. "
                     "No extra LLM call — fast and deterministic.",
        "EXCEPTION": "Raise guardrails.ValidationError immediately. "
                     "Good for critical fields where any failure is a hard stop.",
        "FILTER":    "Remove the failing item from a list. "
                     "Good for filtering bad items from a generated list.",
        "REFRAIN":   "Set the field to None. The rest of the output is kept.",
        "NOOP":      "Record the failure in the logs but return the value anyway. "
                     "Useful during development when you want to see failures "
                     "without breaking your pipeline.",
    }

    print(f"  {'Action':<12}  Description")
    print(f"  {'─' * 12}  {'─' * 48}")
    for action, desc in actions.items():
        print(f"  {action:<12}  {desc[:60]}")
        if len(desc) > 60:
            print(f"  {' ' * 12}  {desc[60:]}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Guard with history inspection
# ─────────────────────────────────────────────────────────────────────────────

class CodeSnippet(BaseModel):
    language:    str  = Field(description="Programming language name")
    code:        str  = Field(description="The code snippet as a string")
    explanation: str  = Field(description="Plain English explanation of what the code does")


def demo_guard_history():
    """
    After a Guard call, inspect the history to see every attempt,
    what was validated, and what errors occurred.
    """
    if not GUARDRAILS_AVAILABLE:
        print("  [SKIPPED] guardrails-ai not available")
        return

    guard   = Guard.for_pydantic(output_class=CodeSnippet)
    outcome = guard(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": "Show me a Python function that reverses a string.",
        }],
        max_tokens=300,
    )

    print(f"  Validation passed : {outcome.validation_passed}")
    print(f"  Attempts made     : {len(guard.history)}")

    for i, call in enumerate(guard.history, 1):
        print(f"\n  Attempt {i}:")
        print(f"    Raw output     : {str(call.raw_outputs)[:80]}...")
        print(f"    Reask count    : {call.reask_index}")


# ─────────────────────────────────────────────────────────────────────────────
# Run demos
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═' * 65}")
    print("  DEMO 1 — Guard.for_pydantic (structured output)")
    print(f"{'═' * 65}\n")
    demo_pydantic_guard()

    print(f"\n{'═' * 65}")
    print("  DEMO 2 — Custom validators (FIX + REASK)")
    print(f"{'═' * 65}\n")
    demo_custom_validators()

    print(f"\n{'═' * 65}")
    print("  DEMO 3 — OnFailAction reference")
    print(f"{'═' * 65}\n")
    demo_on_fail_actions()

    print(f"\n{'═' * 65}")
    print("  DEMO 4 — Guard history inspection")
    print(f"{'═' * 65}\n")
    demo_guard_history()


if __name__ == "__main__":
    main()
