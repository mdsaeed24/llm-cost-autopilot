"""Ten hand-written prompts spanning the three complexity tiers, for the Phase 1 baseline.

``expected`` is a short string the correct answer must contain. It is a crude sanity check
for building intuition, not a quality metric (that comes from the LLM judge in Phase 3).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SmokePrompt(BaseModel):
    """One baseline prompt and the tier we'd intuitively expect it to need."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    kind: str
    intuitive_tier: int
    prompt: str
    expected: str | None = None


SMOKE_PROMPTS: tuple[SmokePrompt, ...] = (
    SmokePrompt(
        prompt_id="p01",
        kind="extraction",
        intuitive_tier=1,
        prompt=(
            "Extract the person's name, company, email and meeting date from this text and return "
            "ONLY a JSON object with keys name, company, email, date.\n\n"
            "Text: Hi team, Priya Raman from Northwind Logistics reached out. Her email is "
            "priya.raman@northwind.example and she'd like to meet on 12 March 2026 to discuss renewal."
        ),
        expected="priya.raman@northwind.example",
    ),
    SmokePrompt(
        prompt_id="p02",
        kind="reformatting",
        intuitive_tier=1,
        prompt=(
            "Convert this list into a Markdown table with columns Fruit, Colour, Price:\n"
            "- apple, red, $1.20\n- banana, yellow, $0.50\n- grape, purple, $3.00"
        ),
        expected="| apple",
    ),
    SmokePrompt(
        prompt_id="p03",
        kind="qa_from_context",
        intuitive_tier=1,
        prompt=(
            "Context: The Eiffel Tower was completed in 1889 for the World's Fair. It is 330 metres "
            "tall and was the tallest man-made structure in the world until 1930.\n\n"
            "Question: In which year did the Eiffel Tower stop being the tallest man-made structure?"
        ),
        expected="1930",
    ),
    SmokePrompt(
        prompt_id="p04",
        kind="summarization",
        intuitive_tier=2,
        prompt=(
            "Summarise in exactly two sentences:\n\n"
            "Remote work adoption surged in 2020 as offices closed. Companies discovered that many "
            "roles could be done from home without productivity loss, though collaboration and "
            "onboarding suffered. By 2023 most firms settled on hybrid policies, with employees in "
            "the office two or three days a week. Commercial real estate in city centres has not "
            "recovered, and some cities are converting empty offices into housing."
        ),
    ),
    SmokePrompt(
        prompt_id="p05",
        kind="classification",
        intuitive_tier=2,
        prompt=(
            "Classify this support ticket. Reply with JSON: {\"category\": one of [billing, bug, "
            "feature_request, account], \"sentiment\": one of [positive, neutral, negative]}.\n\n"
            "Ticket: I was charged twice for my subscription this month and nobody has replied to "
            "my last two emails. This is really frustrating."
        ),
        expected="billing",
    ),
    SmokePrompt(
        prompt_id="p06",
        kind="analysis",
        intuitive_tier=2,
        prompt=(
            "A 5-person startup is choosing between Plan A ($50/user/month, unlimited storage) and "
            "Plan B ($200/month flat for up to 10 users, 1 TB storage). They expect to grow to 12 "
            "people in a year. Compare the plans in a short structured analysis and recommend one."
        ),
    ),
    SmokePrompt(
        prompt_id="p07",
        kind="reasoning",
        intuitive_tier=3,
        prompt=(
            "Pens cost $3 each, or $12 for a pack of 5. Sam needs exactly 23 pens and may buy more "
            "if that is cheaper. What is the minimum he can spend? Show your steps and end with "
            "'Answer: $X'."
        ),
        expected="57",
    ),
    SmokePrompt(
        prompt_id="p08",
        kind="reasoning",
        intuitive_tier=3,
        prompt=(
            "Three boxes are labelled 'Apples', 'Oranges' and 'Apples and Oranges'. Every label is "
            "wrong. You may take out one fruit from one box without looking inside. Which box should "
            "you pick from, and how do you then label all three boxes correctly? Explain step by step."
        ),
    ),
    SmokePrompt(
        prompt_id="p09",
        kind="creative",
        intuitive_tier=3,
        prompt=(
            "Write a 6-line poem about a lighthouse keeper's last night on the job. Every line must "
            "end with a word that rhymes with 'light', and the word 'sea' must not appear."
        ),
    ),
    SmokePrompt(
        prompt_id="p10",
        kind="nuanced_judgment",
        intuitive_tier=3,
        prompt=(
            "A junior engineer discovers that a senior colleague's code, already in production, "
            "logs customers' plaintext passwords. The senior colleague is their mentor and is up for "
            "promotion next week. What should the junior engineer do, and why? Weigh the competing "
            "considerations before giving a recommendation."
        ),
    ),
)
