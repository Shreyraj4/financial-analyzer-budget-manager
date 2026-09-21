# The LLM explanation layer

Code: `backend/app/agent/` (facts, narrators, verify, service, schemas, check),
endpoints in `backend/app/api/reports.py`, storage in the `agent_reports` table.
Setup: `docs/anthropic_setup.md`.

## The problem

The ML layer produces numbers and flags. A user wants a readable explanation.
LLMs write well but can hallucinate figures, so the design goal is:
**Claude may explain, but must not decide any number, and we must be able to prove it.**

## Architecture

```
ML outputs (profile, forecast+budgets, anomalies, spend trends)
        |
   facts.py  (pure code)  ->  numbered facts: "budget.food_dining: Food & Dining: forecast Rs 9,500; ..."
        |
   narrator  (Claude with structured output, or the no-LLM template)
        |
   verify.py: every number in each insight must appear in the facts it cites
        |         fail -> retry once with precise feedback -> drop what still fails
        v
   agent_reports table (report + facts + verification details)  ->  API -> dashboard
```

## Key design decisions (explain these in an interview)

1. **Facts layer.** All figures, including derived ones (percent change, savings),
   are computed in Python and written into fact sentences. The LLM only *restates*.
2. **Structured output.** The report is a Pydantic schema (headline, summary,
   insights with severity and cited fact ids, suggested actions). `client.messages.parse`
   returns a validated object, so we never parse free text.
3. **Grounding via citations.** Each insight/action cites fact ids. Numbers are checked
   against the *cited* facts only, so "the right number attached to the wrong claim" is caught too.
4. **Verification + retry + drop.** Mismatch -> one retry that shows the model exactly what failed ->
   if still wrong, the unverifiable insight is removed; a bad headline/summary is replaced by the
   template wording.
5. **Graceful degradation.** No key, API error, refusal or truncation -> the template narrator
   builds the same report from the facts. The app never returns a 500 because of the LLM.
6. **Prompt-injection defence.** Merchant names in bank statements are attacker-controllable text.
   They are sanitized (printable characters, one line, 60 chars, quotes removed), placed inside a
   JSON data payload, and the system prompt says text inside facts is data, never instructions.
   Even if an injection slipped through, verification limits what numbers can appear, and the
   output schema limits the shape.
7. **Cost control.** Reports are generated on an explicit `POST`, not on every page view; token
   usage is stored with each report.
8. **Auditability.** Each stored report keeps the facts it was allowed to use and the verification
   record, so any sentence can be traced back to the model output that produced it.

## Multiple providers (Anthropic and OpenRouter)

The narrator is an interface (`narrate(facts, previous, feedback)`), so providers are swappable:
`ClaudeNarrator` (Anthropic SDK, schema-constrained output), `OpenRouterNarrator` (plain HTTPS to
OpenRouter's chat API, prompt-for-JSON then strict Pydantic validation), and `TemplateNarrator`
(no LLM). Selection lives in one function (`narrator_choice`) and never changes the pipeline.
Because free models are weaker, the safety net around the LLM (verification, retry, drop, fallback)
matters more than which model is used: **the design assumes the LLM is unreliable and makes that safe.**

Trade-offs to mention: schema-constrained output guarantees shape but not every provider supports it;
prompting for JSON needs tolerant parsing (fences, prose, reasoning blocks) plus a repair round;
free tiers are rate limited and may log prompts (privacy).

## Concepts to be able to explain

- **Grounded generation / faithfulness** vs hallucination.
- **Structured outputs** (schema-constrained generation) and why they beat regex-parsing text.
- **Prompt injection** (direct vs indirect; here it would be *indirect*, via data), and defence in depth.
- **Tool/LLM as narrator, not calculator**: keep deterministic logic out of the model.
- **Fallbacks and graceful degradation** in LLM applications.
- **Temperature/effort/thinking** exist but were left at defaults; determinism comes from verification, not sampling.

## Honest limitations (say these first)

1. **The verifier checks numbers, not meaning.** Claude could say "spending fell" when the fact says
   it rose, using correct figures. Mitigations: severity/wording constraints in the prompt, showing
   cited facts next to each insight in the UI. A proper faithfulness evaluation (human or LLM-judge
   over a sample of reports) would be the next step.
2. **Rounding tolerance** is deliberately tight (about half a rupee, 0.5%), so occasional legitimate
   rewording triggers a retry.
3. **I could not test against the live API** during development (no key yet): tests use a fake
   client with scripted responses, so real-API behaviour (latency, exact wording, refusal rates)
   is unverified until a key is added.
4. **Words for numbers** ("about a quarter") bypass the number check; the prompt asks the model to avoid this.
5. **Two LLM-related risks remain**: the system trusts the model with tone and prioritisation, and
   sends anonymized-but-real spending data to a third-party API (privacy note for the report).
6. Refusal fallbacks to another model were not enabled; refusals fall back to the template report instead.

## Interview questions

**Q: How do you stop the LLM from making up numbers?**
A: It never computes numbers. Code writes facts with every figure; the model must cite fact ids;
a verifier checks every number against the cited facts, retries once, then drops failures.

**Q: What if the user uploads a statement with a malicious merchant name?**
A: It is sanitized and truncated, sent as JSON data, the prompt marks it untrusted, and the
verifier plus output schema bound the damage.

**Q: Why structured output instead of asking for JSON in the prompt?**
A: The API constrains generation to the schema, and the SDK returns a validated object, so no
brittle parsing and no malformed responses.

**Q: What happens without internet or without a key?**
A: A deterministic template narrator produces the same report shape from the facts.

**Q: How would you evaluate the explanations?**
A: Number-faithfulness is checked automatically on every report. For semantic faithfulness and
usefulness, sample reports and score them with human raters or an LLM judge against the facts.

**Q: Why is this an "agent"?**
A: It's a workflow, not an autonomous agent: a fixed pipeline where Claude does one bounded task.
I chose the simplest architecture that meets the need; tool-calling agents add cost and risk
without benefit for a report that needs no open-ended exploration.

## Reading

- Anthropic, "Building effective agents" (workflows vs agents): https://www.anthropic.com/research/building-effective-agents
- Anthropic API docs - Messages API and structured outputs: https://docs.anthropic.com/
- OWASP Top 10 for LLM Applications (LLM01 Prompt Injection, LLM09 Overreliance): https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Pydantic docs (models and validation): https://docs.pydantic.dev/
- Simon Willison, "Prompt injection" series (search his blog) - clear explanations of indirect injection
