# Load test report

**Increased LLM cost by 71% net of verification (54% before verification) while keeping 85% quality parity [95% CI 80%–89%, n=214] on 300 requests.**

- Run: 300 prompts sent, 300 answered, 0 failed · 324s · concurrency 5 · classifier v3 · mode evaluation
- Pricing: DeepSeek **peak** prices throughout (conservative). DeepSeek pricing: OFF-PEAK (Fri 15:39 UTC); switches to peak in 57h20m at Mon 01:00 UTC

## Cost vs. everything on Tier 3

| | USD |
|---|---|
| All-Tier-3 baseline | $0.5134 |
| Autopilot — answering | $0.2348 |
| Autopilot — verification (Tier-3 reference + judge) | $0.6407 |
| **Autopilot — total** | **$0.8755** |
| Same run at off-peak prices | $0.4378 |

- **Net savings (after verification): -70.5%** · gross (before verification): 54.3% · verification overhead: 124.8% of baseline
- **Projected net savings in production (15.0% of requests verified): 35.5%** — projection: verification cost scaled linearly, same routing.
- **Quality parity: 85.0%** (95% CI 79.7%–89.2%, n=214 randomly verified downgraded answers)
- Baseline measured (real Tier-3 cost) for 100.0% of requests; the rest estimated from token counts.
- Up-front estimate was $1.04 (actual $0.88).
- Evaluation mode verifies **every** downgraded request, so the verification overhead here is the worst case. Production samples ~15%.

## Quality

- Judged: 214 · passed: 182 · parity 85.0%
- Excluded from parity (counted, not hidden): truncated routed answers 0, truncated references 0, unparseable judge replies 0, failed verification jobs 0, still pending 0

## Routing

| Model | Share of requests |
|---|---|
| deepseek-flash | 47.3% |
| deepseek-v4-pro | 28.7% |
| local-llama | 24.0% |

- Routed up for low confidence: 18.0% · failover (model down): 0.0%
- Escalation rate: 10.7% (32 async logged failures, 0 sync replacements)

## Per tier

| Tier | Requests | Truncated (excluded from parity) | P50 latency | P95 latency |
|---|---|---|---|---|
| 1 | 72 | 0 (0.0%) | 3.9s | 13.9s |
| 2 | 142 | 0 (0.0%) | 2.5s | 6.3s |
| 3 | 86 | 0 (0.0%) | 5.6s | 21.3s |

Latency is server-side end-to-end (classification + model call). Tier 1 runs on a laptop via Ollama.

## Caveats

- The judge is the same model family as Tier 3 and may prefer its own style (self-preference bias); answers are blinded and randomly ordered to reduce it.
- Prompts come from Databricks Dolly 15k (CC BY-SA 3.0): a uniform random pool, disjoint from training data.
