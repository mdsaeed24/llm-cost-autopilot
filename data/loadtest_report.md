# Load test report

**Increased LLM cost by 64% net of verification (62% before verification) while keeping 78% quality parity [95% CI 75%–81%, n=751] on 1000 requests.**

- Run: 1000 prompts sent, 1000 answered, 0 failed · 1970s · concurrency 5 · classifier v2 · mode evaluation
- Pricing: DeepSeek **peak** prices throughout (conservative). DeepSeek pricing: OFF-PEAK (Fri 12:04 UTC); switches to peak in 60h55m at Mon 01:00 UTC

## Cost vs. everything on Tier 3

| | USD |
|---|---|
| All-Tier-3 baseline | $1.8603 |
| Autopilot — answering | $0.6991 |
| Autopilot — verification (Tier-3 reference + judge) | $2.3587 |
| **Autopilot — total** | **$3.0578** |
| Same run at off-peak prices | $1.5289 |

- **Net savings (after verification): -64.4%** · gross (before verification): 62.4% · verification overhead: 126.8% of baseline
- **Projected net savings in production (15.0% of requests verified): 43.4%** — projection: verification cost scaled linearly, same routing.
- **Quality parity: 77.8%** (95% CI 74.7%–80.6%, n=751 randomly verified downgraded answers)
- Baseline measured (real Tier-3 cost) for 100.0% of requests; the rest estimated from token counts.
- Up-front estimate was $3.44 (actual $3.06).
- Evaluation mode verifies **every** downgraded request, so the verification overhead here is the worst case. Production samples ~15%.

## Quality

- Judged: 751 · passed: 584 · parity 77.8%
- Excluded from parity (counted, not hidden): truncated routed answers 0, truncated references 0, unparseable judge replies 0, failed verification jobs 0, still pending 0

## Routing

| Model | Share of requests |
|---|---|
| local-llama | 46.0% |
| deepseek-flash | 29.1% |
| deepseek-v4-pro | 24.9% |

- Routed up for low confidence: 20.4% · failover (model down): 0.0%
- Escalation rate: 16.7% (167 async logged failures, 0 sync replacements)

## Per tier

| Tier | Requests | Truncated (excluded from parity) | P50 latency | P95 latency |
|---|---|---|---|---|
| 1 | 460 | 0 (0.0%) | 11.7s | 28.7s |
| 2 | 291 | 0 (0.0%) | 1.5s | 6.2s |
| 3 | 249 | 0 (0.0%) | 11.5s | 27.1s |

Latency is server-side end-to-end (classification + model call). Tier 1 runs on a laptop via Ollama.

## Caveats

- The judge is the same model family as Tier 3 and may prefer its own style (self-preference bias); answers are blinded and randomly ordered to reduce it.
- Prompts come from Databricks Dolly 15k (CC BY-SA 3.0): a uniform random pool, disjoint from training data.
