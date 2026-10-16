"""Public demo page (Streamlit Community Cloud): real load-test results + an interactive router.

Read-only by design: no LLM is called and no API key is needed, so the public page can't
spend money or leak a key. The "Try the router" tab runs the trained classifier locally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from autopilot.classifier.registry import DEFAULT_MODELS_DIR, load_classifier
from autopilot.dashboard import charts
from autopilot.dashboard.app import render_dashboard
from autopilot.llm.catalog import ModelCatalog, load_catalog
from autopilot.routing.config import load_routing_config
from autopilot.routing.router import Router, RoutingDecision
from autopilot.settings import PROJECT_ROOT

REPO_URL = "https://github.com/mdsaeed24/llm-cost-autopilot"
DATA = PROJECT_ROOT / "data"
RUNS = {
    "Run 2 — classifier v3, 300 prompts (current)": PROJECT_ROOT / "demo" / "run2_v3_300.db",
    "Run 1 — classifier v2, 1,000 prompts (before the fix)": PROJECT_ROOT / "demo" / "run1_v2_1000.db",
}
ASSUMED_OUTPUT_TOKENS = 400
EXAMPLES = {
    "Extract from text": "Extract the names of the cities mentioned.\n\nContext:\nAfter Lisbon, the team flew to Porto and then took the train to Madrid.",
    "Fact from memory": "Who directed the most episodes of season five of Game of Thrones?",
    "Classify": "Classify each of these as a fruit or a vegetable: apple, carrot, banana, broccoli.",
    "Reasoning puzzle": "Pens cost $3 each or $12 for a pack of 5. Sam needs exactly 23 pens. What is the minimum he can spend? Explain.",
    "Creative": "Write a 6-line poem about a lighthouse keeper's last night on the job.",
}


# ---------- data (pure, testable) ----------

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def headline(report: dict[str, Any]) -> str:
    """Same wording as the README headline, from the current load-test report."""
    c, q, run = report["costs"], report["quality"], report["run"]
    return (
        f"Cut LLM cost by {c['gross_savings_pct']:.0%} vs. sending everything to the top model "
        f"({c['projection']['net_savings_pct']:.0%} net after verifying a random "
        f"{c['projection']['sample_rate']:.0%} of answers), while keeping {q['parity']:.0%} quality parity "
        f"(95% CI {q['ci_low']:.0%}–{q['ci_high']:.0%}) on {run['ok']:,} requests."
    )


def estimate_costs(prompt: str, decision: RoutingDecision, catalog: ModelCatalog, top_model: str) -> tuple[float, float]:
    """(cost on the routed model, cost on Tier 3) for one request, assuming a typical answer length."""
    input_tokens = len(prompt) / 4 + 10

    def cost(model_name: str) -> float:
        p = catalog.get(model_name).pricing
        return (input_tokens * p.input_cache_miss_per_1m + ASSUMED_OUTPUT_TOKENS * p.output_per_1m) / 1_000_000

    return cost(decision.model_name), cost(top_model)


def comparison_table(comparison: dict[str, Any]) -> pd.DataFrame:
    """The paired v2-vs-v3 numbers as a display table."""
    v2, v3 = comparison["v2"], comparison["v3"]

    def ci(run: dict[str, Any]) -> str:
        return f"{run['parity']:.1%} ({run['ci'][0]:.0%}–{run['ci'][1]:.0%})"

    rows = [
        ("Sent to local llama 3B", f"{v2['routing_share']['local-llama']:.0%}", f"{v3['routing_share']['local-llama']:.0%}"),
        ("Quality parity (95% CI)", ci(v2), ci(v3)),
        ("Routing failures found", str(v2["failures"]), str(v3["failures"])),
        ("Gross savings", f"{v2['gross_savings']:.1%}", f"{v3['gross_savings']:.1%}"),
        ("Projected net savings (15% verified)", f"{v2['projected_net_15']:.1%}", f"{v3['projected_net_15']:.1%}"),
    ]
    return pd.DataFrame(rows, columns=["Same 300 prompts", "v2 (before)", "v3 (after)"])


# ---------- page ----------

@st.cache_resource(show_spinner="Loading the trained classifier…")
def get_router() -> Router:
    catalog = load_catalog()
    routing = load_routing_config(catalog)
    return Router(load_classifier(DEFAULT_MODELS_DIR, routing.classifier_version), routing, catalog)


def tab_router(comparison: dict[str, Any]) -> None:
    st.markdown("Type any prompt. The **trained classifier** decides which model should answer it. "
                "*Nothing is sent to an LLM here — this is the routing decision only.*")
    choice = st.radio("Examples", list(EXAMPLES), horizontal=True, index=1)
    prompt = st.text_area("Prompt", value=EXAMPLES[choice], height=120)
    if not prompt.strip():
        st.info("Type a prompt to see the routing decision.")
        return
    router = get_router()
    decision = router.decide(prompt)
    top = router.config.tiers[3]

    cols = st.columns(4)
    cols[0].metric("Predicted tier", f"Tier {decision.predicted_tier}")
    cols[1].metric("Confidence", f"{decision.confidence:.0%}" if decision.confidence is not None else "—")
    cols[2].metric("Routed to", decision.model_name, delta="routed up (low confidence)" if decision.routed_up else None,
                   delta_color="off", delta_arrow="off")
    cols[3].metric("Task type", (decision.task_type or "—").replace("_", " "))
    st.caption(f"Why: {decision.reason}")

    probs = pd.DataFrame({"tier": [f"Tier {t}" for t in sorted(decision.probabilities)],
                          "probability": [decision.probabilities[t] for t in sorted(decision.probabilities)]})
    p = charts.palette_for(st.context.theme.type)
    bars = alt.Chart(probs).mark_bar(color=p.series_1, cornerRadiusEnd=4, size=22).encode(
        y=alt.Y("tier:N", title=None),
        x=alt.X("probability:Q", title="Classifier probability", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("tier:N"), alt.Tooltip("probability:Q", format=".1%")],
    )
    labels = bars.mark_text(align="left", dx=6, color=p.ink).encode(text=alt.Text("probability:Q", format=".0%"))
    st.altair_chart(charts._style(bars + labels, p, height=140), width="stretch", theme=None)

    routed_cost, top_cost = estimate_costs(prompt, decision, router.catalog, top)
    saving = 1 - routed_cost / top_cost if top_cost else 0.0
    tier_parity = comparison["v3"]["parity_by_tier"].get(str(decision.routed_tier), {}).get("parity")
    lines = [f"**Estimated cost** (≈{ASSUMED_OUTPUT_TOKENS}-token answer, peak prices): "
             f"${routed_cost * 1000:.2f} per 1,000 such requests on `{decision.model_name}` vs "
             f"${top_cost * 1000:.2f} on `{top}` — **{saving:.0%} cheaper**."]
    if tier_parity is not None:
        lines.append(f"**Measured quality** for this tier in the load test: answers matched the top model "
                     f"**{tier_parity:.0%}** of the time.")
    elif decision.routed_tier == 3:
        lines.append("Tier 3 *is* the reference model, so there is nothing cheaper to compare against.")
    st.markdown("  \n".join(lines))


def tab_dashboard() -> None:
    run = st.selectbox("Load test", list(RUNS))
    st.caption("Every downgraded answer was verified in these runs, so net savings are negative by design; "
               "the headline projects them to the production rate of 15%. Costs at DeepSeek peak prices.")
    render_dashboard(RUNS[run])


def tab_before_after(comparison: dict[str, Any]) -> None:
    st.markdown(
        "Run 1 showed the local 3B model failing on **fact questions answered from memory** (61% parity). "
        "The labeling rule was changed to send those to DeepSeek Flash, only the 84 affected labels were "
        "re-checked, and classifier v3 was trained. Same 300 prompts, before and after:"
    )
    st.dataframe(comparison_table(comparison), hide_index=True, width="stretch")
    st.markdown(
        "- About **12 points of savings bought about 8 points of quality** — the trade-off is now measurable.\n"
        "- The projected ~93% parity was too optimistic: llama's parity on what it still receives stayed at "
        "~65%. The remaining gap is the 3B model's reading comprehension, not the routing policy.\n"
        "- Retraining with the 167 real failures caught 100% of hard prompts but lost one test prompt of "
        "accuracy, so the *not-worse* promotion rule kept v3."
    )


def tab_how() -> None:
    st.markdown(
        """
1. **Classify** the request (TF-IDF + hand-crafted features, random forest; trained on hand-labeled prompts).
2. **Route** to the cheapest capable model — local llama 3B ($0), DeepSeek Flash, or DeepSeek V4 Pro.
   If the classifier is unsure, **route one tier up**.
3. **Verify a random 15%** in the background: the top model answers the same prompt and a **blinded
   LLM judge** compares the two answers.
4. **Learn**: each confirmed failure becomes a training example one tier up; a retrained model only goes
   live if it is **not worse** on a frozen test set.
5. **Report honestly**: savings are never shown without quality parity, its confidence interval, and the
   cost of verification.
"""
    )
    st.markdown(f"Code, tests, architecture decisions and full write-up: **[GitHub]({REPO_URL})**")


def main() -> None:
    st.set_page_config(page_title="LLM Cost Autopilot — live demo", page_icon="🧭", layout="wide")
    report = load_json(DATA / "loadtest_report_v3.json")
    comparison = load_json(DATA / "v2_vs_v3_comparison.json")

    st.title("🧭 LLM Cost Autopilot")
    st.markdown("A routing layer that sends each LLM request to the **cheapest model that can handle it**, "
                "checks a random sample against the top model, and learns from its mistakes.")
    st.success(f"**{headline(report)}**")
    st.link_button("View the code on GitHub", REPO_URL)
    st.caption("Public demo: a read-only snapshot of real load tests (Databricks Dolly 15k prompts, CC BY-SA 3.0). "
               "No LLM is called on this page.")

    router_tab, dashboard_tab, change_tab, how_tab = st.tabs(
        ["🧭 Try the router", "📊 Load-test dashboard", "🔁 Before vs after the fix", "🏗 How it works"]
    )
    with router_tab:
        tab_router(comparison)
    with dashboard_tab:
        tab_dashboard()
    with change_tab:
        tab_before_after(comparison)
    with how_tab:
        tab_how()
