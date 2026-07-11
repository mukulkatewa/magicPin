"""
compose() — core message engine for Vera.
OpenRouter LLM at temperature=0 for determinism.
"""

import json
import os
from openai import OpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-3-haiku"

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OPENROUTER_BASE,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
    return _client


SYSTEM_PROMPT = """You are Vera, magicpin's WhatsApp AI for merchant growth in India.
Compose ONE sharp message. Use DERIVED_FACTS numbers directly — they are pre-verified.

Pick ONE engagement formula:
A) PRE-LOAD: "Maine [X] ready kar diya — sirf YES bolna hai."
B) LOSS ANCHOR: "Aap [specific number] [thing] miss kar rahe hain — [fix in N min]?"
C) CURIOSITY GAP: "Want to see [specific thing only reply reveals]?"
D) BINARY COMMIT: "Reply YES to [action] / STOP to skip." (add time-bound if possible)

Category voice — use ≥2 words from the list:
• dentists → fluoride recall, caries recurrence, high-risk cohort, DCI, Dr.
• salons → balayage, keratin, bridal trial, retention, same-day slot, footfall
• restaurants → covers, AOV, footfall, delivery radius, Swiggy/Zomato, dine-in
• gyms → trial-to-paid, churn, HIIT, member retention, September wave
• pharmacies → chronic-Rx, batch, molecule, dispensed, compliance, refill
Taboos: dentists→cure/guaranteed/best price; pharmacies→alarming language

Trigger data to cite:
• research_digest → cited_study source + trial_n + delta% + link to merchant's patient segment
• perf_dip → exact metric + delta_pct + baseline → loss aversion
• perf_spike → metric + delta% + likely_driver → capitalize before it fades
• renewal_due → days_remaining + plan name + what pauses on expiry
• supply_alert → batch numbers + molecule + affected_chronic_patients count
• competitor_opened → distance_km + their_offer vs your_offer → how to respond
• festival_upcoming → days_until + specific offer + one execution idea
• recall_due / chronic_refill_due → months_lapsed + available_slots + price
• dormant_with_vera → DO NOT say "N days since last message". Use best signal:
  priority: CTR vs peer gap > lapsed customers > no active offers > stale posts
• milestone_reached → current value + milestone + what next milestone unlocks
• review_theme_emerged → theme + occurrence_count + common_quote snippet
• curious_ask_due → ONE specific guess-question: "Is hafte [specific guess] chal raha?"
• winback_eligible → days_since_expiry + perf_dip_pct + lapsed_customers_since_expiry
• active_planning_intent → deliver the artifact NOW (pricing table, draft copy, plan)
• seasonal_perf_dip → cite peer range (-25 to -35%) + reframe as retention opportunity
• gbp_unverified → estimated_uplift_pct + verification path + time to complete

Rules:
1. Only use numbers from context. Never invent data, citations, or competitor names.
2. ONE CTA in the LAST sentence only. Binary for action triggers; question for info/curiosity.
3. Use owner_first_name always. Never "Hi there" / "Dear Merchant".
4. Hindi-English mix if languages includes "hi". Natural, not forced.
5. No preamble. Cut straight to the point.
6. Service+price: "Haircut @ ₹99" not "10% off".
7. scope=customer → send_as=merchant_on_behalf. Else → vera.
8. Body: ≤100 words merchant-facing, ≤70 words customer-facing.

--- FEW-SHOT EXAMPLES ---

Example 1 — dentist, research_digest (50/50 reference):
Context: Dr. Meera Nair, dentist, Bangalore, 124 high-risk adult patients. Trigger: JIDA Oct 2026 study, 2100-patient trial, fluoride recall cuts caries recurrence 38%.
Output:
{
  "body": "Dr. Meera, JIDA's Oct issue landed. 2,100-patient RCT shows 3-month fluoride recall cuts caries recurrence 38% vs 6-month schedules — directly relevant to your 124 high-risk adult cohort. Maine draft patient-ed WhatsApp ready kar diya — sirf YES bolna hai.",
  "cta": "binary",
  "send_as": "vera",
  "suppression_key": "dentist:research_digest:meera_nair",
  "rationale": "research_digest → JIDA Oct RCT (2100 patients, 38% recurrence reduction) → 124 high-risk patients = clear recall opportunity"
}

Example 2 — pharmacy, supply_alert (50/50 reference):
Context: Ramesh Medical, Mumbai, 340 chronic-Rx patients. Trigger: Metformin 500mg batch MH-2024-089 recall, 2 batches affected.
Output:
{
  "body": "Ramesh bhai, urgent: ~61 of your 340 chronic-Rx patients likely dispensed Metformin 500mg batch MH-2024-089. Compliance risk + potential dispensing liability. Maine affected-patient list ready kar diya — reply YES, main aapko 10 min mein bhejta hoon.",
  "cta": "binary",
  "send_as": "vera",
  "suppression_key": "pharmacy:supply_alert:ramesh_medical",
  "rationale": "supply_alert → Metformin batch recall → 61 affected chronic-Rx patients = urgent compliance action"
}

--- END EXAMPLES ---

Return ONLY valid JSON, no markdown:
{
  "body": "...",
  "cta": "binary" | "open_ended" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "category:trigger_kind:merchant_short_id",
  "rationale": "trigger_kind → specific signal used → why this merchant now"
}"""


def _derive_facts(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    """Pre-compute derived numbers so the LLM cites them directly without arithmetic."""
    facts = {}
    perf = merchant.get("performance", {})
    cust_agg = merchant.get("customer_aggregate", {})
    peer = category.get("peer_stats", {})

    # CTR gap
    ctr = perf.get("ctr")
    peer_ctr = peer.get("avg_ctr")
    if ctr and peer_ctr:
        gap_pct = round(abs(peer_ctr - ctr) / peer_ctr * 100, 1)
        if ctr < peer_ctr:
            facts["ctr_gap"] = f"CTR {ctr*100:.1f}% — peer avg {peer_ctr*100:.1f}% — you are {gap_pct}% below"
        else:
            facts["ctr_gap"] = f"CTR {ctr*100:.1f}% — {gap_pct}% above peer avg {peer_ctr*100:.1f}%"

    # Lapsed customers
    lapsed = cust_agg.get("lapsed_180d_plus") or cust_agg.get("lapsed_90d_plus")
    if lapsed:
        facts["lapsed_customers"] = f"{lapsed} customers lapsed (not visited in 90-180+ days)"

    # Active member churn risk for gyms
    if cust_agg.get("total_active_members"):
        churn = cust_agg.get("monthly_churn_pct", 0)
        at_risk = round(cust_agg["total_active_members"] * churn)
        facts["members_at_risk_monthly"] = f"{at_risk} of {cust_agg['total_active_members']} members at churn risk monthly"

    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})

    # Supply alert: estimate affected patients
    if trg_kind == "supply_alert":
        chronic = cust_agg.get("chronic_rx_count", 0)
        batches = trg_payload.get("affected_batches", [])
        estimated = min(round(chronic * 0.09 * len(batches)), chronic)
        facts["affected_chronic_patients"] = f"~{estimated} of your {chronic} chronic-Rx patients likely dispensed affected batch(es)"

    # Perf delta for dip/spike
    delta_7d = perf.get("delta_7d", {})
    if trg_kind in ("perf_dip", "perf_spike", "seasonal_perf_dip"):
        metric = trg_payload.get("metric", "calls")
        delta = trg_payload.get("delta_pct") or delta_7d.get(f"{metric}_pct")
        baseline = trg_payload.get("vs_baseline") or perf.get(metric)
        if delta and baseline:
            direction = "up" if delta > 0 else "down"
            facts["perf_signal"] = f"{metric} {direction} {abs(delta)*100:.0f}% this week (baseline: {baseline})"

    # Renewal urgency
    sub = merchant.get("subscription", {})
    days_rem = sub.get("days_remaining") or trg_payload.get("days_remaining")
    if trg_kind == "renewal_due" and days_rem:
        amount = trg_payload.get("renewal_amount")
        facts["renewal_urgency"] = f"{days_rem} days left on {sub.get('plan','Pro')} plan" + (f" — renewal ₹{amount}" if amount else "")

    # Customer recall
    if customer and trg_kind in ("recall_due", "chronic_refill_due"):
        rel = customer.get("relationship", {})
        last_visit = rel.get("last_visit") or trg_payload.get("last_refill")
        slots = trg_payload.get("available_slots", [])
        slot_labels = [s.get("label") for s in slots[:2] if s.get("label")]
        if last_visit:
            facts["recall_info"] = f"Last visit: {last_visit}" + (f" | Open slots: {', '.join(slot_labels)}" if slot_labels else "")

    # Research digest: pre-lookup the cited study so LLM can quote exact source
    if trg_kind == "research_digest":
        top_item_id = trg_payload.get("top_item_id") or trg_payload.get("item_id")
        digest_items = category.get("digest", [])
        if top_item_id:
            match = next((d for d in digest_items if d.get("id") == top_item_id), None)
        else:
            match = digest_items[0] if digest_items else None
        if match:
            parts = []
            if match.get("source"):
                parts.append(f"Source: {match['source']}")
            if match.get("trial_n"):
                parts.append(f"n={match['trial_n']}")
            if match.get("delta_pct") is not None:
                parts.append(f"delta={match['delta_pct']}%")
            if match.get("title"):
                parts.append(f"Title: {match['title']}")
            if match.get("link"):
                parts.append(f"Link: {match['link']}")
            if parts:
                facts["cited_study"] = " | ".join(parts)

    return facts


def _extract_context(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> str:
    voice = category.get("voice", {})
    peer = category.get("peer_stats", {})
    digest = category.get("digest", [])[:2]
    seasonal = category.get("seasonal_beats", [])[:2]
    cat_offers = category.get("offer_catalog", [])[:3]
    trend_signals = category.get("trend_signals", [])[:2]

    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    history = merchant.get("conversation_history", [])[-2:]

    derived = _derive_facts(category, merchant, trigger, customer)

    customer_block = None
    if customer:
        customer_block = {
            "name": customer.get("identity", {}).get("name"),
            "language_pref": customer.get("identity", {}).get("language_pref"),
            "state": customer.get("state"),
            "relationship": customer.get("relationship", {}),
            "preferred_slots": customer.get("preferences", {}).get("preferred_slots"),
            "consent_scope": customer.get("consent", {}).get("scope", []),
        }

    ctx = {
        "DERIVED_FACTS": derived,
        "category": {
            "slug": category.get("slug"),
            "voice_tone": voice.get("tone"),
            "taboos": voice.get("taboos", [])[:3],
            "peer_stats": peer,
            "top_digest": digest,
            "seasonal_beats": seasonal,
            "offer_catalog": cat_offers,
            "trend_signals": trend_signals,
        },
        "merchant": {
            "id": merchant.get("merchant_id"),
            "name": identity.get("name"),
            "owner": identity.get("owner_first_name"),
            "city": identity.get("city"),
            "locality": identity.get("locality"),
            "languages": identity.get("languages", ["en"]),
            "verified": identity.get("verified"),
            "subscription": merchant.get("subscription", {}),
            "perf_30d": {
                "views": perf.get("views"),
                "calls": perf.get("calls"),
                "ctr": perf.get("ctr"),
                "leads": perf.get("leads"),
                "delta_7d": perf.get("delta_7d", {}),
            },
            "active_offers": active_offers,
            "signals": merchant.get("signals", []),
            "customer_aggregate": merchant.get("customer_aggregate", {}),
            "review_themes": merchant.get("review_themes", [])[:2],
            "recent_conversation": history,
        },
        "trigger": {
            "kind": trigger.get("kind"),
            "scope": trigger.get("scope"),
            "urgency": trigger.get("urgency"),
            "payload": trigger.get("payload", {}),
            "suppression_key": trigger.get("suppression_key"),
        },
        "customer": customer_block,
    }

    return (
        json.dumps(ctx, ensure_ascii=False, indent=2)
        + "\n\nCompose the message. Cite DERIVED_FACTS numbers directly. Every number must come from context above."
    )


def _parse_json(raw: str) -> dict:
    """Robust JSON parser — handles markdown fences and control characters."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(raw, strict=False)


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """Deterministic compose at temperature=0. Returns {body, cta, send_as, suppression_key, rationale}."""
    client = _get_client()

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=600,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _extract_context(category, merchant, trigger, customer)},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty content")

    result = _parse_json(content)

    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get(
            "suppression_key", f"trigger:{trigger.get('id', 'unknown')}"
        )
    return result
