"""
compose() — the core message engine for Vera.
Calls an LLM via OpenRouter at temperature=0 for deterministic output.
"""

import json
import os
from openai import OpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-3-haiku"  # fast + cheap on OpenRouter

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OPENROUTER_BASE,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
    return _client


SYSTEM_PROMPT = """You are Vera, magicpin's WhatsApp AI assistant for merchant growth in India.

Given 4 context layers, compose ONE sharp WhatsApp message that makes the merchant want to reply immediately.

═══ SPECIFICITY RULE (most important for scoring) ═══
Every message MUST contain at least 2 concrete verifiable facts pulled directly from the context:
- Real numbers: CTR%, views, calls, member count, price (₹), lapsed count, delta%
- Real dates: slot times, renewal dates, expiry dates
- Real citations: journal name + page, batch numbers, molecule names
- Real comparisons: "your CTR 2.1% vs peer avg 3.0%" is gold

NEVER use vague phrases: "kaafi time ho gaya", "improve your profile", "boost your sales".
Always prefer: "aapka CTR 2.1% hai — peer avg 3.0% hai, 43% gap hai" or "22 lapsed patients mein se 78 180+ days se nahi aaye".

═══ TRIGGER-SPECIFIC STRATEGY ═══
Read the trigger KIND carefully and anchor the message to that specific event:

• research_digest → cite the study: trial_n + delta% + source (journal p.XX) + link to merchant's patient segment
• perf_dip → cite exact metric: "calls -50% this week vs baseline of 12" — frame as loss aversion
• perf_spike → cite the spike metric + ask merchant to capitalize NOW before it fades
• renewal_due → cite days_remaining + what they lose on expiry (profile maintenance paused)
• supply_alert → cite batch numbers + molecule + count of affected customers (derived from chronic_rx_count)
• competitor_opened → cite distance_km + their price vs your price — frame as response needed
• festival_upcoming → cite days_until + specific offer from catalog + one creative execution idea
• recall_due / chronic_refill_due → cite last_visit date + months_lapsed + available slots + price
• dormant_with_vera → DO NOT just say "N days since last message". Instead pivot to:
  - their strongest underperforming metric (CTR vs peer, lapsed customers, no active offers)
  - or the most urgent signal in merchant.signals[]
  - Make it about their business state, not about Vera being ignored
• milestone_reached → cite the milestone value + what next milestone unlocks
• review_theme_emerged → cite exact theme + occurrence count + common_quote snippet
• curious_ask_due → ask ONE specific question about their most-booked service or recent trend
• winback_eligible → cite days_since_expiry + perf_dip_pct + lapsed_customers added since expiry
• active_planning_intent → deliver the requested artifact immediately (pricing, draft copy, plan)
• seasonal_perf_dip → reassure with peer data range + reframe as opportunity
• gbp_unverified → cite estimated_uplift_pct + simple verification path

═══ HARD RULES ═══
1. Use ONLY facts from provided context. No invented numbers, no fabricated offers, no fake citations.
2. ONE CTA in the LAST sentence only:
   - action triggers (perf_dip, renewal, supply_alert, competitor) → binary: "Reply YES / STOP"
   - info/curiosity triggers (research_digest, curious_ask, milestone) → open-ended question
   - pure data/compliance → no CTA
3. Category voice (violating this = -2 pts):
   - dentists: clinical-peer tone, cite sources, "Dr." prefix, taboo: "cure/guaranteed/best"
   - salons: warm-visual, use treatment names ("balayage", "keratin"), aspirational
   - restaurants: operator-to-operator, use "covers", "AOV", "delivery radius", "footfall"
   - gyms: motivating-coach, no shame, cite member counts and class specifics
   - pharmacies: trustworthy-precise, full molecule names, batch numbers, no alarmism
4. Use owner_first_name always (never "Hi there" or "Dear Merchant").
5. If languages includes "hi" → mix Hindi naturally: "aapka", "hai", "chahiye", "karo" etc.
6. No preamble. No "I hope you're doing well". Cut straight to the point.
7. service+price format beats "X% off": "Haircut @ ₹99" not "10% off haircuts".
8. If trigger scope = "customer" → send_as = "merchant_on_behalf". Else → send_as = "vera".
9. Body: ≤120 words merchant-facing, ≤80 words customer-facing.
10. No re-introduction if conversation_history shows prior messages.

═══ COMPULSION LEVERS — USE 1-2 PER MESSAGE ═══
• Loss aversion: "aap 78 lapsed patients ko miss kar rahe hain" / "before window closes"
• Social proof: "3 dentists in your area ran this campaign this month"
• Effort externalization: "maine draft kar diya — sirf YES bolna hai" / "2 min kaam"
• Curiosity: "want to see which 22 patients need follow-up?" / "kaun competitor khula?"
• Asking the merchant: "is hafte sabse zyada kya poocha customers ne?"
• Binary commit: "Reply YES" — removes all friction

═══ OUTPUT FORMAT ═══
Return ONLY valid JSON, no markdown wrapper:
{
  "body": "the WhatsApp message text",
  "cta": "binary" | "open_ended" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "category:trigger_kind:merchant_id_short",
  "rationale": "trigger_kind → specific signal used → why this merchant now (1 sentence)"
}"""


def _extract_context(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> str:
    """Build a focused context string with all scoring-relevant fields."""

    voice = category.get("voice", {})
    peer = category.get("peer_stats", {})
    digest_items = category.get("digest", [])[:3]
    seasonal = category.get("seasonal_beats", [])[:2]
    offers_catalog = category.get("offer_catalog", [])[:4]
    trend_signals = category.get("trend_signals", [])[:2]

    cat_block = {
        "slug": category.get("slug"),
        "voice": {
            "tone": voice.get("tone"),
            "vocab_allowed": voice.get("vocab_allowed", [])[:6],
            "taboos": voice.get("taboos", [])[:4],
        },
        "peer_stats": peer,
        "top_digest": digest_items,
        "seasonal_beats": seasonal,
        "offer_catalog": offers_catalog,
        "trend_signals": trend_signals,
    }

    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    all_offers = merchant.get("offers", [])
    active_offers = [o for o in all_offers if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    history = merchant.get("conversation_history", [])[-3:]
    cust_agg = merchant.get("customer_aggregate", {})
    review_themes = merchant.get("review_themes", [])[:3]
    subscription = merchant.get("subscription", {})

    # CTR gap is a strong specificity anchor — compute it explicitly
    ctr = perf.get("ctr")
    peer_ctr = peer.get("avg_ctr")
    ctr_gap_note = None
    if ctr and peer_ctr:
        pct_gap = round(((peer_ctr - ctr) / peer_ctr) * 100, 1)
        if pct_gap > 0:
            ctr_gap_note = f"CTR {ctr} is {pct_gap}% BELOW peer avg {peer_ctr} — strong lever"
        else:
            ctr_gap_note = f"CTR {ctr} is {abs(pct_gap)}% ABOVE peer avg {peer_ctr} — highlight strength"

    merchant_block = {
        "merchant_id": merchant.get("merchant_id"),
        "name": identity.get("name"),
        "owner_first_name": identity.get("owner_first_name"),
        "city": identity.get("city"),
        "locality": identity.get("locality"),
        "languages": identity.get("languages", ["en"]),
        "verified_gbp": identity.get("verified"),
        "subscription": subscription,
        "performance_30d": {
            "views": perf.get("views"),
            "calls": perf.get("calls"),
            "directions": perf.get("directions"),
            "ctr": ctr,
            "leads": perf.get("leads"),
            "delta_7d": perf.get("delta_7d", {}),
        },
        "ctr_vs_peer_note": ctr_gap_note,
        "active_offers": active_offers,
        "all_signals": signals,
        "customer_aggregate": cust_agg,
        "review_themes": review_themes,
        "recent_conversation_last3": history,
    }

    trigger_block = {
        "id": trigger.get("id"),
        "kind": trigger.get("kind"),
        "scope": trigger.get("scope"),
        "source": trigger.get("source"),
        "urgency_1to5": trigger.get("urgency"),
        "full_payload": trigger.get("payload", {}),
        "suppression_key": trigger.get("suppression_key"),
        "expires_at": trigger.get("expires_at"),
    }

    customer_block = None
    if customer:
        customer_block = {
            "customer_id": customer.get("customer_id"),
            "name": customer.get("identity", {}).get("name"),
            "language_pref": customer.get("identity", {}).get("language_pref"),
            "state": customer.get("state"),
            "relationship": customer.get("relationship", {}),
            "preferences": customer.get("preferences", {}),
            "consent_scope": customer.get("consent", {}).get("scope", []),
        }

    return f"""CATEGORY:
{json.dumps(cat_block, ensure_ascii=False, indent=2)}

MERCHANT:
{json.dumps(merchant_block, ensure_ascii=False, indent=2)}

TRIGGER (why message NOW):
{json.dumps(trigger_block, ensure_ascii=False, indent=2)}

CUSTOMER:
{json.dumps(customer_block, ensure_ascii=False, indent=2) if customer_block else "null — this is a merchant-facing message"}

TASK: Compose the message. Include at least 2 concrete facts from context above. Every number must come from context — no invention."""


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """
    Core composition function. Deterministic at temperature=0.
    Returns: {body, cta, send_as, suppression_key, rationale}
    """
    client = _get_client()
    user_content = _extract_context(category, merchant, trigger, customer)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=700,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if model wraps output
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    result = json.loads(raw)

    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get(
            "suppression_key", f"trigger:{trigger.get('id', 'unknown')}"
        )

    return result
