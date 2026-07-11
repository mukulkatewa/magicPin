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

Given 4 context layers, compose ONE focused WhatsApp message that will make the merchant (or customer) want to reply.

═══ HARD RULES (judge will penalize violations) ═══
1. ONLY use numbers/facts from the context provided — never invent data, offers, or citations.
2. ONE CTA, placed as the LAST sentence. Binary "Reply YES / STOP" for action triggers. Open-ended question for info/curiosity. No CTA for pure compliance updates.
3. Category voice:
   - dentists: clinical-peer, cite sources, no "cure/guaranteed", no hype
   - salons: warm-visual, aspirational, use treatment names
   - restaurants: operator-to-operator, use "covers/AOV/delivery radius"
   - gyms: motivating-coach, no shame, evidence-based
   - pharmacies: trustworthy-precise, molecule names, no alarmism
4. Use the merchant's owner_first_name. Never "Hi there" or "Dear Merchant".
5. If merchant's languages includes "hi", naturally mix Hindi words (not forced).
6. No preamble. No "I hope you're doing well". No re-introduction after the first message.
7. service+price format ("Haircut @ ₹99") beats "X% off" — always prefer it.
8. If trigger scope is "customer": set send_as = "merchant_on_behalf". Otherwise send_as = "vera".
9. Body under 120 words for merchant-facing. Under 80 words for customer-facing.

═══ COMPULSION LEVERS (use 1-2 per message) ═══
• Specificity: real number + source ("JIDA Oct p.14", "38% better", "22 of your customers")
• Loss aversion: "you're missing X" / "before this window closes"
• Social proof: "3 dentists in your locality did Y this month"
• Effort externalization: "I've drafted X — just say go" / "takes 5 min"
• Curiosity: "want to see who?" / "want the full breakdown?"
• Asking the merchant: "what's been your most-booked service this week?"
• Single binary commit: "Reply YES" removes friction

═══ OUTPUT FORMAT ═══
Return ONLY valid JSON, no markdown:
{
  "body": "the WhatsApp message text",
  "cta": "binary" | "open_ended" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "short:unique:key",
  "rationale": "signal chosen → why this merchant right now (1 sentence)"
}"""


def _extract_context(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> str:
    """Build a focused context string — only the fields that drive message quality."""

    # Category: voice + top digest + peer stats
    voice = category.get("voice", {})
    peer = category.get("peer_stats", {})
    digest_items = category.get("digest", [])[:2]  # top 2 only
    seasonal = category.get("seasonal_beats", [])[:2]
    offers_catalog = category.get("offer_catalog", [])[:3]

    cat_block = {
        "slug": category.get("slug"),
        "voice": {"tone": voice.get("tone"), "taboos": voice.get("taboos", [])[:4]},
        "peer_stats": peer,
        "top_digest": digest_items,
        "seasonal_beats": seasonal,
        "offer_catalog": offers_catalog,
    }

    # Merchant: identity + performance + active offers + signals + last 2 conversation turns
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    history = merchant.get("conversation_history", [])[-2:]
    cust_agg = merchant.get("customer_aggregate", {})
    review_themes = merchant.get("review_themes", [])[:2]

    merchant_block = {
        "merchant_id": merchant.get("merchant_id"),
        "name": identity.get("name"),
        "owner_first_name": identity.get("owner_first_name"),
        "city": identity.get("city"),
        "locality": identity.get("locality"),
        "languages": identity.get("languages", ["en"]),
        "verified": identity.get("verified"),
        "subscription": merchant.get("subscription", {}),
        "performance": {
            "views_30d": perf.get("views"),
            "calls_30d": perf.get("calls"),
            "ctr": perf.get("ctr"),
            "peer_avg_ctr": peer.get("avg_ctr"),
            "delta_7d": perf.get("delta_7d", {}),
        },
        "active_offers": active_offers,
        "signals": signals,
        "customer_aggregate": cust_agg,
        "review_themes": review_themes,
        "recent_conversation": history,
    }

    # Trigger: full payload — this is the "why now"
    trigger_block = {
        "id": trigger.get("id"),
        "kind": trigger.get("kind"),
        "scope": trigger.get("scope"),
        "source": trigger.get("source"),
        "urgency": trigger.get("urgency"),
        "payload": trigger.get("payload", {}),
        "suppression_key": trigger.get("suppression_key"),
        "expires_at": trigger.get("expires_at"),
    }

    # Customer: key fields only
    customer_block = None
    if customer:
        customer_block = {
            "customer_id": customer.get("customer_id"),
            "name": customer.get("identity", {}).get("name"),
            "language_pref": customer.get("identity", {}).get("language_pref"),
            "state": customer.get("state"),
            "relationship": customer.get("relationship", {}),
            "preferences": customer.get("preferences", {}),
        }

    return f"""CATEGORY:
{json.dumps(cat_block, ensure_ascii=False, indent=2)}

MERCHANT:
{json.dumps(merchant_block, ensure_ascii=False, indent=2)}

TRIGGER:
{json.dumps(trigger_block, ensure_ascii=False, indent=2)}

CUSTOMER:
{json.dumps(customer_block, ensure_ascii=False, indent=2) if customer_block else "null (merchant-facing message)"}

Compose the message now. Every number in your output must come from the context above."""


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """
    Core composition function. Deterministic at temperature=0.
    Returns: {body, cta, send_as, suppression_key, rationale}
    """
    client = _get_client()
    user_content = _extract_context(category, merchant, trigger, customer)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=600,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if model adds them
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

    result = json.loads(raw)

    # Ensure suppression_key falls back to trigger's own key
    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get("suppression_key", f"trigger:{trigger.get('id', 'unknown')}")

    return result
