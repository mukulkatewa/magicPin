"""
compose() — core message engine for Vera.
AWS Bedrock Nova Lite, temperature=0, tool-use for guaranteed JSON output.
"""

import re
import json
import os
import boto3

_bedrock = None

def _get_client():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
    return _bedrock

MODEL = os.environ.get("BEDROCK_NOVA_MODEL_ID", "amazon.nova-lite-v1:0")

# ── Tool schema — forces model to return valid structured JSON ─────────────────
COMPOSE_TOOL = [{
    "toolSpec": {
        "name": "compose_message",
        "description": "Return the composed WhatsApp message and metadata",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "body": {"type": "string", "description": "Message body. <= 100 words merchant-facing, <= 70 customer-facing."},
                    "cta": {"type": "string", "enum": ["binary", "open_ended", "none"]},
                    "send_as": {"type": "string", "enum": ["vera", "merchant_on_behalf"]},
                    "suppression_key": {"type": "string", "description": "Format: category:trigger_kind:merchant_short_id"},
                    "rationale": {"type": "string", "description": "trigger_kind -> signal used -> why this merchant now"},
                },
                "required": ["body", "cta", "send_as", "suppression_key", "rationale"],
            }
        },
    }
}]

# ── Base rules (shared across all categories) ─────────────────────────────────
_BASE = """You are Vera, magicpin's WhatsApp AI for merchant growth in India.
Compose ONE sharp message that makes the merchant reply immediately.
Use numbers from DERIVED_FACTS directly — they are pre-verified, never invent data.

Engagement formula — pick ONE, do not mix:
A) PRE-LOAD: "Maine [X] ready kar diya — sirf YES bolna hai."
B) LOSS ANCHOR: "Aap [specific number] [thing] miss kar rahe hain — [fix in N min]?"
C) CURIOSITY GAP: "Want to see [specific thing only reply reveals]?"
D) BINARY COMMIT: "Reply YES to [action] / STOP to skip." + time-bound

Hard rules:
- Only numbers from context. Never invent citations or competitor names.
- ONE CTA in the LAST sentence only.
- Use owner_first_name always. Never "Hi there" / "Dear Merchant".
- Hindi-English mix if languages includes "hi". Natural, not forced.
- No preamble. No "I hope you're doing well." Cut straight to the point.
- Price: "Service @ Rs.99" not "10% off".
- scope=customer -> send_as=merchant_on_behalf. Else -> vera.
- Body: <= 100 words merchant-facing, <= 70 words customer-facing.
- Do NOT use em dashes (--) or smart quotes inside the body text."""

# ── Category-specific system prompts ─────────────────────────────────────────
_DENTIST = _BASE + """

CATEGORY: DENTAL CLINIC
Voice: peer-clinical, evidence-based, collegial. You are addressing a fellow professional.
Salutation: always "Dr. {first_name}"
Mandatory vocab (use >= 2): fluoride varnish, caries, periodontal, RCT, OPG, IOPA, zirconia, aligner, high-risk cohort, DCI, JIDA, patient-ed
Taboos: guaranteed, 100% safe, completely cure, best in city

Trigger strategies:
- research_digest: cite DERIVED_FACTS.cited_study (source + n= + delta%) -> link to merchant's patient segment -> PRE-LOAD with draft patient-ed
- regulation_change: cite DCI circular number + compliance deadline + what changes + "Maine checklist ready kar diya" -> PRE-LOAD
- recall_due / chronic_refill_due: cite months_lapsed + available_slots + patient name -> PRE-LOAD with appointment slot
- cde_opportunity: cite source + credits + date + topic relevance to merchant's case-mix -> CURIOSITY GAP or BINARY COMMIT
- competitor_opened: cite distance_km + their offer vs merchant offer -> differentiation angle -> BINARY COMMIT
- perf_dip: cite perf_signal (metric + delta + baseline) -> LOSS ANCHOR
- dormant_with_vera: strongest signal first (ctr_gap > lapsed_customers > no offers) -> translate gap to lost patients

Perfect example (research_digest):
body: "Dr. Meera, JIDA Oct 2026 p.14 landed. 2,100-patient trial shows 3-month fluoride varnish recall cuts caries recurrence 38% vs 6-month -- directly hits your high-risk adult cohort. Maine draft patient-ed WhatsApp ready kar diya -- sirf YES bolna hai."
cta: binary | send_as: vera"""

_SALON = _BASE + """

CATEGORY: SALON / BEAUTY
Voice: warm, practical, aspirational. Treat the owner as a savvy business person.
Salutation: first_name (no title)
Mandatory vocab (use >= 2): balayage, keratin, bridal trial, retention, same-day slot, footfall, Olaplex, highlights, manicure/pedicure
Taboos: guaranteed glow, permanent results, instant transformation, miracle, best in city

Trigger strategies:
- festival_upcoming: cite days_until + specific service from offer_catalog at price + one execution idea -> BINARY COMMIT
- wedding_package_followup: cite customer name + package discussed + slot availability -> PRE-LOAD with booking link/draft
- curious_ask_due: one specific business guess -> "Is hafte [bridal trials / keratin demand] chal raha?" -> CURIOSITY GAP
- winback_eligible: cite days_since_expiry + perf_dip_pct + lapsed_customers -> LOSS ANCHOR
- dormant_with_vera: use ctr_gap -> translate to missed footfall/bookings per week -> PRE-LOAD campaign

Perfect example (dormant_with_vera, CTR gap):
body: "Anjali, aapka CTR 2.2% hai -- peer avg 4.0% se 45% neeche. Rough math: roughly 70+ potential customers har hafte profile dekh ke bina call kiye nikal jaate hain. Maine ek same-day slot campaign draft kar diya -- sirf YES bolna hai."
cta: binary | send_as: vera"""

_RESTAURANT = _BASE + """

CATEGORY: RESTAURANT / CAFE
Voice: warm, busy, revenue-focused. Owner is time-poor, numbers-first.
Salutation: first_name
Mandatory vocab (use >= 2): covers, AOV, footfall, dine-in, delivery radius, Swiggy/Zomato, table turnover, reservations
Taboos: best food in city, guaranteed packed house, miracle marketing

Trigger strategies:
- ipl_match_today / festival_upcoming: cite match/event name + venue proximity + specific offer -> BINARY COMMIT with time-bound
- review_theme_emerged: cite theme + occurrence_count + one verbatim quote snippet -> propose fix action -> BINARY COMMIT
- milestone_reached: cite current value + milestone label + next milestone + what it unlocks -> CURIOSITY GAP
- active_planning_intent: deliver the artifact NOW (menu, pricing table, draft copy) -> PRE-LOAD
- perf_dip / perf_spike: cite perf_signal (metric + delta + baseline) -> LOSS ANCHOR or capitalize angle

Perfect example (review_theme_emerged):
body: "Suresh, 4 customers recently complained about late delivery -- average mention across 6 reviews this week. One said 'waited 55 mins for a 3 km order.' Maine ek delivery-SLA tweak plan ready kar diya -- sirf YES bolna hai."
cta: binary | send_as: vera"""

_GYM = _BASE + """

CATEGORY: GYM / FITNESS
Voice: energetic, disciplined, data-driven. Owner cares about retention numbers.
Salutation: first_name
Mandatory vocab (use >= 2): membership churn, trial-to-paid, PT sessions, HIIT, retention, September wave, 1RM, footfall
Taboos: guaranteed weight loss, shred in 7 days, miracle transformation

Trigger strategies:
- seasonal_perf_dip: cite perf_signal + peer range (-25 to -35%) + reframe as retention opportunity -> BINARY COMMIT
- customer_lapsed_hard: cite customer name + days_lapsed + their last activity -> PRE-LOAD with re-engagement offer
- active_planning_intent: deliver the artifact NOW (program draft, schedule, pricing) -> PRE-LOAD
- trial_followup: cite trial customer name + activity done + trial-to-paid conversion angle -> BINARY COMMIT
- perf_spike: cite metric + delta% + likely_driver -> capitalize before it fades -> CURIOSITY GAP

Perfect example (seasonal_perf_dip):
body: "Karthik, April-June is lowest acquisition window for gyms -- peer avg dip is -28%. Aapka bhi same pattern. But retention is where money is: members who complete 90 days churn at 3x lower rate. Maine a 90-day challenge template ready kar diya -- sirf YES bolna hai."
cta: binary | send_as: vera"""

_PHARMACY = _BASE + """

CATEGORY: PHARMACY / CHEMIST
Voice: trustworthy, precise, compliance-first. Never use alarming language.
Salutation: first_name or bhai/didi (match merchant's name gender)
Mandatory vocab (use >= 2): chronic-Rx, batch, molecule, dispensed, compliance, refill, schedule H, generic, OTC
Taboos: miracle cure, guaranteed result, 100% safe, alarming language about patient risk

Trigger strategies:
- supply_alert: cite batch numbers + molecule + DERIVED_FACTS.affected_chronic_patients -> compliance/liability angle -> PRE-LOAD with patient list
- chronic_refill_due: cite customer name + molecule + days_since_last_refill + available_slots -> PRE-LOAD with appointment
- category_seasonal: cite seasonal demand shift + specific molecules in demand + stocking angle -> BINARY COMMIT
- gbp_unverified: cite estimated_uplift_pct + verification steps + time to complete -> PRE-LOAD checklist

Perfect example (supply_alert):
body: "Ramesh bhai, urgent update: approx 43 of your 240 chronic-Rx patients likely dispensed Atorvastatin 10mg batch RJ-2024-077. Compliance risk from missed doses. Maine affected-patient list ready kar diya -- reply YES, 10 min mein bhejta hoon."
cta: binary | send_as: vera"""

CATEGORY_PROMPTS = {
    "dentists": _DENTIST,
    "salons": _SALON,
    "restaurants": _RESTAURANT,
    "gyms": _GYM,
    "pharmacies": _PHARMACY,
}


def _get_system(category_slug: str) -> str:
    return CATEGORY_PROMPTS.get(category_slug, _BASE)


def _sanitize_body(text: str) -> str:
    """Clean body text so the judge's scorer LLM doesn't produce malformed JSON."""
    text = text.replace("—", " - ").replace("–", " - ")   # em/en dash
    text = text.replace("“", '"').replace("”", '"')        # smart double quotes
    text = text.replace("‘", "'").replace("’", "'")        # smart single quotes
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _derive_facts(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    """Pre-compute numbers the LLM must cite — prevents hallucination and wrong arithmetic."""
    facts = {}
    perf = merchant.get("performance", {})
    cust_agg = merchant.get("customer_aggregate", {})
    peer = category.get("peer_stats", {})

    # CTR gap
    ctr = perf.get("ctr")
    peer_ctr = peer.get("avg_ctr")
    if ctr and peer_ctr:
        gap_pct = round(abs(peer_ctr - ctr) / peer_ctr * 100, 1)
        direction = "below" if ctr < peer_ctr else "above"
        facts["ctr_gap"] = f"CTR {ctr*100:.1f}% -- peer avg {peer_ctr*100:.1f}% -- {gap_pct}% {direction} peers"

    # Lapsed customers
    lapsed = cust_agg.get("lapsed_180d_plus") or cust_agg.get("lapsed_90d_plus")
    if lapsed:
        facts["lapsed_customers"] = f"{lapsed} customers lapsed (not visited in 90-180+ days)"

    # Gym churn risk
    if cust_agg.get("total_active_members"):
        churn = cust_agg.get("monthly_churn_pct", 0)
        at_risk = round(cust_agg["total_active_members"] * churn)
        facts["members_at_risk_monthly"] = f"{at_risk} of {cust_agg['total_active_members']} members at monthly churn risk"

    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})

    # Supply alert: affected patients estimate
    if trg_kind == "supply_alert":
        chronic = cust_agg.get("chronic_rx_count", 0)
        batches = trg_payload.get("affected_batches", [])
        estimated = min(round(chronic * 0.09 * len(batches)), chronic)
        facts["affected_chronic_patients"] = f"~{estimated} of your {chronic} chronic-Rx patients likely dispensed affected batch(es)"

    # Perf delta
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
        facts["renewal_urgency"] = (
            f"{days_rem} days left on {sub.get('plan', 'Pro')} plan"
            + (f" -- renewal Rs.{amount}" if amount else "")
        )

    # Customer recall
    if customer and trg_kind in ("recall_due", "chronic_refill_due"):
        rel = customer.get("relationship", {})
        last_visit = rel.get("last_visit") or trg_payload.get("last_refill")
        slots = trg_payload.get("available_slots", [])
        slot_labels = [s.get("label") for s in slots[:2] if s.get("label")]
        if last_visit:
            facts["recall_info"] = f"Last visit: {last_visit}" + (
                f" | Open slots: {', '.join(slot_labels)}" if slot_labels else ""
            )

    # Research digest: pre-lookup exact study
    if trg_kind == "research_digest":
        top_item_id = trg_payload.get("top_item_id") or trg_payload.get("item_id")
        digest_items = category.get("digest", [])
        match = next((d for d in digest_items if d.get("id") == top_item_id), None) if top_item_id else None
        if not match and digest_items:
            match = digest_items[0]
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
            if match.get("patient_segment"):
                parts.append(f"Segment: {match['patient_segment']}")
            if parts:
                facts["cited_study"] = " | ".join(parts)

    # Winback
    if trg_kind == "winback_eligible":
        days_exp = trg_payload.get("days_since_expiry")
        dip = trg_payload.get("perf_dip_pct")
        lapsed_since = trg_payload.get("lapsed_customers_since_expiry")
        parts = []
        if days_exp:
            parts.append(f"{days_exp} days since expiry")
        if dip:
            parts.append(f"perf dip {abs(dip)*100:.0f}%")
        if lapsed_since:
            parts.append(f"{lapsed_since} customers lapsed since")
        if parts:
            facts["winback_signal"] = " | ".join(parts)

    return facts


def _extract_context(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> str:
    voice = category.get("voice", {})
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
        }

    ctx = {
        "DERIVED_FACTS": derived,
        "category": {
            "slug": category.get("slug"),
            "peer_stats": category.get("peer_stats", {}),
            "top_digest": category.get("digest", [])[:2],
            "seasonal_beats": category.get("seasonal_beats", [])[:2],
            "offer_catalog": category.get("offer_catalog", [])[:3],
            "trend_signals": category.get("trend_signals", [])[:2],
            "taboos": voice.get("vocab_taboo", [])[:3],
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
        + "\n\nCompose for THIS merchant using DERIVED_FACTS numbers. Every number must come from context above."
    )


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """Compose at temperature=0 using tool-use for guaranteed JSON. Returns {body, cta, send_as, suppression_key, rationale}."""
    client = _get_client()
    category_slug = category.get("slug", "")
    system_prompt = _get_system(category_slug)
    user_content = _extract_context(category, merchant, trigger, customer)

    response = client.converse(
        modelId=MODEL,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_content}]}],
        inferenceConfig={"temperature": 0, "maxTokens": 600},
        toolConfig={
            "tools": COMPOSE_TOOL,
            "toolChoice": {"tool": {"name": "compose_message"}},
        },
    )

    # Tool use response — always valid JSON, no parse errors
    content_blocks = response["output"]["message"]["content"]
    tool_block = next((b for b in content_blocks if "toolUse" in b), None)
    if tool_block:
        result = tool_block["toolUse"]["input"]
    else:
        # Fallback: try text response (shouldn't happen with toolChoice forced)
        text = content_blocks[0].get("text", "")
        if not text:
            raise ValueError("Bedrock returned empty content")
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(text, object_pairs_hook=dict)

    # Sanitize body to prevent judge-side parse errors
    if result.get("body"):
        result["body"] = _sanitize_body(result["body"])

    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get(
            "suppression_key", f"trigger:{trigger.get('id', 'unknown')}"
        )

    return result
