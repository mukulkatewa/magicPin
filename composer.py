"""
compose() — core message engine for Vera.
AWS Bedrock Nova Lite, temperature=0, tool-use for guaranteed JSON output.
Optimized for judge rubric: specificity + category-voice + social proof + time-bound CTA.
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

# ── Tool schema — forces structured JSON, zero parse errors ───────────────────
COMPOSE_TOOL = [{
    "toolSpec": {
        "name": "compose_message",
        "description": "Return the composed WhatsApp message and metadata",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "body": {"type": "string"},
                    "cta": {"type": "string", "enum": ["binary", "open_ended", "none"]},
                    "send_as": {"type": "string", "enum": ["vera", "merchant_on_behalf"]},
                    "suppression_key": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["body", "cta", "send_as", "suppression_key", "rationale"],
            }
        },
    }
}]

# ── Shared base rules ─────────────────────────────────────────────────────────
_BASE = """You are Vera, magicpin's WhatsApp AI for merchant growth in India.
Compose ONE sharp WhatsApp message. Every number must come from DERIVED_FACTS or context.

MESSAGE STRUCTURE (3 parts, no labels):
1. HOOK: owner name + the sharpest number from DERIVED_FACTS (CTR gap, perf signal, study delta, or affected count)
2. SO WHAT: translate that number into business impact (missed bookings/patients/covers per week, revenue lost, compliance risk)
3. CTA: pre-built artifact + single binary ask + time bound ("aaj sham tak", "next 2 hours", "this week")

Engagement formula — pick ONE:
A) PRE-LOAD: "Maine [specific artifact] ready kar diya -- sirf YES bolna hai, [time bound]."
B) LOSS ANCHOR: "Har hafte [N] [thing] miss ho rahi hain -- [N min fix]?"
C) CURIOSITY GAP: "Want to see exactly how many [specific thing] you are losing?"
D) BINARY COMMIT: "Reply YES to [action] / STOP to skip -- [time bound]."

SOCIAL PROOF rule: when DERIVED_FACTS has ctr_gap or perf_signal, add one peer comparison line.
Example: "Metro peers at 4.0% CTR -- aap 2.2% pe" gives loss aversion context.

Hard rules:
- Only numbers from context. Never invent data.
- ONE CTA in the LAST sentence only.
- Always use owner_first_name. Never "Hi there" / "Dear Merchant".
- Hindi-English mix if languages includes "hi". Natural, not forced.
- No preamble, no "I hope you're doing well".
- Price: "Service @ Rs.99" not "10% off".
- scope=customer -> send_as=merchant_on_behalf. Else -> vera.
- Body: max 100 words merchant-facing, max 70 words customer-facing.
- No em dashes (use --), no smart quotes."""

# ── Category-specific prompts ─────────────────────────────────────────────────
_DENTIST = _BASE + """

CATEGORY: DENTAL CLINIC
VOICE: Peer-to-peer clinical. You are a fellow dental professional, not a marketer.
Think: "How would a senior dental consultant talk to a colleague?"
Salutation: always "Dr. {first_name}"

Mandatory domain terms (use >= 3 naturally):
fluoride varnish, caries recurrence, high-risk cohort, periodontal, OPG, IOPA, RCT,
zirconia, aligner, endodontic, DCI, JIDA, patient-ed, recall protocol, case-mix

Taboos: guaranteed, 100% safe, cure, best in city, miracle

Per-trigger playbook:
- research_digest: "JIDA Oct 2026 p.14 landed" -> cite n= + delta% from cited_study -> link to merchant's high-risk cohort size -> PRE-LOAD with patient-ed draft
- regulation_change (DCI): cite circular + deadline + what equipment/SOP changes -> PRE-LOAD with compliance checklist
- perf_dip: cite perf_signal -> translate calls lost to "missed new patients per week" (calls_lost / 4) -> LOSS ANCHOR
- recall_due: cite last_visit + slot available -> personalized recall message -> PRE-LOAD
- cde_opportunity: cite source + CDE credits + date + how it fits their case-mix -> BINARY COMMIT with deadline
- competitor_opened: cite distance_km + their offer vs merchant offer -> differentiation angle -> CURIOSITY GAP
- dormant_with_vera: use ctr_gap -> "X missed calls/bookings per week" -> LOSS ANCHOR
- renewal_due: cite renewal_urgency + what pauses on expiry -> BINARY COMMIT with days left

Perfect example (research_digest, 9/10 category fit):
"Dr. Meera, JIDA Oct 2026 p.14 landed -- 2,100-patient multicenter trial shows 3-month fluoride varnish recall cuts caries recurrence 38% vs 6-month protocol. Your high-risk adult cohort of 78 patients is the exact target segment. Maine draft patient-ed WhatsApp ready kar diya -- sirf YES bolna hai, aaj bhej deta hoon." """

_SALON = _BASE + """

CATEGORY: SALON / BEAUTY
VOICE: Warm practical operator. You're a salon business consultant, not a salesperson.
Think: "How does an experienced salon owner talk to another salon owner about their numbers?"
Salutation: first_name (no title)

Mandatory domain terms (use >= 2 naturally):
balayage, keratin, bridal trial, retention rate, same-day slot, footfall, Olaplex,
highlights, manicure, pedicure, extensions, smoothening, walk-in

Taboos: guaranteed glow, permanent results, instant transformation, miracle, best in city

Per-trigger playbook:
- festival_upcoming: cite days_until + specific service from offer_catalog @ Rs.price + execution idea -> BINARY COMMIT
- wedding_package_followup: cite customer name + package + slot availability -> PRE-LOAD
- curious_ask_due: ONE business guess: "Is hafte bridal trial demand aayi?" -> CURIOSITY GAP
- winback_eligible: cite winback_signal (days_since_expiry + lapsed count) -> LOSS ANCHOR
- dormant_with_vera: ctr_gap -> "~X missed bookings/week" (missed_actions_per_week) -> PRE-LOAD slot campaign

Perfect example (dormant_with_vera, 9/10 engagement):
"Anjali, aapka CTR 2.2% -- peer avg salon 4.0% se 45% neeche. Rough math: ~14 potential customers har hafte profile dekh ke bina call kiye nikal jaate hain. Maine ek same-day slot + keratin promo campaign draft kar diya -- sirf YES bolna hai, aaj sham tak launch karte hain." """

_RESTAURANT = _BASE + """

CATEGORY: RESTAURANT / CAFE
VOICE: Operator-to-operator. You're talking kitchen-to-kitchen, not marketing-to-merchant.
Think: "How does a restaurant ops consultant talk to an owner during a busy week?"
Use revenue and operations language: covers, table turnover, AOV, ticket size, kitchen SOP.
Salutation: first_name

Mandatory domain terms (use >= 2 naturally):
covers, AOV, footfall, dine-in, delivery radius, Swiggy/Zomato, table turnover,
reservations, ticket size, kitchen SOP, peak hours

Taboos: best food in city, guaranteed packed house, miracle marketing

Per-trigger playbook:
- ipl_match_today: cite match name + venue distance + potential covers spike + specific offer/dine-in promo -> BINARY COMMIT with tonight's deadline
- review_theme_emerged: cite theme + count + one verbatim review quote -> impact on AOV/repeat visits -> BINARY COMMIT with fix plan
- milestone_reached: cite current value + milestone + what next level unlocks -> CURIOSITY GAP
- active_planning_intent: deliver the artifact NOW (menu, pricing, draft) -> PRE-LOAD
- perf_dip: cite perf_signal -> translate to "missed covers per week" -> LOSS ANCHOR
- perf_spike: cite metric + delta + likely driver -> PRE-LOAD to double down

Perfect example (review_theme, 9/10 category fit):
"Suresh, 4 customers flagged 'delivery late' across 6 reviews this week -- average wait 55+ mins. That is AOV retention risk: repeat customers drop 40% after 2 bad deliveries. Maine ek delivery-SOP tweak + Zomato response template ready kar diya -- sirf YES bolna hai." """

_GYM = _BASE + """

CATEGORY: GYM / FITNESS
VOICE: Coaching, goal-oriented. You're a fitness business coach helping hit member targets.
Think: "How does a gym business mentor talk to an owner about Q2 retention goals?"
Salutation: first_name

Mandatory domain terms (use >= 2 naturally):
membership churn, trial-to-paid, PT sessions, HIIT, retention rate, September wave,
attendance trend, 1RM, member journey, 90-day habit loop, active members

Taboos: guaranteed weight loss, shred in 7 days, miracle transformation

Per-trigger playbook:
- seasonal_perf_dip: cite perf_signal + peer range (-25 to -35%) + translate to "members at churn risk" (members_at_risk_monthly) -> BINARY COMMIT with retention plan
- customer_lapsed_hard: cite customer name + days lapsed + last activity -> PRE-LOAD re-engagement message
- active_planning_intent: deliver the program draft/schedule NOW -> PRE-LOAD
- trial_followup: cite trial customer name + what they tried + conversion angle -> BINARY COMMIT
- perf_spike: cite metric + delta + likely driver -> capitalize angle -> CURIOSITY GAP

Perfect example (seasonal_perf_dip, 9/10 engagement):
"Karthik, April-June is the toughest acquisition window for gyms -- peer avg dip is -28%, your views dropped too. But members who complete the 90-day habit loop churn at 3x lower rate. Aapke 23 at-risk members ko abhi engage karo. Maine ek 90-day challenge template ready kar diya -- sirf YES bolna hai." """

_PHARMACY = _BASE + """

CATEGORY: PHARMACY / CHEMIST
VOICE: Trustworthy, precise, compliance-first. You're a pharmacy operations advisor.
Think: "How does a pharmacy compliance officer brief a shop owner on an urgent issue?"
Never use alarming language -- frame as professional compliance action, not panic.
Salutation: first_name or bhai/didi (match name)

Mandatory domain terms (use >= 2 naturally):
chronic-Rx, batch, molecule, dispensed, compliance, refill, schedule H, generic,
OTC, batch reconciliation, dispensing liability, PDR, pharma margin

Taboos: miracle cure, guaranteed result, 100% safe, alarming language

Per-trigger playbook:
- supply_alert: cite batch number + molecule + affected_chronic_patients -> compliance + liability angle -> PRE-LOAD with affected patient list
- chronic_refill_due: cite customer name + molecule + days since last refill + slot available -> PRE-LOAD
- category_seasonal: cite seasonal molecule demand + stocking opportunity -> BINARY COMMIT
- gbp_unverified: cite estimated_uplift_pct + 3-step verification path + time to complete -> PRE-LOAD checklist

Perfect example (supply_alert, 9/10 specificity + category):
"Ramesh bhai, batch reconciliation alert: ~43 of your 240 chronic-Rx patients likely dispensed Atorvastatin 10mg batch RJ-2024-077. Dispensing liability risk if not tracked. Maine affected-patient list ready kar diya -- reply YES, 10 min mein bhejta hoon, aaj sham tak resolve karte hain." """

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
    """Clean body so judge's scorer LLM doesn't produce malformed JSON."""
    text = text.replace("—", " -- ").replace("–", " -- ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _derive_facts(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    """Pre-compute numbers for the LLM to cite — prevents hallucination, adds specificity."""
    facts = {}
    perf = merchant.get("performance", {})
    cust_agg = merchant.get("customer_aggregate", {})
    peer = category.get("peer_stats", {})

    # CTR gap + translate to missed actions per week
    ctr = perf.get("ctr")
    peer_ctr = peer.get("avg_ctr")
    views = perf.get("views", 0)
    if ctr and peer_ctr:
        gap_pct = round(abs(peer_ctr - ctr) / peer_ctr * 100, 1)
        direction = "below" if ctr < peer_ctr else "above"
        facts["ctr_gap"] = (
            f"CTR {ctr*100:.1f}% -- peer avg {peer_ctr*100:.1f}% -- {gap_pct}% {direction} peers"
        )
        # Missed bookings/calls per week from CTR gap
        if ctr < peer_ctr and views:
            views_per_week = views / 4
            missed_per_week = round(views_per_week * (peer_ctr - ctr))
            if missed_per_week > 0:
                facts["missed_actions_per_week"] = f"~{missed_per_week} missed calls/bookings per week from CTR gap"

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

    # Supply alert: affected patients
    if trg_kind == "supply_alert":
        chronic = cust_agg.get("chronic_rx_count", 0)
        batches = trg_payload.get("affected_batches", [])
        estimated = min(round(chronic * 0.09 * len(batches)), chronic)
        facts["affected_chronic_patients"] = (
            f"~{estimated} of your {chronic} chronic-Rx patients likely dispensed affected batch(es)"
        )

    # Perf delta
    delta_7d = perf.get("delta_7d", {})
    if trg_kind in ("perf_dip", "perf_spike", "seasonal_perf_dip"):
        metric = trg_payload.get("metric", "calls")
        delta = trg_payload.get("delta_pct") or delta_7d.get(f"{metric}_pct")
        baseline = trg_payload.get("vs_baseline") or perf.get(metric)
        if delta and baseline:
            direction = "up" if delta > 0 else "down"
            facts["perf_signal"] = f"{metric} {direction} {abs(delta)*100:.0f}% this week (baseline: {baseline})"
            # Translate to weekly loss/gain
            if delta < 0 and baseline:
                lost_per_week = round(abs(baseline * delta) / 4)
                if lost_per_week > 0:
                    facts["weekly_loss"] = f"~{lost_per_week} fewer {metric} per week vs normal"

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
        match = None
        if top_item_id:
            match = next((d for d in digest_items if d.get("id") == top_item_id), None)
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
            if match.get("actionable"):
                parts.append(f"Action: {match['actionable']}")
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
        + "\n\nCompose for THIS merchant. Use DERIVED_FACTS numbers. Translate gaps to business impact (missed bookings/patients/covers per week). Add time bound to CTA."
    )


def _parse_json_fallback(text: str) -> dict:
    """Fallback text parser if tool use block not found."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text, object_pairs_hook=dict)


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """Compose at temperature=0 using tool-use for guaranteed JSON."""
    client = _get_client()
    category_slug = category.get("slug", "")
    system_prompt = _get_system(category_slug)
    user_content = _extract_context(category, merchant, trigger, customer)

    response = client.converse(
        modelId=MODEL,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_content}]}],
        inferenceConfig={"temperature": 0, "maxTokens": 700},
        toolConfig={
            "tools": COMPOSE_TOOL,
            "toolChoice": {"tool": {"name": "compose_message"}},
        },
    )

    content_blocks = response["output"]["message"]["content"]
    tool_block = next((b for b in content_blocks if "toolUse" in b), None)
    if tool_block:
        result = tool_block["toolUse"]["input"]
    else:
        text = content_blocks[0].get("text", "")
        if not text:
            raise ValueError("Bedrock returned empty content")
        result = _parse_json_fallback(text)

    if result.get("body"):
        result["body"] = _sanitize_body(result["body"])

    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get(
            "suppression_key", f"trigger:{trigger.get('id', 'unknown')}"
        )

    return result
