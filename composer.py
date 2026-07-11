"""
compose() — core message engine for Vera.
AWS Bedrock Nova Lite at temperature=0 for determinism.
"""

import json
import os
import boto3

_bedrock: boto3.client = None

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


SYSTEM_PROMPT = """You are Vera, magicpin's WhatsApp AI for merchant growth in India.
Compose ONE sharp WhatsApp message that makes the merchant reply immediately.
Use numbers from DERIVED_FACTS directly — they are pre-verified, never invent data.

Pick ONE engagement formula (do not mix):
A) PRE-LOAD: "Maine [X] ready kar diya — sirf YES bolna hai."
B) LOSS ANCHOR: "Aap [specific number] [thing] miss kar rahe hain — [fix in N min]?"
C) CURIOSITY GAP: "Want to see [specific thing only reply reveals]?"
D) BINARY COMMIT: "Reply YES to [action] / STOP to skip." + time-bound if possible

Category voice — use ≥2 words/phrases from the relevant list:
• dentists → fluoride recall, caries recurrence, high-risk cohort, DCI, Dr., patient-ed
• salons → balayage, keratin, bridal trial, retention, same-day slot, footfall
• restaurants → covers, AOV, footfall, delivery radius, Swiggy/Zomato, dine-in
• gyms → trial-to-paid, churn, HIIT, member retention, September wave
• pharmacies → chronic-Rx, batch, molecule, dispensed, compliance, refill
Taboos: dentists→cure/guaranteed/best price; pharmacies→alarming language

Per-trigger data to cite:
• research_digest → DERIVED_FACTS.cited_study (source + n= + delta%) + merchant's patient segment
• perf_dip → exact metric + delta_pct + baseline number → loss aversion frame
• perf_spike → metric + delta% + likely_driver → capitalize before it fades
• renewal_due → days_remaining + plan name + what pauses on expiry
• supply_alert → batch numbers + molecule + DERIVED_FACTS.affected_chronic_patients count
• competitor_opened → distance_km + their_offer vs your_offer
• festival_upcoming → days_until + specific offer from catalog + one execution idea
• recall_due / chronic_refill_due → months_lapsed + available_slots + price
• dormant_with_vera → DO NOT say "N days since last message". Use strongest signal:
  priority order: CTR vs peer gap > lapsed_customers > no active offers > stale posts
  connect CTR gap to lost revenue or missed bookings, not just "profile stats"
• milestone_reached → current value + milestone label + what next milestone unlocks
• review_theme_emerged → theme + occurrence_count + common_quote + fix action
• curious_ask_due → ONE specific guess: "Is hafte [specific guess] chal raha?"
• winback_eligible → days_since_expiry + perf_dip_pct + lapsed_customers_since_expiry
• active_planning_intent → deliver the artifact NOW (pricing table, draft copy, plan)
• seasonal_perf_dip → peer range (-25 to -35%) + reframe as retention opportunity
• gbp_unverified → estimated_uplift_pct + verification path + time to complete

Hard rules:
1. Use ONLY numbers from context. Never invent facts, citations, or competitor names.
2. ONE CTA in the LAST sentence only. Binary for action triggers; question for curiosity.
3. Always use owner_first_name. Never "Hi there" or "Dear Merchant".
4. Hindi-English mix if languages includes "hi". Natural, not forced.
5. No preamble. No "I hope you're doing well." Cut to the point immediately.
6. Price format: "Haircut @ ₹99" not "10% off".
7. scope=customer → send_as=merchant_on_behalf. Else → vera.
8. Body: ≤100 words merchant-facing, ≤70 words customer-facing.

--- FEW-SHOT EXAMPLES (study these, then compose for the actual context) ---

Example 1 — dentist + research_digest — GOOD (PRE-LOAD + category vocab + specific citation):
{
  "body": "Dr. Priya, JIDA's Nov issue landed. 2,800-patient RCT shows 3-month fluoride recall cuts caries recurrence 41% vs 6-month schedules — directly relevant to your 156 high-risk adult cohort. Maine draft patient-ed WhatsApp ready kar diya — sirf YES bolna hai aur main bhej deta hoon.",
  "cta": "binary",
  "send_as": "vera",
  "suppression_key": "dentist:research_digest:priya_dental",
  "rationale": "research_digest → JIDA Nov RCT (2800 patients, 41% recurrence reduction) → high-risk cohort = immediate recall opportunity"
}

Example 2 — pharmacy + supply_alert — GOOD (LOSS ANCHOR + urgent batch cite + affected count):
{
  "body": "Sharma bhai, urgent: ~57 of your 380 chronic-Rx patients likely dispensed Metformin 500mg batch MP-2024-112. Compliance risk aur potential dispensing liability dono hain. Maine affected-patient list ready kar diya — reply YES, 10 min mein bhejta hoon.",
  "cta": "binary",
  "send_as": "vera",
  "suppression_key": "pharmacy:supply_alert:sharma_medical",
  "rationale": "supply_alert → Metformin batch MP-2024-112 → 57 at-risk chronic-Rx patients = urgent compliance action"
}

Example 3 — salon + dormant_with_vera — GOOD (LOSS ANCHOR using CTR gap, not dormancy):
{
  "body": "Rekha, aapka CTR 1.8% hai — peer avg 3.5% se 49% neeche. Iska matlab: roughly 60+ potential bookings har hafte aapke profile pe aa ke nikal ja rahi hain bina call kiye. Maine ek same-day slot campaign draft kar diya — sirf YES bolna hai.",
  "cta": "binary",
  "send_as": "vera",
  "suppression_key": "salon:dormant_with_vera:rekha_salon",
  "rationale": "dormant_with_vera → strongest signal is CTR gap (49% below peer) → translate gap to missed bookings"
}

--- END EXAMPLES ---

Return ONLY valid JSON (no markdown fences, no extra text):
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

    # CTR gap — always compute, used by dormant_with_vera and others
    ctr = perf.get("ctr")
    peer_ctr = peer.get("avg_ctr")
    if ctr and peer_ctr:
        gap_pct = round(abs(peer_ctr - ctr) / peer_ctr * 100, 1)
        if ctr < peer_ctr:
            facts["ctr_gap"] = f"CTR {ctr*100:.1f}% — peer avg {peer_ctr*100:.1f}% — {gap_pct}% below peers"
        else:
            facts["ctr_gap"] = f"CTR {ctr*100:.1f}% — {gap_pct}% above peer avg {peer_ctr*100:.1f}%"

    # Lapsed customers
    lapsed = cust_agg.get("lapsed_180d_plus") or cust_agg.get("lapsed_90d_plus")
    if lapsed:
        facts["lapsed_customers"] = f"{lapsed} customers lapsed (not visited in 90-180+ days)"

    # Gym churn risk
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

    # Perf delta for dip/spike/seasonal
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
            + (f" — renewal ₹{amount}" if amount else "")
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

    # Research digest: pre-lookup the exact study so LLM can quote source/n/delta
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
        + "\n\nNow compose the message for THIS merchant. Use DERIVED_FACTS numbers. Do NOT copy the examples — generate fresh for this specific context."
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
    user_content = _extract_context(category, merchant, trigger, customer)

    response = client.converse(
        modelId=MODEL,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": user_content}]}],
        inferenceConfig={"temperature": 0, "maxTokens": 600},
    )

    content = response["output"]["message"]["content"][0]["text"]
    if not content:
        raise ValueError("Bedrock returned empty content")

    result = _parse_json(content)

    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get(
            "suppression_key", f"trigger:{trigger.get('id', 'unknown')}"
        )
    return result
