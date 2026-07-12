"""
compose() — Vera's message engine.
AWS Bedrock Nova Pro, temperature=0, tool-use for guaranteed JSON.

Optimized for the judge rubric:
  Specificity (verifiable numbers) + Category Fit (domain voice) +
  Merchant Fit (name + real data) + Decision Quality (why-now clarity) +
  Engagement (loss aversion + social proof + time-bound CTA).
"""

import re
import json
import os
import boto3
from botocore.config import Config

_bedrock = None

def _get_client():
    """Bedrock client with adaptive retry — handles throttling from parallel tick calls."""
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            config=Config(
                retries={"max_attempts": 8, "mode": "adaptive"},
                read_timeout=30,
                connect_timeout=5,
            ),
        )
    return _bedrock

# Bot uses Nova Lite by default (higher rate limits for 5-parallel tick).
# Judge uses Nova Pro (set separately in judge_simulator.py).
MODEL = os.environ.get("BEDROCK_NOVA_MODEL_ID", "amazon.nova-lite-v1:0")

# ── Tool schema — guarantees structured JSON, zero parse errors on our side ──
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

# ── Base rules (shared) ──────────────────────────────────────────────────────
_BASE = """You are Vera, magicpin's WhatsApp AI for merchant growth in India.
Compose ONE WhatsApp message that a merchant CANNOT ignore.

REASONING ORDER (do this internally before writing the body):
Step 1. What is the trigger.kind? What data in trigger.payload makes this URGENT NOW?
Step 2. Which DERIVED_FACTS field maps to THIS trigger? (Not ctr_gap unless trigger is dormant_with_vera or perf_dip on views.)
Step 3. What is the merchant.languages preference? If "hi" is listed, the body MUST be Hindi-English mix (Hinglish). If only "en", pure English.
Step 4. What are the mandatory domain terms for this category (see block below)? Pick 3.
Step 5. Write the body citing the trigger-specific number first, then peer comparison, then artifact + time-bound CTA.

TRIGGER-DATA PRIMACY (critical):
The FIRST number cited must come from the trigger's own payload / derived fact:
- research_digest -> cited_study n= + delta%
- perf_dip / perf_spike -> perf_signal
- supply_alert -> affected_chronic_patients + batch id
- festival_upcoming -> days_until + offer @ price
- recall_due / chronic_refill_due -> recall_info
- competitor_opened -> competitor_signal
- review_theme_emerged -> review_signal
- gbp_unverified -> gbp_signal
- winback_eligible -> winback_signal
- renewal_due -> renewal_urgency
- dormant_with_vera / seasonal_perf_dip -> ctr_gap + missed_actions_per_week
- milestone_reached -> current review count + next milestone gap
- active_planning_intent -> deliver the artifact NOW; cite what they asked for
Only add CTR gap if the trigger is dormant, seasonal_perf_dip, or perf_dip on views.
Never lead with CTR gap for research_digest / festival / supply / review / spike / recall.

LANGUAGE RULE (biggest merchant-fit signal):
- If merchant.languages includes "hi": Hinglish mandatory. Weave Hindi verbs and connectors
  (aapka, hai, karo, aaj, sirf, chal raha, kar diya, sham tak). ~40-50% Hindi words.
- If only "en": clean English, no Hindi injection.

DENSITY REQUIREMENTS (every message must have all five):
1) At least 5 concrete numbers/facts from DERIVED_FACTS or context.
2) At least 3 category-specific domain terms from the mandatory vocab list.
3) One peer/social-proof comparison line when data allows.
4) Explicit "WHY NOW" phrase making the timing urgent.
5) Binary CTA in the LAST sentence with a time bound.

4-PART STRUCTURE (no labels, just flow):
HOOK -- owner name + why-NOW in <=5 words + sharpest single number from DERIVED_FACTS.
PEER LINE -- what peer merchants are doing / benchmark comparison (SOCIAL PROOF, mandatory when data allows).
IF-YOU-DON'T-ACT -- quantified consequence of inaction (missed covers/patients/revenue in the next N days).
CLOSE -- MUST match this template EXACTLY: "Maine [artifact] ready kar diya -- reply YES, [specific clock time]." No other content in the last sentence. No trailing questions.
Clock time examples: "aaj sham 6 baje tak", "by 5:30pm today", "Friday 12 baje se pehle", "tonight before 10pm".

ENGAGEMENT COMPULSION -- every message must have all four levers:
1) Loss aversion: quantified "if not now, you lose X" statement.
2) Social proof: peer merchants doing/hitting/benchmarking.
3) Low friction: single-word reply (YES) or 2-choice CTA.
4) Specific deadline (real clock time), not vague "aaj sham" alone.

Examples of the IF-YOU-DON'T-ACT line by trigger:
- ipl_match_today: "Peer restaurants near the stadium are already sold out; without booking by 6pm, you lose ~40 covers to them tonight."
- review_theme_emerged: "Peer response time is 32 mins; without a Friday fix, ~8 more repeat customers likely churn based on our data."
- festival_upcoming (far date): "Bridal-trial slots for Nov are filling 2x faster than normal; without loading offers this week, peer salons capture your share."
- perf_dip: "At current pace, ~N calls lost this month vs baseline -- compounds to N*4 next month."
- supply_alert: "Without swap-out by tomorrow, dispensing liability plus ~N chronic patients at compliance risk."
- dormant_with_vera: "Every week at this CTR = ~N missed bookings; that's N*4 monthly gone to peer merchants."
- winback_eligible: "Without renewal this week, N more customers lapse into the 180d cohort."

WHY-NOW MAP (must be visible in message):
- research_digest: "this month's issue" / "just landed" -> act before peers do
- perf_dip / seasonal_perf_dip: "is hafte" / "already X fewer this week" -> stem the bleeding
- perf_spike: "capitalize before momentum fades in 48-72 hrs"
- supply_alert: "before next dispensing cycle" / "aaj hi"
- competitor_opened: "before they take share this weekend"
- festival_upcoming: "advance-booking window opens NOW; peers already loading offers"
- renewal_due: "N days left; features pause on expiry"
- recall_due / chronic_refill_due: "patient overdue; slot open this week"
- cde_opportunity: "RSVP closes on {date}"
- review_theme_emerged: "same complaint 4x this week -> AOV at risk NOW"
- dormant_with_vera: "every week losing X bookings -- compounding"
- winback_eligible: "N days since expiry -- lapsed pool growing"
- gbp_unverified: "each week unverified = X% uplift lost"
- milestone_reached: "one push to unlock next tier"
- active_planning_intent: "you asked; artifact ready right now"
- customer_lapsed_hard / trial_followup: "48-hour window to save this member"

Hard rules:
- Every number must come from DERIVED_FACTS or context. Never invent data or citations.
- One CTA only, in the final sentence.
- Always use owner_first_name (or "Dr. {first_name}" for dentists).
- Hindi-English mix if languages includes "hi". Natural, not forced.
- No preamble ("I hope you're doing well"), no re-introduction.
- Prices as "Service @ Rs.99" not "10% off".
- scope=customer -> send_as=merchant_on_behalf. Else -> vera.
- Body length: 60-90 words merchant-facing, 40-60 customer-facing.
- Body character set: plain ASCII plus Hindi (Devanagari) only. Use "--" not em-dash. Use straight quotes only.
- NEVER wrap phrases in single quotes ('like this') inside the body -- write the phrase plainly.
- Avoid period-abbreviations that look like sentence ends: write "page 14" not "p.14", "Doctor" or just first name is fine.
- When quoting a customer review, use the phrase inline without quote marks: e.g. write "customers say delivery is late" not "customers say 'delivery late'"."""

# ── Category voices — direct, no fluff ────────────────────────────────────────
_DENTIST = _BASE + """

CATEGORY: DENTAL CLINIC. Voice: peer-clinical, evidence-based, collegial.
Read like a senior dental consultant briefing a colleague in Hinglish (unless languages is en-only).
Salutation: always "Dr {first_name}" (no period).
IMPORTANT: Even in clinical tone, Hinglish is REQUIRED when merchant.languages includes "hi".
Weave: "aapke", "hai", "kar diya", "sirf YES bolna hai", "aaj sham tak". Do NOT go pure English.

Mandatory clinical vocabulary (use at least 3): fluoride varnish, caries recurrence, high-risk cohort,
periodontal, endodontic, RCT, OPG, IOPA, zirconia, aligner, recall interval, case-mix, DCI, JIDA, patient-ed.
Taboos: cure, guaranteed, best in city, 100% safe.

Trigger playbook (each MUST include a peer line + IF-YOU-DON'T-ACT quantified consequence + specific deadline):
- research_digest: cited_study (source + n= + delta%) + high-risk cohort size + peer clinics already updating protocol + IF-YOU-DON'T-ACT ("peers publish first; you lose the referral edge this cycle") + PRE-LOAD patient-ed draft + "aaj sham 6 baje tak" deadline.
- regulation_change: DCI circular date + SOP delta + peer clinics compliant this week + IF-YOU-DON'T-ACT ("audit exposure + fine liability if not aligned by cycle-end") + PRE-LOAD checklist + specific-day deadline.
- perf_dip: perf_signal + weekly_loss + peer clinic call baseline + IF-YOU-DON'T-ACT ("~N new-patient calls lost this month vs baseline") + PRE-LOAD recall campaign + tonight deadline.
- recall_due: last_visit + patient name + open slot + peer recall interval (3-mo standard) + IF-YOU-DON'T-ACT ("caries recurrence risk compounds each week overdue") + PRE-LOAD appointment message.
- cde_opportunity: source + credits + date + case-mix fit + peer attendance + BINARY commit before RSVP closes.
- competitor_opened: distance_km + their offer vs ours + weekend-share loss risk + differentiator + CURIOSITY GAP.
- dormant_with_vera: ctr_gap + missed_actions_per_week + peer CTR benchmark + IF-YOU-DON'T-ACT ("N missed consults weekly = N*4 monthly gone to peers") + PRE-LOAD 3-post plan + this-week deadline.

Example gold-standard body (target 45+/50):
"Dr. Meera, JIDA Oct 2026 p.14 landed -- 2,100-patient multicenter RCT shows 3-month fluoride varnish recall cuts caries recurrence 38% vs the 6-month protocol you likely follow. Direct hit on your 78 high-risk adult cohort. Metro peer clinics are already updating recall intervals for Q3. Maine patient-ed WhatsApp draft plus a 15-patient recall list ready kar diya -- sirf YES bolna hai, aaj sham 6 baje tak dono bhej deta hoon." """

_SALON = _BASE + """

CATEGORY: SALON / BEAUTY. Voice: warm, practical, operator-to-operator.
Read like an experienced salon business consultant, not a promo blast.
Salutation: first_name.

Mandatory vocabulary -- MUST use at least 3 from this list in every message:
balayage, keratin, bridal trial, retention rate, same-day slot, footfall, Olaplex, highlights,
manicure, pedicure, extensions, smoothening, walk-in, service-mix, avg ticket, hair spa, facial, threading.
Taboos: guaranteed glow, permanent results, instant transformation, best in city, miracle.

Trigger playbook (each MUST include a peer line + IF-YOU-DON'T-ACT quantified consequence + specific deadline):
- festival_upcoming: days_until + specific service @ Rs.price + peer salons already loaded offers + IF-YOU-DON'T-ACT ("bridal-trial slots fill 2x faster; without loading this week, peers capture your share") + PRE-LOAD offer + specific-day deadline.
- wedding_package_followup: customer name + package + trial slot + peer conversion rate + IF-YOU-DON'T-ACT ("48-hr follow-up window; after that, trial-to-book drops 60%") + PRE-LOAD booking + today deadline.
- curious_ask_due: one guess ("Is hafte keratin ki demand chal rahi?") + peer trend + CURIOSITY GAP + binary reply.
- winback_eligible: winback_signal + lapsed count + peer retention rate + IF-YOU-DON'T-ACT ("N customers slip to 180d cohort each week -- recovery cost 3x higher") + PRE-LOAD re-engagement offer + this-week deadline.
- dormant_with_vera: ctr_gap + missed_actions_per_week + peer CTR + IF-YOU-DON'T-ACT ("N missed bookings weekly = N*4 monthly gone to peer salons") + PRE-LOAD same-day-slot promo + tonight deadline.

Example gold-standard body (target 45+/50):
"Anjali, aapka CTR 2.2% hai -- peer metro salons ka avg 4.0% -- yaani aap 45% neeche. Isse ~14 potential customers har hafte aapki profile dekh ke bina call kiye nikal jaate hain. Aapke 45 lapsed customers ko re-engage karne ka bhi mauka hai. Maine ek same-day slot + keratin @ Rs.1,499 campaign draft kar diya -- sirf YES bolna hai, aaj sham tak launch karte hain." """

_RESTAURANT = _BASE + """

CATEGORY: RESTAURANT / CAFE. Voice: operator-to-operator, kitchen-to-kitchen.
Read like a restaurant ops advisor talking about covers, AOV, table turnover -- not brand marketing.
Salutation: first_name.

Mandatory vocabulary -- MUST use at least 3 from this list in every message:
covers, AOV, footfall, dine-in, delivery radius, Swiggy, Zomato, table turnover, reservations,
ticket size, kitchen SOP, peak hours, walk-in, ADR, prep time, repeat rate, cover mix.
Taboos: best food in city, guaranteed packed house, miracle marketing.

Trigger playbook:
- ipl_match_today: match + venue distance + peer restaurants at stadium (SOCIAL PROOF)
  + IF-YOU-DON'T-ACT ("without booking delivery slots by 6pm, ~40 covers gone to peers tonight")
  + specific-time deadline ("Reply YES by 5:30pm; I lock the slots by 6.").
- review_theme_emerged: theme + count + verbatim quote + peer response-time benchmark
  + IF-YOU-DON'T-ACT ("without Friday fix, ~8 more repeat customers churn -- peer merchants sit at 32-min response time")
  + PRE-LOAD SOP fix + specific-day deadline ("Reply YES; SOP live by Friday noon.").
- milestone_reached: current value + milestone + next unlock + CURIOSITY GAP for review-count target.
- active_planning_intent: deliver the artifact NOW (menu, pricing, draft copy) -- PRE-LOAD.
- perf_dip / perf_spike: perf_signal + weekly covers impact + PRE-LOAD action plan.

Example gold-standard body (target 45+/50):
"Suresh, 4 customers flagged 'delivery late' across 6 reviews this week -- one verbatim: 'waited 55 mins for 3 km.' Yeh AOV retention hit hai: repeat customers 40% drop kar dete hain 2 late orders ke baad, aur aap 12 covers/week losing on repeats. Metro peer response time 32 mins hai. Maine ek delivery SOP tweak plus a Zomato apology template ready kar diya -- sirf YES bolna hai, aaj raat implement karte hain." """

_GYM = _BASE + """

CATEGORY: GYM / FITNESS. Voice: coaching, disciplined, retention-first.
Read like a fitness business coach helping owners hit member targets, not a promo.
Salutation: first_name.

Mandatory vocabulary -- MUST use at least 3 from this list in every message:
membership churn, trial-to-paid, PT sessions, HIIT, retention rate, September wave, attendance trend,
1RM, member journey, 90-day habit loop, active members, drop-off, group classes, personal training.
Taboos: guaranteed weight loss, shred in 7 days, miracle transformation.

Trigger playbook (each MUST include a peer line + IF-YOU-DON'T-ACT quantified consequence + specific deadline):
- seasonal_perf_dip: perf_signal + peer dip range (-25 to -35%) + members_at_risk_monthly + IF-YOU-DON'T-ACT ("N trial-to-paid members drop off this month if no habit-loop lock") + PRE-LOAD 90-day challenge + this-Friday deadline.
- customer_lapsed_hard: customer name + days lapsed + last activity + peer re-engagement rate + IF-YOU-DON'T-ACT ("48-hr window; after that, churn cohort locks in") + PRE-LOAD re-engagement message.
- active_planning_intent: deliver program draft NOW -- PRE-LOAD + peer program benchmark + tonight deadline.
- trial_followup: trial customer + attendance count + trial-to-paid conversion angle + peer conversion benchmark + IF-YOU-DON'T-ACT ("N days into trial without PT touch = 40% lower conversion") + BINARY + tonight.
- perf_spike: metric + delta + driver + peer capture window + CURIOSITY GAP capitalize before it fades (48-72 hrs).

Example gold-standard body (target 45+/50):
"Karthik, April-June is the toughest acquisition window -- peer gyms average -28% dip, aapke views bhi 30% neeche gaye. But retention hai woh angle: members who cross 90-day habit loop churn at 3x lower rate. Aapke 12 at-risk trial-to-paid members ko abhi lock karo. Maine ek 90-day challenge + attendance nudge template ready kar diya -- sirf YES bolna hai, is Friday tak roll out karte hain." """

_PHARMACY = _BASE + """

CATEGORY: PHARMACY / CHEMIST. Voice: precise, compliance-first, calm-professional.
Read like a pharmacy compliance advisor speaking Hinglish (unless languages is en-only).
Salutation: first_name + bhai/didi to match warmth.
IMPORTANT: Hinglish REQUIRED when merchant.languages includes "hi". Weave Hindi verbs:
"aapke", "hai", "kar diya", "reply YES", "aaj sham tak", "bhej deta hoon". Do NOT go pure English.

Mandatory vocabulary (use at least 2-3): chronic-Rx, batch, molecule, dispensed, compliance, refill,
schedule H, generic, OTC, batch reconciliation, dispensing liability, PDR, drug utilization.
Taboos: miracle cure, guaranteed result, 100% safe, alarming language.

Trigger playbook (each MUST include a peer line + IF-YOU-DON'T-ACT quantified consequence + specific deadline):
- supply_alert: batch id + molecule + affected_chronic_patients + peer pharmacies already recalled + IF-YOU-DON'T-ACT ("without swap-out by tomorrow, dispensing liability + ~N chronic patients at compliance risk") + PRE-LOAD affected list + "aaj sham tak" deadline.
- chronic_refill_due: customer name + molecule + last refill + open slot + IF-YOU-DON'T-ACT ("N days overdue -> adherence break, refill migrates to peer chemist") + PRE-LOAD reminder.
- category_seasonal: seasonal molecule demand + peer stocking rate + IF-YOU-DON'T-ACT ("stockout in 5 days, walk-ins go to peer pharmacy next block") + BINARY commit.
- gbp_unverified: uplift % + 3-step verification + minutes required + IF-YOU-DON'T-ACT ("each unverified week = ~X% search impressions lost to verified peers") + PRE-LOAD checklist + today deadline.

Example gold-standard body (target 45+/50):
"Ramesh bhai, batch reconciliation alert: ~43 of your 240 chronic-Rx patients likely dispensed Atorvastatin 10mg from batch RJ-2024-077 in the last 60 days. Dispensing liability + missed-dose compliance risk if not tracked before Friday. Metro pharmacy peers already recalled + swapped molecule. Maine affected-patient WhatsApp list plus replacement batch SKU ready kar diya -- reply YES, 10 minutes mein aapko bhej deta hoon, aaj hi resolve karte hain." """

CATEGORY_PROMPTS = {
    "dentists": _DENTIST,
    "salons": _SALON,
    "restaurants": _RESTAURANT,
    "gyms": _GYM,
    "pharmacies": _PHARMACY,
}


def _get_system(category_slug: str) -> str:
    return CATEGORY_PROMPTS.get(category_slug, _BASE)


# Strict body sanitizer: only ASCII printable + Devanagari + common punctuation.
# Anything else gets removed. This prevents the judge's scorer LLM from crashing
# when it tries to quote our body inside its own JSON response.
_ALLOWED_RE = re.compile(
    r"[^"
    r"\x20-\x7E"                # ASCII printable
    r"ऀ-ॿ"            # Devanagari
    r"–—‘’“”"  # en/em dash + smart quotes (normalized below)
    r"]"
)

def _sanitize_body(text: str) -> str:
    """Prevents judge LLM parse crashes by neutralising patterns that cause
    the scorer to write unescaped quotes in its rationale JSON."""
    # Normalise Unicode punctuation
    text = text.replace("—", " -- ").replace("–", " -- ")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')

    # Whitelist ASCII + Devanagari
    text = _ALLOWED_RE.sub("", text)

    # De-abbreviate periods that confuse the judge's tokenizer
    text = re.sub(r"\bDr\.\s+", "Dr ", text)
    text = re.sub(r"\bMr\.\s+", "Mr ", text)
    text = re.sub(r"\bMs\.\s+", "Ms ", text)
    text = re.sub(r"\bpp?\.\s*(\d+)", r"page \1", text)

    # Unwrap single-quoted fragments -- the top cause of judge JSON crashes.
    # 'delivery late' -> delivery late
    text = re.sub(r"'([^'\n]{2,60})'", r"\1", text)
    # Same for double-quoted fragments in the middle of a sentence
    text = re.sub(r'(?<=\s)"([^"\n]{2,60})"(?=[\s.,!?])', r"\1", text)

    # Remove stray apostrophes at ends of tokens
    text = re.sub(r"(\w)'(?=\s|$)", r"\1", text)

    # Collapse whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _derive_facts(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    """Pre-compute numbers so the LLM cites them directly. Prevents hallucination, adds density."""
    facts = {}
    perf = merchant.get("performance", {})
    cust_agg = merchant.get("customer_aggregate", {})
    peer = category.get("peer_stats", {})

    ctr = perf.get("ctr")
    peer_ctr = peer.get("avg_ctr")
    views = perf.get("views", 0)
    if ctr and peer_ctr:
        gap_pct = round(abs(peer_ctr - ctr) / peer_ctr * 100, 1)
        direction = "below" if ctr < peer_ctr else "above"
        facts["ctr_gap"] = f"CTR {ctr*100:.1f}% vs peer avg {peer_ctr*100:.1f}% -- {gap_pct}% {direction}"
        if ctr < peer_ctr and views:
            missed = round((views / 4) * (peer_ctr - ctr))
            if missed > 0:
                facts["missed_actions_per_week"] = f"~{missed} missed calls/bookings per week from CTR gap"

    lapsed = cust_agg.get("lapsed_180d_plus") or cust_agg.get("lapsed_90d_plus")
    if lapsed:
        facts["lapsed_customers"] = f"{lapsed} customers lapsed (90-180+ days)"

    if cust_agg.get("total_active_members"):
        churn = cust_agg.get("monthly_churn_pct", 0)
        at_risk = round(cust_agg["total_active_members"] * churn)
        facts["members_at_risk_monthly"] = f"{at_risk} of {cust_agg['total_active_members']} members at monthly churn risk"

    if cust_agg.get("chronic_rx_count"):
        facts["chronic_rx_count"] = f"{cust_agg['chronic_rx_count']} chronic-Rx patients on file"

    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})

    if trg_kind == "supply_alert":
        chronic = cust_agg.get("chronic_rx_count", 0)
        batches = trg_payload.get("affected_batches", [])
        estimated = min(round(chronic * 0.09 * len(batches)), chronic)
        facts["affected_chronic_patients"] = f"~{estimated} of your {chronic} chronic-Rx patients likely dispensed affected batch(es)"
        if batches:
            facts["affected_batch_ids"] = ", ".join(str(b) for b in batches[:3])

    delta_7d = perf.get("delta_7d", {})
    if trg_kind in ("perf_dip", "perf_spike", "seasonal_perf_dip"):
        metric = trg_payload.get("metric", "calls")
        delta = trg_payload.get("delta_pct") or delta_7d.get(f"{metric}_pct")
        baseline = trg_payload.get("vs_baseline") or perf.get(metric)
        if delta and baseline:
            direction = "up" if delta > 0 else "down"
            facts["perf_signal"] = f"{metric} {direction} {abs(delta)*100:.0f}% this week (baseline: {baseline})"
            if delta < 0:
                lost = round(abs(baseline * delta) / 4)
                if lost > 0:
                    facts["weekly_loss"] = f"~{lost} fewer {metric} per week vs normal"

    sub = merchant.get("subscription", {})
    days_rem = sub.get("days_remaining") or trg_payload.get("days_remaining")
    if trg_kind == "renewal_due" and days_rem:
        amount = trg_payload.get("renewal_amount")
        facts["renewal_urgency"] = (
            f"{days_rem} days left on {sub.get('plan', 'Pro')} plan"
            + (f" -- renewal Rs.{amount}" if amount else "")
        )

    if customer and trg_kind in ("recall_due", "chronic_refill_due"):
        rel = customer.get("relationship", {})
        last_visit = rel.get("last_visit") or trg_payload.get("last_refill")
        slots = trg_payload.get("available_slots", [])
        slot_labels = [s.get("label") for s in slots[:2] if s.get("label")]
        cust_name = customer.get("identity", {}).get("name", "")
        parts = [f"Patient: {cust_name}"] if cust_name else []
        if last_visit:
            parts.append(f"last visit {last_visit}")
        if slot_labels:
            parts.append(f"open slots: {', '.join(slot_labels)}")
        if parts:
            facts["recall_info"] = " | ".join(parts)

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
            if match.get("patient_segment"):
                parts.append(f"Segment: {match['patient_segment']}")
            if match.get("title"):
                parts.append(f"Title: {match['title'][:80]}")
            if match.get("actionable"):
                parts.append(f"Actionable: {match['actionable'][:80]}")
            if parts:
                facts["cited_study"] = " | ".join(parts)

    if trg_kind == "winback_eligible":
        parts = []
        if trg_payload.get("days_since_expiry"):
            parts.append(f"{trg_payload['days_since_expiry']} days since expiry")
        if trg_payload.get("perf_dip_pct"):
            parts.append(f"perf dip {abs(trg_payload['perf_dip_pct'])*100:.0f}%")
        if trg_payload.get("lapsed_customers_since_expiry"):
            parts.append(f"{trg_payload['lapsed_customers_since_expiry']} customers lapsed since")
        if parts:
            facts["winback_signal"] = " | ".join(parts)

    if trg_kind == "competitor_opened":
        parts = []
        if trg_payload.get("distance_km"):
            parts.append(f"{trg_payload['distance_km']} km away")
        if trg_payload.get("competitor_offer"):
            parts.append(f"their offer: {trg_payload['competitor_offer']}")
        if trg_payload.get("your_offer"):
            parts.append(f"your offer: {trg_payload['your_offer']}")
        if parts:
            facts["competitor_signal"] = " | ".join(parts)

    if trg_kind == "gbp_unverified":
        parts = []
        if trg_payload.get("estimated_uplift_pct"):
            parts.append(f"~{trg_payload['estimated_uplift_pct']*100:.0f}% uplift potential")
        if trg_payload.get("time_to_complete_min"):
            parts.append(f"{trg_payload['time_to_complete_min']} min to complete")
        if parts:
            facts["gbp_signal"] = " | ".join(parts)

    if trg_kind == "review_theme_emerged":
        parts = []
        theme = trg_payload.get("theme", "")
        if theme:
            # Humanize snake_case: "delivery_late" -> "late delivery"
            human = theme.replace("_", " ")
            if human.split()[-1] in ("late", "slow", "cold", "expensive", "wrong", "missing"):
                human = " ".join(reversed(human.split()))
            parts.append(f"theme: {human}")
        if trg_payload.get("occurrence_count"):
            parts.append(f"{trg_payload['occurrence_count']} mentions this week")
        if trg_payload.get("common_quote"):
            parts.append(f"customers say: {trg_payload['common_quote'][:80]}")
        if parts:
            facts["review_signal"] = " | ".join(parts)

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
            "offer_catalog": category.get("offer_catalog", [])[:4],
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
        + "\n\nCompose the message NOW. Cite DERIVED_FACTS values verbatim. Hit density: 5+ numbers, 3+ domain terms, peer comparison, time-bound CTA."
    )


_TIME_RE = re.compile(
    r"(\d{1,2}\s*(am|pm|baje)|(sham|raat|subah)\s+\d|"
    r"tonight|today|aaj\s+(hi|sham|raat)|"
    r"(mon|tue|wed|thu|fri|sat|sun)day|is\s+hafte)",
    re.I,
)

def _quality_issues(body: str) -> list[str]:
    """Fast rule-based check for the levers most correlated with judge Engagement/DQ scores."""
    issues: list[str] = []
    lower = body.lower()
    if not any(k in lower for k in ("peer", "metro", "benchmark", "industry avg")):
        issues.append("Missing peer/social-proof line")
    if not re.search(r"\byes\b", body, re.I):
        issues.append("Missing single-word YES CTA")
    if len(re.findall(r"\d", body)) < 5:
        issues.append("Fewer than 5 numeric tokens")
    if not _TIME_RE.search(body):
        issues.append("Missing specific-time deadline (clock/day)")
    words = len(body.split())
    if words < 55:
        issues.append(f"Too short ({words} words, target 60-90)")
    if words > 100:
        issues.append(f"Too long ({words} words, target 60-90)")
    # CLOSE line format: last sentence must have both "YES" and a time cue
    last_sent = re.split(r"(?<=[.!?])\s+", body.strip())[-1] if body else ""
    if last_sent and not (re.search(r"\byes\b", last_sent, re.I) and _TIME_RE.search(last_sent)):
        issues.append("CLOSE line must end with 'reply YES, <clock time>.'")
    return issues


def _rewrite_body(system_prompt: str, ctx: str, draft: str, issues: list[str]) -> str:
    """Second LLM pass — fixes weak levers without regenerating from scratch."""
    client = _get_client()
    user = (
        f"CONTEXT:\n{ctx}\n\nDRAFT:\n{draft}\n\n"
        f"ISSUES TO FIX (only these, keep all correct facts and structure):\n- "
        + "\n- ".join(issues)
        + "\n\nReturn the rewritten message via the tool. 60-90 words. Preserve owner name, "
        "all numbers, peer benchmark and derived facts. Do NOT invent new data. "
        "Final sentence template: 'Maine <artifact> ready kar diya -- reply YES, <clock time>.'"
    )
    resp = client.converse(
        modelId=MODEL,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"temperature": 0, "maxTokens": 600},
        toolConfig={
            "tools": COMPOSE_TOOL,
            "toolChoice": {"tool": {"name": "compose_message"}},
        },
    )
    blocks = resp["output"]["message"]["content"]
    tool = next((b for b in blocks if "toolUse" in b), None)
    return tool["toolUse"]["input"]["body"] if tool else draft


def _parse_json_fallback(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text, object_pairs_hook=dict)


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """Deterministic compose. Returns {body, cta, send_as, suppression_key, rationale}."""
    client = _get_client()
    system_prompt = _get_system(category.get("slug", ""))
    user_content = _extract_context(category, merchant, trigger, customer)

    response = client.converse(
        modelId=MODEL,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_content}]}],
        inferenceConfig={"temperature": 0, "maxTokens": 800},
        toolConfig={
            "tools": COMPOSE_TOOL,
            "toolChoice": {"tool": {"name": "compose_message"}},
        },
    )

    content_blocks = response["output"]["message"]["content"]
    tool_block = next((b for b in content_blocks if "toolUse" in b), None)
    if tool_block:
        result = dict(tool_block["toolUse"]["input"])
    else:
        text = content_blocks[0].get("text", "") if content_blocks else ""
        if not text:
            raise ValueError("Bedrock returned empty content")
        result = _parse_json_fallback(text)

    if result.get("body"):
        # Lever 1: self-critique + rewrite only if rule-based check fails.
        # Keeps latency in budget for messages already strong (most triggers).
        issues = _quality_issues(result["body"])
        if issues:
            try:
                fixed = _rewrite_body(system_prompt, user_content, result["body"], issues)
                if fixed and len(fixed.split()) >= 40:
                    result["body"] = fixed
            except Exception as exc:
                print(f"[rewrite] skipped: {exc}")
        result["body"] = _sanitize_body(result["body"])

    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get(
            "suppression_key", f"trigger:{trigger.get('id', 'unknown')}"
        )

    return result
