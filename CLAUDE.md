# Vera Engine — magicpin AI Challenge

## What is this project asking?

magicpin has ~100k merchant partners (restaurants, salons, dentists, gyms, pharmacies). They run **Vera**, a WhatsApp AI assistant that messages merchants daily to help them grow — improve their Google profile, run campaigns, answer customer questions.

The challenge: **build the brain behind Vera's messages**.

Given these 4 inputs, produce the perfect next WhatsApp message:

| Input | What it contains |
|---|---|
| `category` | Business type norms — voice, allowed vocabulary, offer patterns, competitor benchmarks, research digests |
| `merchant` | This specific business — name, city, performance metrics (views/calls/CTR), active offers, conversation history, signals |
| `trigger` | Why message NOW — a research digest dropped, their calls dipped 40%, a customer recall is due, a festival is 4 days away |
| `customer` (optional) | When messaging a merchant's customer on their behalf — name, language pref, last visit, consent |

Output per message:
```json
{ "body": "...", "cta": "binary|open_ended|none", "send_as": "vera|merchant_on_behalf",
  "suppression_key": "...", "rationale": "..." }
```

Served over a live HTTP API the judge harness calls: `/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata`.

---

## Scoring (5 dimensions × 10 = 50 total)

| Dimension | What kills your score | What wins |
|---|---|---|
| **Specificity** | "10% off", "boost your sales", "kaafi din ho gaye" | Real numbers: "CTR 2.1% vs peer 3.0%", "78 lapsed patients", "JIDA Oct p.14" |
| **Category fit** | Dentist message sounding like a retail promo | Clinical vocab, peer tone, source citations for dentists; operator language for restaurants |
| **Merchant fit** | Not using owner name. Ignoring their actual numbers. Wrong language | "Meera, aapke 78 lapsed patients…" — name + their real data + Hindi mix |
| **Trigger relevance** | Generic nudge not connected to the trigger event | "Calls -50% this week vs baseline 12" — exact trigger payload data |
| **Engagement compulsion** | Two CTAs. CTA not last. "Let me know if interested" | Loss aversion + single "Reply YES" at the end |

---

## Our Approach: LLM-in-hot-path (temperature=0)

We use Claude Haiku via OpenRouter with a carefully engineered prompt. The brief allows this.

**Why LLM over deterministic templates:**
- Templates hit a ceiling on specificity — can't pre-write every merchant+trigger combination
- Determinism is free at temperature=0 — same input → same output every run
- Speed: Haiku responds in ~1-2s, well inside the 30s budget
- Adaptability: when judge injects new context mid-test, LLM naturally uses it; templates require explicit slot-mapping for every new field

**Tradeoff:** LLM calls cost money and fail on bad API keys. Mitigated with Haiku (cheapest) and try/except around every compose call (bot never crashes, just skips the trigger).

---

## Pipeline

```
POST /v1/context  →  store[(scope, context_id)] = {version, payload}
POST /v1/tick     →  for each active trigger:
                       1. Look up merchant + category
                       2. Check suppression (skip if sent within 2min)
                       3. compose(category, merchant, trigger, customer?) via LLM
                       4. Check anti-repetition (skip if same body hash as last message)
                       5. Return action
POST /v1/reply    →  classify reply → auto-reply/intent/hostile → send/wait/end
```

---

## Iteration Log (how we improved, in order)

### Iteration 0 — First working pass
- Basic FastAPI server with all 5 endpoints
- Simple compose prompt asking for JSON output
- TEST_SCENARIO = "all" (only runs operational tests — warmup, auto-reply, intent, hostile)
- **Result: operational tests passing but no message scoring**

### Iteration 1 — Discovered missing full_evaluation
- Friend's bot ran `TEST_SCENARIO = "full_evaluation"` which calls `/v1/tick` + scores every composed message
- Our `"all"` scenario only tested operations, not message quality
- Friend's avg score: **34/50 (68%)**
- Friend's weakness: generic dormancy messages ("N din ho gaye baat kiye") scoring 29-36/50
  - These messages said nothing about the merchant's actual business state
  - Specificity 3-6/10 because no real facts were used

**Fix applied:**
1. Changed `TEST_SCENARIO = "full_evaluation"`
2. Completely rewrote compose prompt with:
   - Trigger-specific strategy table (what data to pull for each trigger kind)
   - Explicit "≥2 concrete facts" requirement
   - CTR gap pre-computed in context (`ctr_vs_peer_note`)
   - Specific dormancy fix: "DO NOT say N days since last message — pivot to their strongest signal"
   - All 5 category voices spelled out with vocabulary examples

### What good vs bad looks like (from case studies)

**BAD (friend's dormancy messages, 29-34/50):**
> "Hi Padma, 79 din ho gaye baat kiye, aur bina kaam ke profile chalti rehti hai..."

What's wrong: No merchant-specific facts. Could be sent to any merchant. No trigger relevance beyond "you haven't replied."

**GOOD (case study, 50/50):**
> "Dr. Meera, JIDA's Oct issue landed. 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Relevant to your 124 high-risk adult patients — worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp?"

What's right: 3 real numbers (2100, 38%, 124), source citation, merchant's specific patient cohort, single CTA, curiosity lever.

---

## Key Anti-patterns (cost -2 per occurrence)

- Multiple CTAs in one message
- CTA not in the last sentence  
- Inventing data not in context (especially fake research citations or competitor names)
- Long preamble ("I hope you're having a great day...")
- Re-introducing Vera after the first message
- Same body sent twice to same merchant
- Generic dormancy framing instead of pivoting to merchant's real signals

---

## Running the Judge

```bash
# Terminal 1 — Start the bot
cd ~/Documents/dev/Projects/magicPin
source venv/bin/activate
uvicorn bot:app --host 0.0.0.0 --port 8080

# Terminal 2 — Run judge (full scoring)
cd ~/Documents/dev/Projects/magicPin
source venv/bin/activate
python magicpin-ai-challenge/judge_simulator.py
```

After each judge run: look at the **weakest scoring dimension**, trace it to the right file:

| Weak dimension | Fix in |
|---|---|
| Specificity | `composer.py` SYSTEM_PROMPT — add more trigger-specific data extraction |
| Category fit | `composer.py` voice rules section |
| Merchant fit | `_extract_context()` — are we passing enough merchant fields? |
| Trigger relevance | SYSTEM_PROMPT trigger strategy table |
| Engagement compulsion | SYSTEM_PROMPT compulsion levers section |

---

## File Map

| File | Purpose |
|---|---|
| `bot.py` | FastAPI server — all 5 endpoints, in-memory stores, suppression, anti-repetition |
| `composer.py` | `compose()` — LLM prompt + context extraction + structured output |
| `conversation.py` | Multi-turn reply handler — auto-reply detection, intent routing, LLM fallback |
| `requirements.txt` | fastapi, uvicorn, openai, python-dotenv |
| `.env` | `OPENROUTER_API_KEY` — never committed to git |
