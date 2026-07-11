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

Served over a live HTTP API that the judge harness calls: `/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata`.

---

## Scoring (5 dimensions × 10 = 50 total)

| Dimension | What kills your score |
|---|---|
| **Specificity** | "10% off" vs "Dental Cleaning @ ₹299". Generic = 4/10. Numbers from context = 9-10/10. |
| **Category fit** | Dentist message sounding like a retail promo. Use the category voice. |
| **Merchant fit** | Not using the owner's first name. Ignoring their actual numbers. Wrong language. |
| **Trigger relevance** | The message not explaining *why now*. Research digest trigger → mention the research. |
| **Engagement compulsion** | Two CTAs in one message. CTA not in last sentence. Generic "let me know" close. |

---

## Our Approach: LLM-in-hot-path with structured prompting

We use Claude (temperature=0 for determinism) with a carefully engineered prompt. The brief allows this and it's the right call because:

- **Templates hit a ceiling** on specificity. Real merchant data needs reasoning, not slot-filling.
- **Determinism is free** at temperature=0 — same input → same output every time.
- **Speed**: Claude Haiku responds in ~1-2s, well inside the 30s budget.
- **Adaptability**: when the judge injects new context mid-test (updated performance, new digest items), the LLM naturally uses the new data. A template system requires explicit slot-mapping for every new field.

**Tradeoff**: LLM calls cost money and can timeout under load. We mitigate with Haiku (cheap + fast) and a 25s hard timeout per compose call.

---

## Pipeline

```
POST /v1/context  →  store[scope][context_id] = {version, payload}
POST /v1/tick     →  for each active trigger:
                       1. Look up merchant + category context
                       2. Check suppression (skip if sent recently)
                       3. compose(category, merchant, trigger, customer?) via LLM
                       4. Check anti-repetition (skip if body == last sent to merchant)
                       5. Return action
POST /v1/reply    →  classify merchant reply → send/wait/end
```

---

## How We Improve Accuracy

### 1. Context compression (what goes in the prompt)
Don't dump the full JSON into the prompt. Extract the signals that drive decision quality:
- Merchant: name, city, CTR vs peer median, active offers, top signal, conversation history tail
- Category: voice rules, top 2 digest items, peer stats, seasonal beats
- Trigger: kind + full payload (the "why now")
- Customer: name, state, language pref, preferred slots

### 2. Prompt constraints (enforced in system prompt)
- "Use ONLY numbers from the given context" → prevents hallucination
- "Single CTA in the last sentence" → prevents multi-CTA anti-pattern
- "service+price format beats X% off" → drives specificity
- "If languages includes 'hi', mix Hindi naturally" → drives merchant fit
- Voice map per category spelled out → drives category fit

### 3. Suppression + anti-repetition
- Suppress by `suppression_key` — don't re-send the same trigger's message
- Also suppress by merchant + message body hash — don't repeat identical text
- Short TTL (2 min) during testing; production would use 24h+

### 4. Multi-turn reply intelligence
Reply handler classifies merchant message:
- **Auto-reply**: repeated verbatim canned message → `action: end` gracefully
- **Explicit intent** ("yes", "ok let's do it", "send me") → immediately advance, no re-qualifying
- **Not interested** / hostile → `action: end` politely
- **Question** → answer then re-offer next step
- **Unknown** → ask one clarifying question, don't pad

### 5. Rationale field discipline
Judge cross-checks rationale against the body. Keep rationale as:
`"<trigger signal chosen> → <why this merchant right now>"`

Example: `"research_digest: JIDA fluoride trial anchors to her 124 high-risk adults; CTR below peer = clear growth lever"`

---

## Key Anti-patterns (cost -2 per occurrence)

- Multiple CTAs in one message
- CTA not in the last sentence
- Inventing data not in context
- Long preamble ("I hope you're having a great day...")
- Re-introducing Vera after first message
- Same body sent twice to same merchant
- Sending when nothing relevant to say (spam penalty)

---

## Running the Judge

```bash
# 1. Start the bot
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn bot:app --host 0.0.0.0 --port 8080

# 2. Generate the expanded dataset (run once)
cd magicpin-ai-challenge
python dataset/generate_dataset.py --seed-dir dataset --out expanded

# 3. Run the judge simulator
# Edit judge_simulator.py: set BOT_URL, LLM_PROVIDER, LLM_API_KEY
python judge_simulator.py
```

Track the **weakest scoring dimension** after each judge run and fix the prompt for that dimension specifically. Don't rewrite broadly.

---

## File Map

| File | Purpose |
|---|---|
| `bot.py` | FastAPI server — all 5 endpoints, in-memory stores |
| `composer.py` | `compose()` — LLM prompt building + structured output |
| `conversation.py` | Multi-turn reply classification + state machine |
| `requirements.txt` | Dependencies |
