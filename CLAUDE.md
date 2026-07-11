# Vera Engine — magicpin AI Challenge
> **Interview reference** — yeh file har iteration ke baad update hoti hai. Isme problem, approach aur fixes simple language mein likhe hain.

---

## Problem kya hai? (Simple explanation)

magicpin ke ~1 lakh merchant partners hain — restaurants, salons, dentists, gyms, pharmacies. Unke paas **Vera** naam ka WhatsApp AI assistant hai jo merchants ko daily message karta hai:
- Google profile improve karo
- Campaign chalaao
- Customer questions answer karo

**Challenge:** Vera ka "brain" banao — ek function jo 4 inputs leke best WhatsApp message compose kare.

```
compose(category, merchant, trigger, customer?) → message
```

| Input | Kya hota hai isme |
|---|---|
| `category` | Business type ke rules — doctor ka tone alag, salon ka alag |
| `merchant` | Is specific shop ki details — name, city, CTR%, active offers, signals |
| `trigger` | Abhi message kyo? — calls -40% hua, research paper aaya, festival 4 din mein |
| `customer` | Agar merchant ki taraf se kisi customer ko message — uska naam, language, last visit |

Output:
```json
{ "body": "...", "cta": "binary/open_ended/none", "send_as": "vera/merchant_on_behalf",
  "suppression_key": "...", "rationale": "..." }
```

Yeh sab ek **live HTTP API** ke through judge harness call karta hai: `/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata`

---

## Scoring (5 dimensions × 10 = 50 total)

| Dimension | Kya galat hota hai | Kya sahi hota hai |
|---|---|---|
| **Specificity** | "boost your sales", "10% off", "kaafi din ho gaye" | Real numbers: "CTR 2.1% vs peer 3.0%", "78 lapsed patients", "JIDA Oct p.14" |
| **Category fit** | Dentist ko retail promo jaisa message | Clinical vocab, peer tone, source citations |
| **Merchant fit** | Owner ka naam nahi, unke actual numbers nahi | "Meera, aapke 78 lapsed patients..." — name + real data + Hindi mix |
| **Trigger relevance** | Generic nudge — trigger se connected nahi | "Calls -50% this week vs baseline 12" — trigger payload ka data |
| **Engagement compulsion** | 2 CTAs, CTA last mein nahi, "let me know" close | Loss aversion + single "Reply YES" at the end |

---

## Hamara approach: LLM (temperature=0) via OpenRouter

Claude Haiku use karte hain OpenRouter ke through, carefully engineered prompt ke saath.

**Kyo LLM, templates nahi?**
- Templates ka ceiling hota hai — har merchant+trigger combination ke liye alag template nahi likh sakte
- temperature=0 pe determinism free hai — same input → same output har baar
- Haiku respond karta hai ~1-2s mein, judge ka 15s budget kaafi hai
- Jab judge mid-test naya context inject kare, LLM automatically use kar leta hai — templates ko manual slot-mapping chahiye

**Tradeoff:** LLM calls costly hain aur timeout ho sakti hain. Mitigation: Haiku (cheapest), aur har trigger ke liye try/except (bot crash nahi hoga, sirf trigger skip hoga).

---

## Architecture

```
POST /v1/context  →  store[(scope, context_id)] = {version, payload}

POST /v1/tick     →  PARALLEL mein sab triggers process karo:
                       asyncio.gather([compose(trigger_1), compose(trigger_2), ...])
                       → ek saath sab LLM calls, 3s mein done (sequential 15s+ tha)

POST /v1/reply    →  Reply classify karo → auto-reply/intent/hostile → send/wait/end
```

---

## Iteration Log — Kya fix kiya, kyo kiya

### Iteration 0 — Pehla working pass
- Basic FastAPI server with all 5 endpoints
- Simple compose prompt with JSON output
- `TEST_SCENARIO = "all"` — sirf operational tests (warmup, auto-reply, intent, hostile)
- **Problem: message scoring hi nahi ho raha tha**

### Iteration 1 — Full evaluation mode discover kiya
**Kya hua:** Friend ka bot `full_evaluation` use kar raha tha — jo `/v1/tick` call karta hai aur har composed message ko score karta hai. Hamara `"all"` sirf operational tests tha, message quality test hi nahi tha.

**Friend ka score:** 34/50 (68%) average
- **Weakness:** Generic dormancy messages scoring 29-36/50
- Reason: Yeh messages bas kehte the "N din ho gaye baat kiye" — merchant ke actual business data ka use nahi tha

**Kya fix kiya:**
1. `TEST_SCENARIO = "full_evaluation"` set kiya
2. Compose prompt completely rewrite kiya:
   - Har trigger kind ke liye specific strategy (research_digest, perf_dip, dormant, etc.)
   - "≥2 concrete facts" requirement — koi bhi generic line nahi
   - CTR gap pre-compute kiya context mein — built-in specificity anchor
   - Dormancy fix: "sirf 'N days ago' mat kaho — merchant ke strongest signal pe pivot karo"

### Iteration 2 — 3 critical bugs fix kiye (YEH WALA ITERATION)

**Bug 1: Tick timeout** ❌
- **Kya tha:** `/v1/tick` mein 5 triggers sequentially process hote the
- **Calculation:** 5 triggers × 3s/LLM call = 15s — judge ka timeout bhi 15s hai → timeout!
- **Fix:** `asyncio.gather()` use kiya — ab saare triggers parallel mein process hote hain
- **Result:** 5 triggers → ~3s total (5× faster)

```python
# PEHLE (sequential — timeout hota tha)
for trg_id in triggers:
    result = compose(...)  # 3s wait
    
# AB (parallel — 3s mein sab ho jaata hai)
results = await asyncio.gather(
    *[_process_trigger(tid) for tid in triggers]
)
```

**Bug 2: JSON parse error** ❌
- **Kya tha:** LLM output mein control characters aate the (e.g. `\x0b`) jo JSON.loads() crash karte the
- **Error:** `Invalid control character at: line 2 column 188`
- **Fix:** Ek regex add kiya jo JSON parse karne se pehle control chars strip kare:
```python
raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)
```

**Bug 3: LLM_MODEL wrong tha** ❌
- **Kya tha:** `judge_simulator.py` mein `LLM_MODEL = "anthropic/claude-3-haiku"` set tha
- **Yeh kya hai:** Yeh JUDGE ka model hai (scoring ke liye), hamara bot ka nahi
- **Friend ka config:** `LLM_MODEL = ""` (empty — OpenRouter apna default use karta hai)
- **Fix:** `LLM_MODEL = ""` set kiya

---

## Good vs Bad Examples (interview mein samjhao)

**BAD (friend ka dormancy message — 29/50):**
> "Hi Padma, 79 din ho gaye baat kiye, aur bina kaam ke profile chalti rehti hai..."

Kya galat: Koi real fact nahi. Har merchant ko same message. Trigger relevance zero.

**GOOD (case study — 50/50):**
> "Dr. Meera, JIDA's Oct issue landed. 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Relevant to your 124 high-risk adult patients — want me to pull it + draft a patient-ed WhatsApp?"

Kya sahi: 3 real numbers (2100, 38%, 124), source citation, merchant ki specific patient cohort, single CTA, curiosity lever.

---

## Anti-patterns (har ek pe -2 penalty)

- Ek message mein 2 CTAs
- CTA last sentence mein nahi
- Context mein jo nahi hai wo invent karna (fake research, fake competitor names)
- Long preamble ("I hope you're doing well...")
- First message ke baad re-introduction
- Same body twice to same merchant
- Generic dormancy message instead of merchant ke real signals

---

### Iteration 3 — Score 76% → target 90%+ (YEH WALA ITERATION)

**Score analysis after Iteration 2:**
- Specificity: 8/10 ✓
- Category Fit: 7/10 ← WEAKEST (generic tone, wrong vocab)
- Merchant Fit: 8/10 ✓
- Decision Quality: 8/10 ✓
- Engagement: 7/10 ← WEAKEST (asking "Want me to?" instead of "Maine kar diya")

**Bug 4: Race condition — same message sent twice to Anjali** ❌
- **Kya tha:** `asyncio.gather()` se parallel compose → dono triggers ne body-hash check kiya BEFORE either wrote to `last_body`
- **Result:** Same CTR message Anjali ko twice gaya, judge ne -2 penalty diya
- **Fix:** `tick()` mein gather ke baad `seen_merchants` set se dedup karo

**Bug 5: JSON control chars still failing on 3 triggers** ❌
- **Kya tha:** Regex se strip karna enough nahi tha — chars JSON string values ke andar the
- **Fix:** `json.loads(raw, strict=False)` — Python ka built-in permissive parser, control chars allow karta hai strings mein

**Bug 6: NoneType on trg_022** ❌  
- **Kya tha:** LLM ne `None` content return kiya (empty response)
- **Fix:** `if not content: raise ValueError(...)` — explicit check, clean error

**Category Fit 7→9 fix:**
- System prompt mein per-category vocabulary checklist add kiya
- Model upgrade: `claude-3-haiku` → `claude-3.5-haiku` (better instruction following)
- "Use ≥2 words/phrases from this list" — mandatory vocabulary enforcement

**Engagement 7→9 fix:**
- "Want me to do X?" se shift kiya "Maine X ready kar diya — sirf YES"
- 4 engagement formulas diye: PRE-LOAD / LOSS ANCHOR / CURIOSITY GAP / BINARY COMMIT
- LLM ko force kiya ek specific formula pick karne ke liye

**Derived facts pre-computation added (`_derive_facts()`):**
- CTR gap % (e.g. "30% below peer avg") — LLM ko calculate nahi karna
- Lapsed customer count — direct cite
- Members at churn risk = total_members × monthly_churn_pct
- Affected chronic patients for supply_alert = chronic_rx_count × 0.09 × batch_count
- Renewal urgency string
- Recall slot info

Yeh derived facts context mein explicitly pass karte hain taaki LLM inhe directly cite kare — no arithmetic, no hallucination.

---

### Iteration 4 — Prompt sharpening + few-shot examples + digest pre-lookup

**Research findings (internet se):**
- claude-3-haiku ka realistic ceiling: 80-85% (95%+ ke liye GPT-4o ya Claude 3.5 Sonnet chahiye)
- Sabse high-ROI technique small LLMs ke liye: **few-shot examples** (model dekhe ke sirf padhne se achha siikhta hai)
- **Aggressive language** ("CRITICAL!", "NEVER EVER", sab caps) Claude models pe ulta effect karta hai — instruction following kamzor hoti hai
- Pre-loaded derived numbers hallucination prevent karte hain (model ko arithmetic nahi karni)

**Kya fix kiya:**

**Fix 1: Few-shot examples add kiye** ✅
- 2 perfect 50/50 reference messages directly SYSTEM_PROMPT mein daale
- Example 1: dentist + research_digest (exact journal cite, patient count, PRE-LOAD CTA)
- Example 2: pharmacy + supply_alert (batch number, affected count, urgency)
- Effect: Model ab pattern follow karta hai instead of generic structure banana

**Fix 2: Digest item pre-lookup in `_derive_facts()`** ✅
- Pehle: LLM "JIDA's latest study" likhta tha (vague)
- Ab: `trigger.payload.top_item_id` se exact digest item dhundha jaata hai
- `facts["cited_study"]` mein: Source, n=, delta%, Title, Link — sab pre-extracted
- LLM directly cite kar sakta hai without having to parse nested JSON

```python
# Pehle: LLM ko khud dhundna padta tha digest mein
# Ab: pre-lookup
if trg_kind == "research_digest":
    top_item_id = trg_payload.get("top_item_id")
    match = next((d for d in digest_items if d.get("id") == top_item_id), None)
    if match:
        facts["cited_study"] = f"Source: {match['source']} | n={match['trial_n']} | delta={match['delta_pct']}%"
```

**Fix 3: SYSTEM_PROMPT simplify kiya** ✅
- Aggressive caps headings ("━━━ STEP 1 ━━━", "HARD RULES") remove kiye
- Ek clear list mein sab rules — cleaner instruction following
- Kam verbose = model focus karta hai content pe, formatting pe nahi

**OpenRouter API key aur accuracy ka kya rishta hai?**
> **Short answer: Koi rishta nahi.** API key tier (free vs paid) model quality affect nahi karta — sirf rate limits change hote hain. Score improve karna ho toh **model upgrade** karo, ya **prompt** improve karo.

---

### Iteration 5 — Switch to AWS Bedrock Nova Lite + 3 prompt fixes

**Kyo switch kiya OpenRouter → Bedrock?**
- Claude Haiku ka ceiling ~80-85% tha — is se upar jaana mushkil tha
- Amazon Nova Lite: faster, cheaper, aur better instruction following for structured JSON
- User ke paas AWS credentials the (`amazon.nova-lite-v1:0` model)
- boto3 Converse API use ki — cleaner than raw HTTP, same async pattern (asyncio.to_thread)

**Kya fix kiya:**

**Fix 1: Model upgrade** ✅
- `anthropic/claude-3-haiku` (OpenRouter) → `amazon.nova-lite-v1:0` (AWS Bedrock)
- composer.py + conversation.py dono update kiye
- boto3 `client.converse()` use kiya — system aur user messages clean separation

**Fix 2: Few-shot example names fix** ✅
- Pehle: examples mein "Dr. Meera Nair" tha — actual test merchant bhi "Meera" hai
- Problem: model ne example ko verbatim copy kar diya → judge ka LLM parse fail hua → 30/50 (fake score)
- Fix: Example names change kiye to "Dr. Priya", "Sharma bhai", "Rekha" — test data se alag

**Fix 3: Dormant CTR example add kiya** ✅
- Salon ka dormant_with_vera example add kiya (CTR gap → missed bookings → PRE-LOAD)
- "49% below peer = ~60 missed bookings/week" — quantified impact
- Ab 3 few-shot examples: dentist research, pharmacy supply_alert, salon dormant

**Fix 4: Warning added in prompt** ✅
- "Do NOT copy the examples — generate fresh for this specific context"
- Prevents model from templating off examples instead of using actual data

**Bedrock API format:**
```python
import boto3
client = boto3.client('bedrock-runtime', region_name='us-east-1', ...)
response = client.converse(
    modelId='amazon.nova-lite-v1:0',
    system=[{'text': SYSTEM_PROMPT}],
    messages=[{'role': 'user', 'content': [{'text': user_content}]}],
    inferenceConfig={'temperature': 0, 'maxTokens': 600}
)
content = response['output']['message']['content'][0]['text']
```

**.env mein add karo (never commit):**
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
BEDROCK_NOVA_MODEL_ID=amazon.nova-lite-v1:0
```

---

### Iteration 6 — Category-specific prompts + Tool use JSON + Body sanitization

**Root cause analysis of 37/50 (74%):**
- 2 messages scored 30/50 — those are FAKE. Judge's LLM crashed (parse error / choices error)
- Without those 2: (all other scores) / 12 = ~83.5%
- Real problem: judge's scorer LLM fails on messages with em dashes, smart quotes, complex mixed-script
- Category Fit stuck at 7 — one giant prompt serves all 5 categories, model gets confused

**Fix 1: Category-specific system prompts** ✅
- 5 separate prompts: `_DENTIST`, `_SALON`, `_RESTAURANT`, `_GYM`, `_PHARMACY`
- Each has: exact voice tone, mandatory vocab list, category-specific trigger strategies, 1 perfect example
- Selected at runtime: `_get_system(category.slug)`
- Effect: model only reads rules relevant to THIS category — Category Fit 7→9

**Fix 2: Nova Lite Tool Use for guaranteed JSON** ✅
- Defined `COMPOSE_TOOL` with exact JSON schema (body, cta, send_as, suppression_key, rationale)
- `toolChoice: {"tool": {"name": "compose_message"}}` — forces model to fill schema
- Extract from `response["output"]["message"]["content"][toolUse]["input"]`
- Zero parse errors on our side — valid JSON guaranteed every time
- Same for `conversation.py` — `REPLY_TOOL` for next_action

**Fix 3: Body sanitization** ✅
- `_sanitize_body()` runs AFTER getting AI response, BEFORE returning
- Replaces: em dashes (—→ -), en dashes (–→ -), smart quotes
- Collapses double spaces
- Effect: judge's scorer LLM doesn't crash on our message content → no more fake 30/50

**Fix 4: Missing trigger kinds added** ✅
- regulation_change, cde_opportunity, wedding_package_followup, ipl_match_today,
  customer_lapsed_hard, trial_followup, category_seasonal — all now handled in category prompts

**Fix 5: Winback derived fact** ✅
- `_derive_facts()` now pre-computes winback_signal: days_since_expiry + perf_dip_pct + lapsed_customers_since_expiry

**How Tool Use works (interview explain karo):**
Instead of asking "return JSON", we give the model a "function" with exact parameter types.
Model is FORCED to call that function — output is always structured, typed, valid.
Like a typed API contract vs freeform string parsing.

```
Before: LLM returns text → we parse JSON → crash if malformed
After:  LLM fills tool schema → we read .input → always valid dict
```

---

## Commands

```bash
# Terminal 1 — Bot start karo
cd ~/Documents/dev/Projects/magicPin
source venv/bin/activate
uvicorn bot:app --host 0.0.0.0 --port 8080

# Terminal 2 — Judge run karo
source venv/bin/activate
python magicpin-ai-challenge/judge_simulator.py
```

**Har judge run ke baad:** Sabse weak dimension dekho, us file mein fix karo:

| Weak dimension | Kahan fix karna hai |
|---|---|
| Specificity | `composer.py` → SYSTEM_PROMPT trigger strategy table |
| Category fit | `composer.py` → voice rules section |
| Merchant fit | `composer.py` → `_extract_context()` mein aur fields add karo |
| Trigger relevance | `composer.py` → trigger-specific strategy |
| Engagement compulsion | `composer.py` → compulsion levers section |

---

## File Map

| File | Kya karta hai |
|---|---|
| `bot.py` | FastAPI server — 5 endpoints, parallel tick, in-memory stores |
| `composer.py` | `compose()` — LLM prompt + context extraction + JSON output |
| `conversation.py` | Multi-turn replies — auto-reply detect, intent route, LLM fallback |
| `requirements.txt` | fastapi, uvicorn, openai, python-dotenv |
| `.env` | `OPENROUTER_API_KEY` — git mein kabhi commit mat karo |
