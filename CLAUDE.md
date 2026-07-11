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
