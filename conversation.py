"""
Multi-turn conversation handling for /v1/reply.
Classifies merchant/customer replies and decides: send / wait / end.
"""

import json
import os
from openai import OpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-3-haiku"

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OPENROUTER_BASE,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
    return _client


# Auto-reply fingerprints — common WhatsApp Business canned replies
_AUTO_REPLY_PHRASES = [
    "aapki jaankari ke liye",
    "thank you for contacting",
    "i am an automated",
    "main ek automated",
    "हमारी टीम तक",
    "hamari team tak",
    "we will get back",
    "hum aapko jald",
]

def _is_auto_reply(message: str, history: list[dict]) -> bool:
    """Detect WhatsApp Business canned auto-replies."""
    lower = message.lower()
    # Phrase match
    if any(phrase in lower for phrase in _AUTO_REPLY_PHRASES):
        return True
    # Repeated verbatim (3+ times)
    recent = [t["msg"] for t in history[-5:] if t.get("from_role") == "merchant"]
    if recent.count(message) >= 2:
        return True
    return False


_INTENT_POSITIVE = ["yes", "haan", "ha ", "ok ", "okay", "sure", "go ahead", "chalega",
                    "karo", "send me", "bhejo", "share karo", "let's do", "proceed"]
_INTENT_NEGATIVE = ["no", "nahi", "nope", "not interested", "stop", "band karo",
                    "mat bhejo", "not now", "abhi nahi", "leave me", "unsubscribe"]
_INTENT_ACTION   = ["i want to join", "judrna hai", "sign me up", "register", "book karo",
                    "slot chahiye", "appointment", "book it", "confirm"]


def _classify_intent(message: str) -> str:
    """Returns: positive | negative | action | question | unknown"""
    lower = message.lower()
    if any(p in lower for p in _INTENT_ACTION):
        return "action"
    if any(p in lower for p in _INTENT_POSITIVE):
        return "positive"
    if any(p in lower for p in _INTENT_NEGATIVE):
        return "negative"
    if "?" in message or lower.startswith(("what", "how", "when", "where", "kya", "kab", "kaise", "kyun")):
        return "question"
    return "unknown"


REPLY_SYSTEM = """You are Vera, magicpin's WhatsApp AI for merchant growth.
The merchant/customer just replied to your previous message. Decide the next move.

RULES:
- If they said YES / agreed / want to proceed: deliver the promised artifact or next step immediately. Do NOT ask another qualifying question.
- If they asked a question: answer it concisely, then re-offer the next step.
- If they seem unsure: give one concrete reason to act, then a binary YES/STOP.
- If hostile or clearly uninterested: end gracefully, no pushback.
- NO preambles, NO re-introduction, NO generic "happy to help".
- Match the merchant's language (Hindi-English mix if they used it).
- Keep body under 80 words.

Return JSON only:
{
  "action": "send" | "wait" | "end",
  "body": "...",       // required if action=send
  "cta": "binary|open_ended|none",  // required if action=send
  "wait_seconds": 0,  // required if action=wait
  "rationale": "..."
}"""


def reply(
    conversation_id: str,
    merchant_id: str | None,
    customer_id: str | None,
    from_role: str,
    message: str,
    turn_number: int,
    history: list[dict],
    contexts: dict,
) -> dict:
    """
    Given a merchant/customer reply and conversation history, decide next action.
    Returns: {action, body?, cta?, wait_seconds?, rationale}
    """

    # Auto-reply detection — exit immediately after 1 gentle retry
    if from_role == "merchant" and _is_auto_reply(message, history):
        if turn_number > 2:
            return {
                "action": "end",
                "rationale": "Auto-reply detected — gracefully exiting to avoid burn turns",
            }
        # First time: try once more with a direct question
        return {
            "action": "send",
            "body": "Lagta hai yeh ek auto-reply hai 🙂 Kya aap ya aapki team ek minute le sakti hai? Quick check hai.",
            "cta": "open_ended",
            "rationale": "Possible auto-reply — one gentle check before exiting",
        }

    # Hard negative / exit signal
    intent = _classify_intent(message)
    if intent == "negative":
        return {
            "action": "end",
            "rationale": "Merchant signaled not interested — exiting gracefully",
        }

    # Immediate action intent — don't re-qualify, deliver
    if intent == "action":
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload", {}) if merchant_id else {}
        name = merchant.get("identity", {}).get("owner_first_name", "")
        greeting = f"{name}, " if name else ""
        return {
            "action": "send",
            "body": f"{greeting}perfect — let me pull that up for you right now. Give me 2 minutes.",
            "cta": "none",
            "rationale": "Explicit action intent detected — routing to fulfillment immediately",
        }

    # Max turns guard
    if turn_number >= 5:
        return {
            "action": "end",
            "rationale": "Reached 5 turns — closing this thread to avoid over-messaging",
        }

    # All other cases: ask the LLM to generate the right reply
    client = _get_client()

    # Build recent conversation context
    recent_turns = history[-4:] if len(history) >= 4 else history
    conv_str = "\n".join(
        f"[{t.get('from_role', t.get('from', 'unknown')).upper()}]: {t.get('msg', t.get('body', ''))}"
        for t in recent_turns
    )

    merchant = contexts.get(("merchant", merchant_id), {}).get("payload", {}) if merchant_id else {}
    merchant_name = merchant.get("identity", {}).get("owner_first_name", "")
    languages = merchant.get("identity", {}).get("languages", ["en"])

    user_content = f"""CONVERSATION SO FAR:
{conv_str}
[{from_role.upper()}]: {message}

MERCHANT: {merchant_name} | Languages: {languages}
TURN: {turn_number} of 5 max

What is Vera's best next move?"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        temperature=0,
        messages=[
            {"role": "system", "content": REPLY_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

    return json.loads(raw)
