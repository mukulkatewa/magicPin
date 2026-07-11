"""
Multi-turn conversation handling for /v1/reply.
Classifies merchant/customer replies and decides: send / wait / end.
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

MODEL = os.environ.get("BEDROCK_NOVA_MODEL_ID", "amazon.nova-pro-v1:0")


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
    lower = message.lower()
    if any(phrase in lower for phrase in _AUTO_REPLY_PHRASES):
        return True
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
The merchant/customer just replied. Decide the next move.

Rules:
- YES / agreed: deliver the promised artifact immediately. No re-qualifying questions.
- Question: answer concisely, then re-offer the next step.
- Unsure: one concrete reason to act, then binary YES/STOP.
- Hostile or uninterested: end gracefully, no pushback.
- No preambles, no re-introduction.
- Match merchant's language (Hindi-English if they used it).
- Body under 80 words."""

REPLY_TOOL = [{
    "toolSpec": {
        "name": "next_action",
        "description": "Return Vera's next action in the conversation",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["send", "wait", "end"]},
                    "body": {"type": "string", "description": "Message body if action=send"},
                    "cta": {"type": "string", "enum": ["binary", "open_ended", "none"]},
                    "wait_seconds": {"type": "integer", "description": "Seconds to wait if action=wait"},
                    "rationale": {"type": "string"},
                },
                "required": ["action", "rationale"],
            }
        },
    }
}]


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
    # Auto-reply detection
    if from_role == "merchant" and _is_auto_reply(message, history):
        if turn_number > 2:
            return {"action": "end", "rationale": "Auto-reply detected — gracefully exiting"}
        return {
            "action": "send",
            "body": "Lagta hai yeh ek auto-reply hai. Kya aap ya aapki team ek minute le sakti hai? Quick check hai.",
            "cta": "open_ended",
            "rationale": "Possible auto-reply — one gentle check before exiting",
        }

    intent = _classify_intent(message)
    if intent == "negative":
        return {"action": "end", "rationale": "Merchant signaled not interested — exiting gracefully"}

    if intent == "action":
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload", {}) if merchant_id else {}
        name = merchant.get("identity", {}).get("owner_first_name", "")
        greeting = f"{name}, " if name else ""
        return {
            "action": "send",
            "body": f"{greeting}perfect -- let me pull that up for you right now. Give me 2 minutes.",
            "cta": "none",
            "rationale": "Explicit action intent — routing to fulfillment immediately",
        }

    if turn_number >= 5:
        return {"action": "end", "rationale": "Reached 5 turns — closing thread"}

    # LLM fallback for unknown/positive/question
    client = _get_client()
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

    response = client.converse(
        modelId=MODEL,
        system=[{"text": REPLY_SYSTEM}],
        messages=[{"role": "user", "content": [{"text": user_content}]}],
        inferenceConfig={"temperature": 0, "maxTokens": 400},
        toolConfig={
            "tools": REPLY_TOOL,
            "toolChoice": {"tool": {"name": "next_action"}},
        },
    )

    content_blocks = response["output"]["message"]["content"]
    tool_block = next((b for b in content_blocks if "toolUse" in b), None)
    if tool_block:
        return tool_block["toolUse"]["input"]

    # Fallback text parse
    raw = content_blocks[0].get("text", "").strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
    return json.loads(raw, object_pairs_hook=dict)
