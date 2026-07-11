"""
Vera Bot — magicpin AI Challenge
FastAPI server implementing the 5-endpoint judging contract.

Start: uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import time
import hashlib
import os
from datetime import datetime, timezone
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from composer import compose
from conversation import reply as conv_reply

app = FastAPI(title="Vera Bot")
START = time.time()

# ── In-memory stores ──────────────────────────────────────────────────────────

# (scope, context_id) → {version: int, payload: dict}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id → list of {from_role, msg, ts}
conversations: dict[str, list[dict]] = {}

# suppression_key → sent_at_unix (float); TTL = SUPPRESS_TTL_SECONDS
sent_keys: dict[str, float] = {}

# merchant_id → last message body hash (anti-repetition)
last_body: dict[str, str] = {}

# Per judge run, short TTL prevents collision between back-to-back runs
SUPPRESS_TTL_SECONDS = 120  # 2 min — keeps test runs clean

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _body_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def _is_suppressed(key: str) -> bool:
    sent_at = sent_keys.get(key)
    if sent_at is None:
        return False
    return (time.time() - sent_at) < SUPPRESS_TTL_SECONDS

def _suppress(key: str):
    sent_keys[key] = time.time()

def _get_ctx(scope: str, ctx_id: str) -> dict | None:
    entry = contexts.get((scope, ctx_id))
    return entry["payload"] if entry else None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Engine",
        "team_members": ["Rajesh"],
        "model": "claude-haiku-4-5-20251001",
        "approach": (
            "LLM composer (temperature=0) with structured prompt engineering. "
            "Extracts key signals from all 4 context layers, builds a focused "
            "context block, and asks Claude to compose with hard specificity + voice constraints."
        ),
        "contact_email": "rajeshrkk112003@gmail.com",
        "version": "1.0.0",
        "submitted_at": "2026-07-12T00:00:00Z",
    }


class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

@app.post("/v1/context")
async def push_context(body: ContextBody):
    key = (body.scope, body.context_id)
    current = contexts.get(key)

    if current and current["version"] >= body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current["version"],
        }

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": _now_iso(),
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        # Guard: max 20 actions per tick
        if len(actions) >= 20:
            break

        trg = _get_ctx("trigger", trg_id)
        if not trg:
            continue

        # Suppression check
        sup_key = trg.get("suppression_key", trg_id)
        if _is_suppressed(sup_key):
            continue

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")

        merchant = _get_ctx("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue

        category_slug = merchant.get("category_slug")
        category = _get_ctx("category", category_slug) if category_slug else None
        if not category:
            continue

        customer = _get_ctx("customer", customer_id) if customer_id else None

        try:
            result = compose(category, merchant, trg, customer)
        except Exception as exc:
            # Compose failed — skip this trigger, don't crash the tick
            print(f"[tick] compose failed for {trg_id}: {exc}")
            continue

        body_text = result.get("body", "")
        if not body_text:
            continue

        # Anti-repetition: skip if identical to last message sent to this merchant
        bh = _body_hash(body_text)
        if last_body.get(merchant_id) == bh:
            continue
        last_body[merchant_id] = bh

        # Mark suppressed
        _suppress(sup_key)

        conv_id = f"conv_{merchant_id}_{trg_id}"

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
            "template_params": [
                merchant.get("identity", {}).get("name", ""),
                trg.get("kind", ""),
                body_text[:40],
            ],
            "body": body_text,
            "cta": result.get("cta", "open_ended"),
            "suppression_key": result.get("suppression_key", sup_key),
            "rationale": result.get("rationale", ""),
        })

    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    # Store this turn
    history = conversations.setdefault(body.conversation_id, [])
    history.append({
        "from_role": body.from_role,
        "msg": body.message,
        "ts": body.received_at,
    })

    result = conv_reply(
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id,
        customer_id=body.customer_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number,
        history=history,
        contexts=contexts,
    )

    # Store bot's reply if we're sending
    if result.get("action") == "send":
        history.append({
            "from_role": "vera",
            "msg": result.get("body", ""),
            "ts": _now_iso(),
        })
        # Anti-repetition for replies too
        if body.merchant_id:
            bh = _body_hash(result.get("body", ""))
            last_body[body.merchant_id] = bh

    return result


@app.post("/v1/teardown")
async def teardown():
    """Judge calls this at end of test — wipe all state."""
    contexts.clear()
    conversations.clear()
    sent_keys.clear()
    last_body.clear()
    return {"ok": True}
