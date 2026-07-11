"""
Vera Bot — magicpin AI Challenge
FastAPI server implementing the 5-endpoint judging contract.

Start: uvicorn bot:app --host 0.0.0.0 --port 8080

KEY FIX: tick() processes all triggers in PARALLEL using asyncio.gather()
         so 5 LLM calls finish in ~3s instead of 5×3s = 15s (judge timeout = 15s)
"""

import time
import hashlib
import asyncio
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

contexts: dict[tuple[str, str], dict] = {}       # (scope, context_id) → {version, payload}
conversations: dict[str, list[dict]] = {}         # conv_id → [turns]
sent_keys: dict[str, float] = {}                  # suppression_key → sent_at timestamp
last_body: dict[str, str] = {}                    # merchant_id → last body hash

SUPPRESS_TTL = 120   # 2 min — short so back-to-back judge runs don't collide

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _body_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def _is_suppressed(key: str) -> bool:
    sent_at = sent_keys.get(key)
    return sent_at is not None and (time.time() - sent_at) < SUPPRESS_TTL

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
        "model": "bedrock/amazon.nova-pro-v1:0",
        "approach": (
            "LLM composer (temperature=0) with structured prompt. "
            "Parallel async tick — all triggers composed simultaneously. "
            "Trigger-specific strategy: each trigger kind has its own data extraction path."
        ),
        "contact_email": "rajeshrkk112003@gmail.com",
        "version": "2.0.0",
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


async def _process_trigger(trg_id: str) -> dict | None:
    """
    Compose one message for a trigger. Runs inside asyncio.gather() so all
    triggers in a batch are processed in parallel, not sequentially.
    Returns an action dict or None (skip silently).
    """
    trg = _get_ctx("trigger", trg_id)
    if not trg:
        return None

    sup_key = trg.get("suppression_key", trg_id)
    if _is_suppressed(sup_key):
        return None

    merchant_id = trg.get("merchant_id")
    customer_id = trg.get("customer_id")

    merchant = _get_ctx("merchant", merchant_id) if merchant_id else None
    if not merchant:
        return None

    category_slug = merchant.get("category_slug")
    category = _get_ctx("category", category_slug) if category_slug else None
    if not category:
        return None

    customer = _get_ctx("customer", customer_id) if customer_id else None

    try:
        # asyncio.to_thread runs the blocking LLM call in a thread pool
        # so it doesn't block the event loop — this is what makes parallelism work
        result = await asyncio.to_thread(compose, category, merchant, trg, customer)
    except Exception as exc:
        print(f"[tick] compose failed for {trg_id}: {exc}")
        return None

    body_text = result.get("body", "").strip()
    if not body_text:
        return None

    # Anti-repetition: skip if body is identical to last message we sent this merchant
    bh = _body_hash(body_text)
    if last_body.get(merchant_id) == bh:
        return None
    last_body[merchant_id] = bh
    _suppress(sup_key)

    return {
        "conversation_id": f"conv_{merchant_id}_{trg_id}",
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
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    trg_ids = body.available_triggers[:20]  # cap at 20 per spec

    # Run ALL trigger compositions in parallel
    # Sequential: 5 × 3s = 15s (judge times out). Parallel: ~3s total.
    results = await asyncio.gather(
        *[_process_trigger(tid) for tid in trg_ids],
        return_exceptions=True,
    )

    # Deduplicate by merchant_id — parallel compose can race on body-hash check.
    # Keep first valid action per merchant (preserves the best trigger ordering).
    actions = []
    seen_merchants: set[str] = set()
    for r in results:
        if not r or not isinstance(r, dict):
            continue
        mid = r.get("merchant_id")
        if mid in seen_merchants:
            continue
        seen_merchants.add(mid)
        actions.append(r)

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
    history = conversations.setdefault(body.conversation_id, [])
    history.append({
        "from_role": body.from_role,
        "msg": body.message,
        "ts": body.received_at,
    })

    result = await asyncio.to_thread(
        conv_reply,
        body.conversation_id,
        body.merchant_id,
        body.customer_id,
        body.from_role,
        body.message,
        body.turn_number,
        history,
        contexts,
    )

    if result.get("action") == "send":
        history.append({
            "from_role": "vera",
            "msg": result.get("body", ""),
            "ts": _now_iso(),
        })
        if body.merchant_id:
            last_body[body.merchant_id] = _body_hash(result.get("body", ""))

    return result


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_keys.clear()
    last_body.clear()
    return {"ok": True}
