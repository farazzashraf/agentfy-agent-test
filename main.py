"""
main.py — Restaurant FAQ Agent (Production)

What changed from the dev version:
  - Gemini initialized ONCE at startup via lifespan, not per-request
  - Structured logging replaces print()
  - slowapi rate limiting on all AI endpoints
  - Full input validation (lengths, history schema, stripped strings)
  - Async timeout wrapping every Gemini call
  - Error handling that never leaks internal details to clients
  - Request ID middleware + access log
  - CORS from env variable (not hardcoded *)
  - agentfy.yaml cached at startup (lru_cache)
  - Explicit handling of Gemini content-filter blocks
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import google.generativeai as genai

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("faq_agent")


# ── Constants ──────────────────────────────────────────────────────────────────
MAX_USER_MESSAGE_LEN = 1_000    # characters — sane cap, stops prompt injection attempts
MAX_EMBED_TEXT_LEN   = 5_000    # characters
MAX_HISTORY_MESSAGES = 10       # last N messages kept in context window
AI_TIMEOUT_SECONDS   = 30       # hard wall on every Gemini call
MODEL_NAME           = "gemini-2.5-flash"


# ── Config (loaded once, cached forever) ───────────────────────────────────────
@lru_cache(maxsize=1)
def load_agent_config() -> dict[str, Any]:
    try:
        with open("agentfy.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("agentfy.yaml loaded OK")
        return cfg
    except FileNotFoundError:
        logger.warning("agentfy.yaml not found — using empty defaults")
        return {}
    except yaml.YAMLError as exc:
        logger.error("agentfy.yaml parse error: %s", exc)
        return {}


# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Startup / shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Gemini exactly once. Fail fast if the key is missing."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var is not set — cannot start.")

    init_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("GEMINI_BASE_URL")
    if base_url:
        from google.api_core import client_options as co  # optional dependency
        init_kwargs["transport"] = "rest"
        init_kwargs["client_options"] = co.ClientOptions(api_endpoint=base_url)
        logger.info("Gemini proxied via: %s", base_url)

    genai.configure(**init_kwargs)
    load_agent_config()   # warm the LRU cache now, not on first request
    logger.info("Startup complete — model: %s", MODEL_NAME)

    yield  # app is live here

    logger.info("Shutdown complete.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Restaurant FAQ Agent", lifespan=lifespan)

# Rate limiter state + 429 handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — default to localhost only in prod; override via env
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Request-ID + access log middleware ────────────────────────────────────────
@app.middleware("http")
async def request_context_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    t0 = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - t0) * 1_000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s → %d  (%.1f ms)  req_id=%s",
        request.method, request.url.path,
        response.status_code, duration_ms, request_id,
    )
    return response


# ── Pydantic models ────────────────────────────────────────────────────────────
class HistoryEntry(BaseModel):
    """Single turn of chat history. role must be 'user' or 'model'."""
    role:  str        = Field(..., pattern="^(user|model)$")
    parts: list[str]  = Field(..., min_length=1, max_length=10)


class ChatRequest(BaseModel):
    message:          str               = Field(..., min_length=1, max_length=MAX_USER_MESSAGE_LEN)
    history:          list[HistoryEntry] = Field(default_factory=list)
    injected_context: dict               = Field(default_factory=dict)

    @field_validator("message", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_EMBED_TEXT_LEN)


# ── Helpers ────────────────────────────────────────────────────────────────────
def build_system_prompt(config: dict[str, Any]) -> str:
    name  = config.get("restaurant_name", "our restaurant")
    phone = config.get("phone_number",    "our number")
    return f"""
You are a friendly FAQ assistant for {name}.
Answer customer questions accurately and warmly.
If you don't know something, say "I'm not sure — please call us at {phone}."

=== RESTAURANT DETAILS ===
Name:          {name}
Cuisine:       {config.get("cuisine_type",  "N/A")}
Address:       {config.get("address",       "N/A")}
Phone:         {phone}
Price Range:   {config.get("price_range",   "N/A")}

Opening Hours:
{config.get("opening_hours",    "N/A")}

Menu Highlights:
{config.get("menu_highlights",  "N/A")}

Dietary Options:     {config.get("dietary_options",   "Please call us to ask.")}
Parking:             {config.get("parking_info",      "Please contact us for parking info.")}
Reservation Policy:  {config.get("reservation_policy","Please call us to book.")}

=== RULES ===
1. Be warm and concise. No walls of text.
2. For prices, give the range and suggest calling for exact pricing.
3. For bookings, explain the policy and provide the phone number.
4. If asked something unrelated, redirect politely: "I can only help with questions about {name}."
5. Never claim to be human. You are the restaurant's AI assistant.
""".strip()


def to_gemini_history(entries: list[HistoryEntry]) -> list[dict]:
    """Validated, truncated, Gemini-compatible history list."""
    return [
        {"role": e.role, "parts": e.parts}
        for e in entries[-MAX_HISTORY_MESSAGES:]
    ]


async def _call_gemini_chat(model, message: str, history: list[dict]) -> str:
    """
    Run the blocking Gemini SDK call in a thread pool and enforce a hard timeout.
    Returns the response text. Raises HTTPException on timeout or API error.
    """
    try:
        chat_session = model.start_chat(history=history)
        response = await asyncio.wait_for(
            asyncio.to_thread(chat_session.send_message, message),
            timeout=AI_TIMEOUT_SECONDS,
        )
        return response.text

    except asyncio.TimeoutError:
        logger.warning("Gemini chat timed out after %ds", AI_TIMEOUT_SECONDS)
        raise HTTPException(
            status_code=504,
            detail="The AI took too long to respond. Please try again.",
        )
    except genai.types.BlockedPromptException:
        raise HTTPException(
            status_code=422,
            detail="Your message was flagged by content filters and could not be processed.",
        )
    except Exception as exc:
        logger.exception("Gemini chat error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="AI service is temporarily unavailable. Please try again shortly.",
        )


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root_get():
    return FileResponse("static/index.html")


@app.post("/")
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest):
    """
    Primary chat endpoint. Rate-limited to 30 req/min per IP.
    injected_context takes precedence over the YAML config so the
    Agent-fy platform can override fields per deployment.
    """
    config  = body.injected_context if body.injected_context else load_agent_config()
    history = to_gemini_history(body.history)
    model   = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=build_system_prompt(config),
    )

    reply = await _call_gemini_chat(model, body.message, history)

    return {
        "status":  "success",
        "response": reply,
        "history": history + [
            {"role": "user",  "parts": [body.message]},
            {"role": "model", "parts": [reply]},
        ],
    }


@app.post("/embed")
@limiter.limit("20/minute")
async def embed(request: Request, body: EmbedRequest):
    """Embedding endpoint. Rate-limited to 20 req/min per IP."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                genai.embed_content,
                model="models/gemini-embedding-2",
                content=body.text,
                task_type="retrieval_document",
            ),
            timeout=AI_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Gemini embed timed out after %ds", AI_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="Embedding request timed out.")
    except Exception as exc:
        logger.exception("Gemini embed error: %s", exc)
        raise HTTPException(status_code=502, detail="Embedding service unavailable.")

    return {"status": "success", "embedding": result["embedding"]}


@app.post("/sync")
async def sync(payload: dict):
    """
    Gateway sync hook. This agent is stateless so we just echo back
    whatever state was sent.
    """
    return {"status": "success", "state": payload.get("state", {})}


@app.get("/config")
async def get_config():
    """Returns the cached agent config. Reads from disk only once per process."""
    return load_agent_config()


@app.get("/status")
async def status():
    return {"agent": "Restaurant FAQ Agent", "status": "live", "model": MODEL_NAME}


@app.get("/health")
async def health():
    """
    Liveness probe for Cloud Run / Kubernetes.
    Intentionally does NOT call Gemini — just confirms the process is alive.
    """
    return {"status": "healthy"}


# ── Static files (must be mounted last) ───────────────────────────────────────
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")