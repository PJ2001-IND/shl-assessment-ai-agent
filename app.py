"""
app.py — FastAPI application entry point for the SHL Assessment Agent

Endpoints:
  GET  /health  — Readiness check (returns {"status": "ok"})
  POST /chat    — Stateless conversational recommendation endpoint
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from catalog import catalog_index
from engine import process_chat
from models import ChatRequest, ChatResponse

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Application Lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load catalog and build FAISS index.
    This runs once when the service starts (cold start).
    On Render free tier, the first /health call allows up to 2 minutes.
    """
    logger.info("Starting SHL Assessment Agent...")
    start = time.time()

    try:
        catalog_index.load()
        elapsed = time.time() - start
        logger.info(f"Catalog loaded in {elapsed:.1f}s. {len(catalog_index.records)} assessments indexed.")
    except Exception as e:
        logger.error(f"Failed to load catalog: {e}")
        raise

    yield

    # Shutdown
    logger.info("SHL Assessment Agent shutting down.")


# ── FastAPI App ────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    description=(
        "Conversational agent that recommends SHL assessments through multi-turn dialogue. "
        "Built for the SHL AI Intern Assignment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (needed for evaluator)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request Timing Middleware ──────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    return response


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """
    Readiness check endpoint.
    Returns {"status": "ok"} when the service is ready.
    Allows up to 2 minutes for cold-start wake-up (Render free tier).
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless conversational endpoint.

    Every call must include the full conversation history in `messages`.
    Returns the next agent reply, optional recommendations (1-10 items),
    and an end_of_conversation flag.

    Schema is non-negotiable — deviating breaks the automated evaluator.
    """
    try:
        response = await process_chat(request)
        return response
    except ValueError as e:
        # Pydantic validation error from engine
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /chat: {e}", exc_info=True)
        # Return a valid schema response rather than a 500
        return ChatResponse(
            reply="I encountered an unexpected error. Please try again.",
            recommendations=None,
            end_of_conversation=False,
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ── Dev Server Entry Point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT

    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
    )
