"""
engine.py — Conversation engine (core logic)

Responsibilities:
  1. Parse conversation history to extract context & intent
  2. Build search query and retrieve relevant assessments from catalog
  3. Call Gemini LLM with injected catalog data
  4. Parse + validate LLM response against catalog
  5. Return a schema-compliant ChatResponse
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import google.generativeai as genai

from catalog import catalog_index
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_TIMEOUT_SECONDS,
    MAX_RECOMMENDATIONS,
    MAX_TURNS,
    RECOMMEND_BY_TURN,
    SIMILARITY_TOP_K,
)
from models import ChatRequest, ChatResponse, Recommendation
from prompts import (
    build_system_prompt,
    format_previous_recommendations,
    format_retrieved_assessments,
)

logger = logging.getLogger(__name__)


# ── Previous Recommendations Parser ───────────────────────────────────────

def _extract_previous_recommendations(request: ChatRequest) -> list[dict]:
    """
    Scan ALL assistant messages in reverse to find the most recent
    recommendations array. This is resilient to the harness storing
    either the full JSON blob or just the reply text as assistant content.
    """
    for msg in reversed(request.messages):
        if msg.role != "assistant":
            continue

        content = msg.content
        if not content:
            continue

        # Try direct JSON parse (harness / test suite stores full JSON blob)
        try:
            data = json.loads(content)
            recs = data.get("recommendations")
            if isinstance(recs, list) and recs:
                return recs
        except (json.JSONDecodeError, AttributeError, ValueError):
            pass

        # Fallback: regex scan for embedded recommendations array
        json_match = re.search(r'"recommendations"\s*:\s*(\[.*?\])', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, list) and parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

    return []


# ── Search Query Builder ───────────────────────────────────────────────────

def _build_search_query(messages: list, last_user_msg: str) -> str:
    """
    Build a rich search query by combining the full conversation context
    with the latest user message. The latest user message is repeated 3×
    so that add/drop refinement signals dominate the FAISS retrieval
    (critical for Recall@10 on multi-turn traces).
    """
    context_parts = []

    # Extract all user messages for full context (excluding the latest)
    for msg in messages[:-1]:
        if msg.role == "user":
            context_parts.append(msg.content)

    # Triple-weight the latest user message for refinement precision
    prior_context = " | ".join(context_parts)
    if prior_context:
        full_context = f"{prior_context} | {last_user_msg} | {last_user_msg} | {last_user_msg}"
    else:
        full_context = f"{last_user_msg} | {last_user_msg} | {last_user_msg}"

    return full_context[:2000]  # Cap length for embedding


# ── LLM Call ──────────────────────────────────────────────────────────────

def _configure_gemini() -> genai.GenerativeModel:
    """Configure and return a Gemini generative model."""
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=genai.GenerationConfig(
            temperature=0.2,        # Low temperature for factual, grounded responses
            top_p=0.9,
            max_output_tokens=4096,
            response_mime_type="application/json",  # Force JSON output
        ),
    )


async def _call_llm(
    model: genai.GenerativeModel,
    system_prompt: str,
    conversation_messages: list,
    last_user_message: str,
) -> str:
    """
    Call Gemini with the full conversation history.
    Returns the raw text response.
    """
    # Build Gemini chat history from prior messages (excluding the last user turn)
    history = []
    messages_for_history = conversation_messages[:-1]  # exclude last user msg

    for msg in messages_for_history:
        role = "user" if msg.role == "user" else "model"
        history.append({"role": role, "parts": [msg.content]})

    # Send system prompt + last user message together
    combined_message = f"{system_prompt}\n\n---\nUser's current message: {last_user_message}"

    logger.info(f"DEBUG: Using API Key starting with: '{GEMINI_API_KEY[:5]}'...")

    loop = asyncio.get_event_loop()
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Create chat session with history anew on each attempt
            chat = model.start_chat(history=history)
            response = await loop.run_in_executor(
                None,
                lambda: chat.send_message(combined_message),
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt == max_retries - 1:
                    logger.error(f"LLM rate limit max retries reached: {e}")
                    raise
                # Parse retry_delay from error message if available
                delay_match = re.search(r"retry in (\d+\.?\d*)", error_str, re.IGNORECASE)
                wait = float(delay_match.group(1)) + 2 if delay_match else (10.0 * (2 ** attempt))
                logger.warning(f"Rate limited in chat. Waiting {wait:.1f}s...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"Non-retriable LLM error: {e}")
                raise


# ── Response Parser ────────────────────────────────────────────────────────

def _parse_llm_response(raw_text: str) -> dict[str, Any]:
    """
    Parse the LLM's JSON response. Handles edge cases where the model
    wraps JSON in markdown code blocks despite being in JSON mode.
    """
    text = raw_text.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}\nRaw: {text[:500]}")
        # Return a safe fallback
        return {
            "reply": "I encountered an issue processing your request. Could you please rephrase?",
            "recommendations": None,
            "end_of_conversation": False,
        }


# ── Recommendation Validator ───────────────────────────────────────────────

def _validate_and_ground_recommendations(
    raw_recs: list[dict] | None,
) -> list[Recommendation] | None:
    """
    Validate every recommendation against the catalog.
    - Names are fuzzy-matched to prevent hallucination
    - URLs are corrected to exact catalog URLs
    - test_type codes are derived from catalog keys (not trusted from LLM)
    - Invalid recommendations are silently dropped
    """
    if raw_recs is None:
        return None

    if not isinstance(raw_recs, list) or len(raw_recs) == 0:
        return None

    validated = []
    seen_names = set()

    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue

        name = rec.get("name", "").strip()
        if not name:
            continue

        # Deduplicate
        if name.lower() in seen_names:
            continue

        # Find in catalog
        catalog_record = catalog_index.validate_recommendation_name(name)
        if catalog_record is None:
            logger.warning(f"Recommendation '{name}' not found in catalog — dropping.")
            continue

        seen_names.add(catalog_record.name.lower())

        validated.append(Recommendation(
            name=catalog_record.name,           # Use exact catalog name
            url=catalog_record.url,             # Use exact catalog URL
            test_type=catalog_record.test_type_codes,  # Derived from catalog
        ))

        if len(validated) >= MAX_RECOMMENDATIONS:
            break

    return validated if validated else None


# ── Turn Count Guard ───────────────────────────────────────────────────────

def _is_force_recommend_turn(request: ChatRequest) -> bool:
    """Returns True if we must force a recommendation this turn."""
    return request.turn_count >= RECOMMEND_BY_TURN


def _is_over_turn_limit(request: ChatRequest) -> bool:
    """Returns True if the conversation has exceeded the max turns."""
    # The new user message hasn't been added to history yet, so we check
    # if adding it would reach MAX_TURNS
    return request.turn_count >= MAX_TURNS


# ── Main Engine Function ───────────────────────────────────────────────────

async def process_chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversation processing pipeline:
    1. Guard: turn limit check
    2. Extract previous recommendations from history
    3. Build search query & retrieve relevant assessments
    4. Build system prompt with retrieved catalog data
    5. Call LLM
    6. Parse & validate response
    7. Return schema-compliant ChatResponse
    """

    last_user_msg = request.last_user_message
    turn_count = request.turn_count

    logger.info(f"Processing turn {turn_count}: '{last_user_msg[:100]}'")

    # ── Guard: over turn limit ─────────────────────────────────────────────
    if _is_over_turn_limit(request):
        logger.warning(f"Turn limit reached ({turn_count} turns). Forcing end.")
        prev_recs = _extract_previous_recommendations(request)
        validated = _validate_and_ground_recommendations(prev_recs) if prev_recs else None

        if not validated:
            # Emergency: search for something relevant
            search_query = _build_search_query(request.messages, last_user_msg)
            records = catalog_index.search(search_query, top_k=5)
            validated = [r.to_recommendation() for r in records[:5]] or None

        return ChatResponse(
            reply="I've provided my best recommendation based on our conversation. Happy to continue in a new session if you'd like to refine further.",
            recommendations=validated,
            end_of_conversation=True,
        )

    # ── Extract previous recommendations from history ───────────────────────
    prev_recs = _extract_previous_recommendations(request)
    force_recommend = _is_force_recommend_turn(request)

    # ── Build search query ──────────────────────────────────────────────────
    search_query = _build_search_query(request.messages, last_user_msg)

    # ── Retrieve relevant assessments ───────────────────────────────────────
    records = catalog_index.search(
        query=search_query,
        top_k=SIMILARITY_TOP_K,
    )

    # ── Always-inject baseline assessments ──────────────────────────────────
    # OPQ32r and Verify G+ appear in nearly every conversation
    key_assessments = ["occupational personality questionnaire opq32r", "shl verify interactive g"]
    for key_name in key_assessments:
        found = catalog_index.find_by_name_fuzzy(key_name, threshold=0.5)
        if found and found not in records:
            records.append(found)

    # ── Domain keyword boosting ──────────────────────────────────────────────
    # For recognized domains, ensure domain-specific assessments are always
    # in the context (FAISS alone may miss niche tests in dense vectors).
    DOMAIN_BOOSTS: dict[tuple, list[str]] = {
        # trigger keywords → catalog names to boost
        ("safety", "chemical", "plant operator", "hazard", "industrial", "dsi", "reliability", "procedure"): [
            "dependability and safety instrument",
        ],
        ("sales", "reskill", "re-skill", "restructur", "talent audit", "seller", "rep"): [
            "opq mq sales report",
            "sales transformation",
            "global skills assessment",
        ],
        ("graduate", "final-year", "entry-level", "no work experience", "trainee", "intern"): [
            "graduate scenarios",
        ],
        ("coding", "live coding", "live interview", "pair programming"): [
            "smart interview live coding",
        ],
        ("contact centre", "call centre", "call center", "customer service", "inbound", "svar"): [
            "svar",
        ],
        ("leadership", "executive", "cxo", "director", "c-suite", "ceo", "vp ", " vp"): [
            "opq leadership report",
            "opq universal competency report",
        ],
        ("healthcare", "medical", "nurse", "patient", "hipaa", "clinical", "hospital"): [
            "occupational personality questionnaire opq32r",
        ],
        ("numerical", "financial analyst", "finance", "analyst", "accountant"): [
            "shl verify interactive - numerical reasoning",
            "financial accounting",
        ],
    }

    query_lower = search_query.lower()
    for trigger_keywords, boost_names in DOMAIN_BOOSTS.items():
        if any(kw in query_lower for kw in trigger_keywords):
            for boost_name in boost_names:
                found = catalog_index.find_by_name_fuzzy(boost_name, threshold=0.4)
                if found and found not in records:
                    records.append(found)
                    logger.debug(f"Domain-boosted: {found.name}")

    # ── Format catalog context ──────────────────────────────────────────────
    retrieved_str = format_retrieved_assessments(records[:20])  # Cap at 20 in prompt
    prev_recs_str = format_previous_recommendations(prev_recs)

    # ── Build system prompt ─────────────────────────────────────────────────
    system_prompt = build_system_prompt(
        retrieved_assessments=retrieved_str,
        previous_recommendations=prev_recs_str,
        force_recommend=force_recommend,
    )

    # ── Call LLM with timeout ───────────────────────────────────────────────
    model = _configure_gemini()

    try:
        raw_response = await asyncio.wait_for(
            _call_llm(
                model=model,
                system_prompt=system_prompt,
                conversation_messages=request.messages,
                last_user_message=last_user_msg,
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("LLM call timed out.")
        return ChatResponse(
            reply="I'm taking too long to respond. Please try again.",
            recommendations=None,
            end_of_conversation=False,
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ChatResponse(
            reply="I encountered an error. Please try again.",
            recommendations=None,
            end_of_conversation=False,
        )

    # ── Parse LLM response ──────────────────────────────────────────────────
    parsed = _parse_llm_response(raw_response)

    reply = parsed.get("reply", "Could you please rephrase your question?")
    raw_recs = parsed.get("recommendations")
    end_flag = bool(parsed.get("end_of_conversation", False))

    # ── Validate & ground recommendations ──────────────────────────────────
    validated_recs = _validate_and_ground_recommendations(raw_recs)

    # ── Override end_of_conversation if force_recommend ────────────────────
    if force_recommend and validated_recs and not end_flag:
        # Don't force-end unless user confirmed, but do force a recommendation
        pass

    logger.info(
        f"Turn {turn_count} response: "
        f"{len(validated_recs) if validated_recs else 0} recommendations, "
        f"end_of_conversation={end_flag}"
    )

    return ChatResponse(
        reply=reply,
        recommendations=validated_recs,
        end_of_conversation=end_flag,
    )
