"""
models.py — Pydantic request/response schemas for the SHL Assessment Agent

The response schema is NON-NEGOTIABLE per the assignment spec.
Deviating breaks the automated evaluator.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


# ── Request Models ─────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="Full conversation history. Every call must include all prior turns.",
    )

    @model_validator(mode="after")
    def validate_message_order(self) -> "ChatRequest":
        """First message must always be from user."""
        if self.messages and self.messages[0].role != "user":
            raise ValueError("First message must be from 'user'.")
        return self

    @property
    def turn_count(self) -> int:
        """Total number of turns (user + assistant combined)."""
        return len(self.messages)

    @property
    def last_user_message(self) -> str:
        """Content of the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""

    @property
    def last_assistant_message(self) -> str | None:
        """Content of the most recent assistant message, if any."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def to_prompt_history(self) -> list[dict[str, str]]:
        """Convert to Gemini-compatible message history format."""
        return [
            {"role": "user" if m.role == "user" else "model", "parts": [m.content]}
            for m in self.messages[:-1]  # Exclude last user message (sent separately)
        ]


# ── Response Models ────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    name: str = Field(..., description="Exact assessment name from the SHL catalog.")
    url: str = Field(..., description="Exact catalog URL from the SHL catalog.")
    test_type: str = Field(
        ...,
        description="Comma-separated test type codes (K, P, A, B, S, C, D, E).",
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's natural language response.")
    recommendations: list[Recommendation] | None = Field(
        default=None,
        description=(
            "null when still clarifying or refusing. "
            "Array of 1-10 items when the agent commits to a shortlist."
        ),
    )
    end_of_conversation: bool = Field(
        default=False,
        description="true only when the agent considers the task complete.",
    )

    @model_validator(mode="after")
    def validate_recommendations(self) -> "ChatResponse":
        """Enforce: if recommendations is a list, it must have 1-10 items."""
        if self.recommendations is not None:
            if len(self.recommendations) < 1:
                raise ValueError("recommendations list must have at least 1 item when provided.")
            if len(self.recommendations) > 10:
                raise ValueError("recommendations list must have at most 10 items.")
        return self


# ── Internal Models (not exposed via API) ─────────────────────────────────

class AssessmentRecord(BaseModel):
    """Normalized catalog assessment record."""
    entity_id: str
    name: str
    url: str                         # = link field from JSON
    description: str
    job_levels: list[str]
    languages: list[str]
    duration: str                    # "" if unknown/untimed
    duration_minutes: int | None     # Parsed integer minutes, None if unknown
    remote: bool
    adaptive: bool
    keys: list[str]                  # e.g. ["Knowledge & Skills", "Simulations"]
    test_type_codes: str             # e.g. "K,S"
    searchable_text: str             # Combined text for embedding

    def to_recommendation(self) -> Recommendation:
        return Recommendation(
            name=self.name,
            url=self.url,
            test_type=self.test_type_codes,
        )

    def to_context_str(self) -> str:
        """Compact string representation for LLM context injection."""
        langs = ", ".join(self.languages[:5])
        if len(self.languages) > 5:
            langs += f" (+{len(self.languages) - 5} more)"
        levels = ", ".join(self.job_levels)
        duration_str = self.duration if self.duration else "—"
        return (
            f"Name: {self.name}\n"
            f"URL: {self.url}\n"
            f"Type: {self.test_type_codes} ({', '.join(self.keys)})\n"
            f"Duration: {duration_str}\n"
            f"Job Levels: {levels}\n"
            f"Languages: {langs}\n"
            f"Adaptive: {'Yes' if self.adaptive else 'No'}\n"
            f"Description: {self.description}\n"
        )


class ConversationContext(BaseModel):
    """Extracted slot values from conversation history."""
    role_description: str | None = None
    seniority: str | None = None
    domain: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    language_requirement: str | None = None
    purpose: str | None = None        # "selection", "development", "re-skilling", "screening"
    constraints: list[str] = Field(default_factory=list)
    previous_recommendations: list[dict] = Field(default_factory=list)
    items_to_add: list[str] = Field(default_factory=list)
    items_to_remove: list[str] = Field(default_factory=list)
    is_refinement: bool = False
    is_comparison: bool = False
    is_confirmation: bool = False
    is_off_topic: bool = False
    has_enough_context: bool = False

    def to_search_query(self) -> str:
        """Build a natural language search query from extracted context."""
        parts = []
        if self.role_description:
            parts.append(self.role_description)
        if self.domain:
            parts.append(self.domain)
        if self.required_skills:
            parts.append(f"skills: {', '.join(self.required_skills)}")
        if self.seniority:
            parts.append(f"seniority: {self.seniority}")
        if self.purpose:
            parts.append(f"purpose: {self.purpose}")
        if self.constraints:
            parts.extend(self.constraints)
        return " ".join(parts) if parts else "assessment recommendation"
