"""
models.py

Pydantic schema for the resume evaluator's output. This is the single
source of truth for what a "valid" result looks like: it is passed to
Gemini as the structured-output schema, and it is used again in Python
to independently validate whatever the model returns.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ResumeEvaluation(BaseModel):
    """Validated, guaranteed-shape result of a resume / job-description match."""

    match_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall match score between 0 and 100.",
    )
    top_strengths: list[str] = Field(
        ...,
        min_length=1,
        description="Concrete strengths that are directly supported by the resume text.",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Skills required by the job description that the resume does not evidence.",
    )
    summary: list[str] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Exactly two summary lines describing overall fit.",
    )

    @field_validator("top_strengths", "missing_skills")
    @classmethod
    def _strip_blank_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]

    @field_validator("summary")
    @classmethod
    def _summary_must_have_two_lines(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if len(cleaned) != 2:
            raise ValueError("summary must contain exactly 2 non-empty lines")
        return cleaned
