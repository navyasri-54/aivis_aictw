"""
main.py

AIVI Intelligence - AI Engineering Challenge
Deliverable 3: Working Python AI Script

Evaluates a candidate resume against a job description using the Gemini
API, validates the model's structured output with Pydantic, and returns
a clean, guaranteed-shape JSON result.

Usage:
    python main.py --resume data/resume.txt --jd data/job_description.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from models import ResumeEvaluation
from prompts import SYSTEM_PROMPT, build_user_content

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "The 'google-genai' package is required.\n"
        "Install project dependencies with:\n"
        "    pip install -r requirements.txt"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-2.0-flash"
MAX_RETRIES = 5
BASE_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Transient-failure detection and retry / backoff
# ---------------------------------------------------------------------------

class TransientAPIError(Exception):
    """Raised when the Gemini API fails in a way that is worth retrying."""


def _is_transient(exc: Exception) -> bool:
    """Best-effort detection of retryable failures: 429, 503, and timeouts."""
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in (429, 503):
        return True

    if isinstance(exc, TimeoutError):
        return True

    message = str(exc).lower()
    transient_markers = (
        "429",
        "503",
        "rate limit",
        "resource_exhausted",
        "unavailable",
        "timeout",
        "timed out",
        "deadline exceeded",
    )
    return any(marker in message for marker in transient_markers)


def call_gemini_with_retry(
    client: "genai.Client",
    *,
    contents: str,
    config: "types.GenerateContentConfig",
    max_retries: int = MAX_RETRIES,
) -> Any:
    """Calls the Gemini API, retrying transient failures with exponential
    backoff and jitter. Non-transient errors are raised immediately."""

    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - inspected below
            last_exc = exc
            if not _is_transient(exc) or attempt == max_retries - 1:
                raise

            delay = min(MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2 ** attempt))
            delay += random.uniform(0, 0.5)  # jitter, avoids retry storms
            print(
                f"[retry] transient error on attempt {attempt + 1}/{max_retries} "
                f"({exc}); retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)

    # Defensive: loop above always returns or raises, but keep mypy/pylint calm.
    raise TransientAPIError(str(last_exc))


# ---------------------------------------------------------------------------
# JSON sanitization
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def sanitize_json(raw_text: str) -> dict[str, Any]:
    """Best-effort recovery of a JSON object from a raw model response.

    Handles the common failure modes of LLM "JSON" output:
      - wrapped in ```json ... ``` code fences
      - extra prose before/after the JSON object
      - trailing commas before a closing brace/bracket
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response from model.")

    text = raw_text.strip()
    text = _CODE_FENCE_RE.sub("", text).strip()

    # Fast path: the text is already valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the outermost {...} object and clean it up.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {raw_text!r}")

    candidate = text[start : end + 1]
    candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse sanitized JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# Core evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_resume(
    resume_text: str,
    job_description_text: str,
    api_key: str,
) -> ResumeEvaluation:
    """Runs the full pipeline: call Gemini, sanitize the output, validate it."""

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=ResumeEvaluation,
        temperature=0.2,
    )

    contents = build_user_content(resume_text, job_description_text)

    response = call_gemini_with_retry(client, contents=contents, config=config)

    raw_text = getattr(response, "text", None)
    parsed = sanitize_json(raw_text)

    try:
        return ResumeEvaluation.model_validate(parsed)
    except ValidationError:
        # Even with response_schema set, defensively re-validate: a model
        # can still return an almost-correct shape (e.g. extra prose that
        # slipped past sanitize_json, or a slightly malformed field).
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError as exc:
        raise SystemExit(f"File not found: {path}") from exc
    except OSError as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a resume against a job description using Gemini."
    )
    parser.add_argument(
        "--resume", required=True, help="Path to a plain-text resume file."
    )
    parser.add_argument(
        "--jd", required=True, help="Path to a plain-text job description file."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY is not set. Copy .env.example to .env and "
            "add your key, or export it in your shell.",
            file=sys.stderr,
        )
        return 1

    args = parse_args(argv)
    resume_text = _read_text_file(args.resume)
    jd_text = _read_text_file(args.jd)

    try:
        result = evaluate_resume(resume_text, jd_text, api_key)
    except TransientAPIError as exc:
        print(f"ERROR: Gemini API remained unavailable after retries: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: Could not parse model output as valid JSON: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"ERROR: Model output failed schema validation:\n{exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - final safety net for a CLI tool
        print(f"ERROR: Unexpected failure: {exc}", file=sys.stderr)
        return 1

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
