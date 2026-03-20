"""
PII Anonymizer — dual-layer bias removal.
Layer 1: Regex-based PII masking.
Layer 2: LLM-powered bias-field redaction.
"""

from __future__ import annotations

import re
from typing import Any

from config import settings

_PII_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("email",   "[EMAIL]",   re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}")),
    ("phone",   "[PHONE]",   re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")),
    ("url",     "[URL]",     re.compile(r"https?://[^\s,;\"']+")),
    ("address", "[ADDRESS]", re.compile(
        r"\d{1,5}\s[\w\s]{2,30}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)"
        r"[\w\s,]*\d{5,6}", re.IGNORECASE,
    )),
]

_BIAS_FIELDS = {
    "name", "full_name", "first_name", "last_name",
    "gender", "sex", "age", "date_of_birth", "dob", "birth_date",
    "ethnicity", "race", "nationality", "religion",
    "photo", "photo_url", "image", "avatar",
    "marital_status", "disability", "veteran_status",
    "address", "street", "city", "state", "zip", "zipcode", "country",
    "university",
}

_BIAS_PLACEHOLDERS = {
    "name": "[CANDIDATE]", "full_name": "[CANDIDATE]",
    "first_name": "[CANDIDATE_FIRST]", "last_name": "[CANDIDATE_LAST]",
    "gender": "[REDACTED]", "sex": "[REDACTED]",
    "age": "[REDACTED]", "date_of_birth": "[REDACTED]",
    "dob": "[REDACTED]", "birth_date": "[REDACTED]",
    "ethnicity": "[REDACTED]", "race": "[REDACTED]",
    "nationality": "[REDACTED]", "religion": "[REDACTED]",
    "photo": "[REDACTED]", "photo_url": "[REDACTED]",
    "image": "[REDACTED]", "avatar": "[REDACTED]",
    "marital_status": "[REDACTED]", "disability": "[REDACTED]",
    "veteran_status": "[REDACTED]",
    "address": "[ADDRESS]", "street": "[ADDRESS]",
    "city": "[LOCATION]", "state": "[LOCATION]",
    "zip": "[LOCATION]", "zipcode": "[LOCATION]",
    "country": "[LOCATION]",
    "university": "[UNIVERSITY]",
}


def _mask_pii_in_string(text: str) -> str:
    for _, placeholder, pattern in _PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def _mask_pii_recursive(obj: Any) -> Any:
    if isinstance(obj, str):
        return _mask_pii_in_string(obj)
    if isinstance(obj, dict):
        return {k: _mask_pii_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_pii_recursive(item) for item in obj]
    return obj


def mask_pii(candidate: dict) -> dict:
    result = {}
    for key, value in candidate.items():
        key_lower = key.lower().replace(" ", "_")
        if key_lower in _BIAS_FIELDS:
            result[key] = _BIAS_PLACEHOLDERS.get(key_lower, "[REDACTED]")
        else:
            result[key] = _mask_pii_recursive(value)
    return result


async def redact_bias_with_llm(candidate: dict) -> dict:
    """LLM layer — redacts company names, university names, and remaining bias signals."""
    if not settings.GEMINI_API_KEY:
        # No API key: regex layer is sufficient, skip LLM redaction
        return candidate
    
    print("   [API Call] 🔒 Anonymizer LLM running...")

    import json
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a bias-removal specialist. Review this candidate profile JSON
and redact information that could introduce bias in a hiring decision.

Rules:
1. Replace all university/college names with "[UNIVERSITY]"
2. Replace all company/employer names with "[COMPANY]"
3. Remove any remaining non-job-relevant personal details
4. Keep ALL technical skills, years of experience, projects, and achievements intact
5. Return ONLY valid JSON — no markdown, no explanation."""),
        ("human", "Candidate profile:\n{candidate_json}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.0,
    )
    chain = prompt | llm | JsonOutputParser()

    try:
        redacted = await chain.ainvoke({
            "candidate_json": json.dumps(candidate, indent=2)
        })
        return redacted if isinstance(redacted, dict) else candidate
    except Exception:
        # If LLM redaction fails, fall back to regex-only result
        return candidate


async def anonymize(candidate: dict) -> dict:
    """Full anonymization: regex PII masking + LLM bias removal."""
    masked = mask_pii(candidate)
    redacted = await redact_bias_with_llm(masked)
    return redacted