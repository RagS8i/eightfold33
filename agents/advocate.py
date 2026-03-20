"""
Advocate Agent — argues FOR hiring the candidate.
"""

from __future__ import annotations
import json
from config import settings

MOCK_ADVOCATE_RESPONSE = {
    "position": "hire",
    "confidence": 0.78,
    "key_arguments": [
        "Candidate matches the majority of required technical skills with strong proficiency ratings",
        "Demonstrated production experience with scalable systems handling high traffic",
        "Extra skills (MongoDB, Django, Linux) show versatility and adaptability",
        "AWS and PostgreSQL experience exceeds minimum requirements",
    ],
    "evidence_cited": [
        "7/12 JD skills directly matched at similarity ≥ 0.7",
        "6 years total experience exceeds the 5-year minimum",
        "AWS Certified and CKAD certifications validate cloud/k8s knowledge",
    ],
    "rebuttal_to_critic": None,
}


async def run_advocate(evidence_bundle: dict, debate_context: str = "") -> dict:
    if settings.USE_MOCK_LLM or not settings.GEMINI_API_KEY:
        result = dict(MOCK_ADVOCATE_RESPONSE)
        match_pct = evidence_bundle.get("match_percentage", 0)
        result["key_arguments"][0] = f"Candidate matches {match_pct:.0f}% of required skills"
        result["confidence"] = min(0.95, match_pct / 100 + 0.2)
        return result

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the ADVOCATE agent in a structured hiring debate.
Present the STRONGEST case FOR hiring this candidate based ONLY on the evidence bundle.

Rules:
1. Base ALL arguments on the evidence — do not fabricate.
2. Highlight matched skills, especially similarity ≥ 0.85.
3. Emphasize extra skills as value-add.
4. Frame gaps as learnable.
5. If responding to critic, address their concerns with evidence.

Return ONLY valid JSON:
{{
    "position": "hire",
    "confidence": <float 0-1>,
    "key_arguments": ["argument1", ...],
    "evidence_cited": ["evidence1", ...],
    "rebuttal_to_critic": "response or null"
}}"""),
        ("human", "EVIDENCE:\n{evidence}\n\nCONTEXT:\n{context}\n\nReturn JSON only."),
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.7,
    )
    chain = prompt | llm | JsonOutputParser()
    return await chain.ainvoke({
        "evidence": json.dumps(evidence_bundle, indent=2, default=str),
        "context": debate_context or "Opening round.",
    })
