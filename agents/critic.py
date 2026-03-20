"""
Critic Agent — argues AGAINST hiring the candidate.
"""

from __future__ import annotations
import json
from config import settings

MOCK_CRITIC_RESPONSE = {
    "position": "no_hire",
    "confidence": 0.60,
    "key_arguments": [
        "Candidate is missing Kubernetes, Terraform, GraphQL, and Kafka — all listed as requirements",
        "Weak similarity scores for terraform (0.48), graphql (0.44), and kubernetes (0.32)",
        "REST API Design match at 0.43 is concerningly low for a backend role",
    ],
    "risks_identified": [
        "Kubernetes gap is critical — the JD requires 2+ years production k8s experience",
        "No Terraform means candidate cannot contribute to IaC from day one",
        "GraphQL and Kafka gaps may impact team velocity significantly",
    ],
    "evidence_cited": [
        "5 of 12 JD skills scored below the 0.7 threshold",
        "kubernetes ↔ aws similarity: 0.32 — not a valid match",
    ],
    "rebuttal_to_advocate": None,
}


async def run_critic(evidence_bundle: dict, debate_context: str = "") -> dict:
    if settings.USE_MOCK_LLM or not settings.GEMINI_API_KEY:
        result = dict(MOCK_CRITIC_RESPONSE)
        missing = evidence_bundle.get("missing_skills", [])
        if missing:
            result["key_arguments"][0] = f"Candidate is missing {len(missing)} required skills: {', '.join(missing[:5])}"
        return result

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the CRITIC agent in a structured hiring debate.
Present the STRONGEST case AGAINST hiring this candidate based ONLY on the evidence.

Rules:
1. Base ALL arguments on the evidence — do not fabricate.
2. Highlight missing skills and experience shortfalls.
3. Question weak matches (similarity < 0.7).
4. Note concerning patterns.
5. If responding to advocate, challenge with data.

Return ONLY valid JSON:
{{
    "position": "no_hire",
    "confidence": <float 0-1>,
    "key_arguments": ["argument1", ...],
    "risks_identified": ["risk1", ...],
    "evidence_cited": ["evidence1", ...],
    "rebuttal_to_advocate": "response or null"
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
