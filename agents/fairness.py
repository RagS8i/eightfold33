"""
Fairness Agent — fact-checks both sides and detects bias.
"""

from __future__ import annotations
import json
from config import settings

MOCK_FAIRNESS_RESPONSE = {
    "verified_claims": [
        "Advocate's skill match percentage is consistent with the FAISS similarity scores",
        "Critic's identification of missing Kubernetes, Terraform, GraphQL, Kafka is accurate",
        "Both agents are reasoning from the evidence bundle without fabrication",
    ],
    "disputed_claims": [],
    "bias_flags": [],
    "balance_assessment": "Both agents present evidence-backed arguments. The debate is balanced. Critic has stronger data support for specific gaps.",
    "recommendations": [
        "Weight Kubernetes and Terraform gaps heavily — both are explicitly required",
        "Consider that 6 years experience may offset some skill gaps via learning velocity",
    ],
    "fairness_score": 0.87,
}


async def run_fairness(evidence_bundle: dict, advocate_args: dict, critic_args: dict) -> dict:
    if settings.USE_MOCK_LLM or not settings.GEMINI_API_KEY:
        return dict(MOCK_FAIRNESS_RESPONSE)

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the FAIRNESS agent. Fact-check both sides and ensure debate integrity.

Rules:
1. Verify every claim against the evidence bundle.
2. Flag any argument not supported by data.
3. Detect subtle bias in reasoning.
4. Remain strictly neutral.

Return ONLY valid JSON:
{{
    "verified_claims": ["claim...", ...],
    "disputed_claims": ["claim + explanation...", ...],
    "bias_flags": ["bias description...", ...],
    "balance_assessment": "which agent has stronger evidence",
    "recommendations": ["suggestion...", ...],
    "fairness_score": <float 0-1>
}}"""),
        ("human", "EVIDENCE:\n{evidence}\n\nADVOCATE:\n{advocate}\n\nCRITIC:\n{critic}\n\nReturn JSON only."),
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2,
    )
    chain = prompt | llm | JsonOutputParser()
    return await chain.ainvoke({
        "evidence": json.dumps(evidence_bundle, indent=2, default=str),
        "advocate": json.dumps(advocate_args, indent=2, default=str),
        "critic": json.dumps(critic_args, indent=2, default=str),
    })
