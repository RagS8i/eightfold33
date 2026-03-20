"""
Fairness Agent — fact-checks both sides and detects bias.
Always uses live Gemini LLM reasoning.
"""
from __future__ import annotations
import json
from config import settings


async def run_fairness(evidence_bundle: dict, advocate_args: dict, critic_args: dict) -> dict:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("No GEMINI_API_KEY set. Please add your API key in the sidebar.")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the FAIRNESS agent in a structured hiring debate.
Your job is to fact-check both the Advocate and Critic against the evidence bundle,
detect any bias or fabrication, and provide a balanced assessment.

Rules:
1. Verify every specific claim made by both agents against the evidence bundle.
2. Flag any claim that is not directly supported by the data.
3. Detect subtle bias — e.g. overstating weak matches, dismissing valid experience.
4. Check that similarity scores cited are accurate.
5. Check that missing skills are genuinely absent, not just weakly matched.
6. Remain strictly neutral — do not favor either side.
7. Score fairness based on how evidence-grounded both agents are.

Return ONLY valid JSON, no markdown, no explanation:
{{
    "verified_claims": ["verified claim from either agent", ...],
    "disputed_claims": ["disputed claim + why it's unsupported", ...],
    "bias_flags": ["description of any detected bias", ...],
    "balance_assessment": "1-2 sentence assessment of which agent has stronger evidence support",
    "recommendations": ["specific recommendation for the judge", ...],
    "fairness_score": <float 0.0-1.0, where 1.0 means both agents fully grounded in evidence>
}}"""),
        ("human", "EVIDENCE BUNDLE:\n{evidence}\n\nADVOCATE ARGUMENTS:\n{advocate}\n\nCRITIC ARGUMENTS:\n{critic}\n\nReturn JSON only."),
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