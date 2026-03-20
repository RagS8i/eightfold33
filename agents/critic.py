"""
Critic Agent — argues AGAINST hiring the candidate.
Always uses live Gemini LLM reasoning.
"""
from __future__ import annotations
import json
from config import settings


async def run_critic(evidence_bundle: dict, debate_context: str = "") -> dict:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("No GEMINI_API_KEY set. Please add your API key in the sidebar.")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the CRITIC agent in a structured hiring debate.
Your job is to present the STRONGEST honest case AGAINST hiring this candidate,
based ONLY on the evidence bundle provided.

Rules:
1. Base ALL arguments strictly on the evidence — never fabricate missing skills.
2. Challenge any skill match with similarity < 0.7 as unreliable.
3. Highlight missing required skills and experience shortfalls.
4. Question whether weak vector matches represent real practical skill.
5. Cite specific similarity scores, missing skills, and experience gaps.
6. If responding to the advocate, directly rebut their specific claims with data.
7. Be rigorous but fair — do not dismiss genuinely strong matches.

Return ONLY valid JSON, no markdown, no explanation:
{{
    "position": "no_hire",
    "confidence": <float 0.0-1.0>,
    "key_arguments": ["detailed argument 1", "detailed argument 2", ...],
    "risks_identified": ["specific risk 1", "specific risk 2", ...],
    "evidence_cited": ["specific data point 1", "specific data point 2", ...],
    "rebuttal_to_advocate": "direct response to advocate's arguments, or null if opening round"
}}"""),
        ("human", "EVIDENCE BUNDLE:\n{evidence}\n\nDEBATE CONTEXT:\n{context}\n\nReturn JSON only."),
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.7,
    )
    chain = prompt | llm | JsonOutputParser()
    return await chain.ainvoke({
        "evidence": json.dumps(evidence_bundle, indent=2, default=str),
        "context": debate_context or "Opening round — make your strongest case against hiring.",
    })