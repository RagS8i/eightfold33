"""
Advocate Agent — argues FOR hiring the candidate.
Always uses live Gemini LLM reasoning.
"""
from __future__ import annotations
import json
from config import settings


async def run_advocate(evidence_bundle: dict, debate_context: str = "") -> dict:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("No GEMINI_API_KEY set. Please add your API key in the sidebar.")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the ADVOCATE agent in a structured hiring debate.
Your job is to present the STRONGEST honest case FOR hiring this candidate,
based ONLY on the evidence bundle provided.

Rules:
1. Base ALL arguments strictly on the evidence — never fabricate skills or experience.
2. Highlight skills with similarity ≥ 0.85 as strong matches.
3. Treat extra skills as genuine value-add signals.
4. Frame skill gaps as learnable given the candidate's experience level.
5. Cite specific similarity scores and experience years from the data.
6. If responding to the critic, directly address their specific claims with evidence.
7. Be persuasive but honest — do not overstate weak matches.

Return ONLY valid JSON, no markdown, no explanation:
{{
    "position": "hire",
    "confidence": <float 0.0-1.0>,
    "key_arguments": ["detailed argument 1", "detailed argument 2", ...],
    "evidence_cited": ["specific data point 1", "specific data point 2", ...],
    "rebuttal_to_critic": "direct response to critic's arguments, or null if opening round"
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
        "context": debate_context or "Opening round — make your strongest case for hiring.",
    })