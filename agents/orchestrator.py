"""
Orchestrator — LangGraph StateGraph running the multi-agent debate.
Flow: advocate → critic → fairness → (loop N rounds) → judge
"""

from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from config import settings
from agents.conversation_log import ConversationLog
from agents.advocate import run_advocate
from agents.critic import run_critic
from agents.fairness import run_fairness


def _merge_lists(left: list, right: list) -> list:
    return left + right


class DebateState(TypedDict):
    evidence: dict
    current_round: int
    max_rounds: int
    advocate_args: Annotated[list[dict], _merge_lists]
    critic_args: Annotated[list[dict], _merge_lists]
    fairness_reviews: Annotated[list[dict], _merge_lists]
    verdict: dict | None
    conversation_log: ConversationLog


async def advocate_node(state: DebateState) -> dict:
    log: ConversationLog = state["conversation_log"]
    round_num = state["current_round"]

    context = "Opening round."
    if state["critic_args"]:
        last_critic = state["critic_args"][-1]
        context = f"Round {round_num}. Critic argued:\n{json.dumps(last_critic, indent=2, default=str)}"
        if state["fairness_reviews"]:
            context += f"\n\nFairness:\n{json.dumps(state['fairness_reviews'][-1], indent=2, default=str)}"

    result = await run_advocate(state["evidence"], context)
    log.add_entry(
        agent_name="Advocate", role="advocate", round_number=round_num,
        content=json.dumps(result, indent=2, default=str),
        entry_type="argument" if round_num == 1 else "rebuttal",
    )
    return {"advocate_args": [result]}


async def critic_node(state: DebateState) -> dict:
    log: ConversationLog = state["conversation_log"]
    round_num = state["current_round"]

    last_advocate = state["advocate_args"][-1] if state["advocate_args"] else {}
    context = f"Round {round_num}. Advocate argued:\n{json.dumps(last_advocate, indent=2, default=str)}"
    if state["fairness_reviews"]:
        context += f"\n\nFairness:\n{json.dumps(state['fairness_reviews'][-1], indent=2, default=str)}"

    result = await run_critic(state["evidence"], context)
    log.add_entry(
        agent_name="Critic", role="critic", round_number=round_num,
        content=json.dumps(result, indent=2, default=str),
        entry_type="argument" if round_num == 1 else "rebuttal",
    )
    return {"critic_args": [result]}


async def fairness_node(state: DebateState) -> dict:
    log: ConversationLog = state["conversation_log"]
    round_num = state["current_round"]

    last_advocate = state["advocate_args"][-1] if state["advocate_args"] else {}
    last_critic = state["critic_args"][-1] if state["critic_args"] else {}

    result = await run_fairness(state["evidence"], last_advocate, last_critic)
    log.add_entry(
        agent_name="Fairness", role="fairness", round_number=round_num,
        content=json.dumps(result, indent=2, default=str),
        entry_type="fact_check",
    )
    return {"fairness_reviews": [result], "current_round": round_num + 1}


def should_continue(state: DebateState) -> str:
    if state["current_round"] > state["max_rounds"]:
        return "judge"
    return "continue"


MOCK_VERDICT = {
    "verdict": "LEAN_HIRE",
    "confidence": 0.72,
    "reasoning": (
        "The candidate demonstrates strong alignment with 7 of 12 required skills, "
        "particularly Python, Node.js, PostgreSQL, Docker, AWS, CI/CD, and Redis. "
        "However, notable gaps exist in Kubernetes, Terraform, GraphQL, and Kafka. "
        "The candidate's 6 years of experience and track record of delivering scalable "
        "systems slightly outweigh these gaps, suggesting they could ramp up quickly. "
        "A targeted technical interview on cloud-native tooling is recommended."
    ),
    "advocate_strength": 0.70,
    "critic_strength": 0.65,
    "key_factors": [
        "Strong core skill match (Python, AWS, PostgreSQL, Docker)",
        "Missing Kubernetes and Terraform are significant gaps",
        "Experience level and certifications partially compensate for skill gaps",
        "GraphQL and Kafka gaps are manageable with ramp-up time",
    ],
    "conditions": [
        "Verdict improves if candidate can demonstrate k8s fundamentals in interview",
        "Verdict worsens if Terraform IaC is day-one critical",
    ],
}


async def judge_node(state: DebateState) -> dict:
    log: ConversationLog = state["conversation_log"]

    if settings.USE_MOCK_LLM or not settings.GEMINI_API_KEY:
        match_pct = state["evidence"].get("match_percentage", 50)
        verdict = dict(MOCK_VERDICT)
        if match_pct >= 80:
            verdict["verdict"] = "HIRE"
            verdict["confidence"] = 0.82
        elif match_pct >= 60:
            verdict["verdict"] = "LEAN_HIRE"
            verdict["confidence"] = 0.68
        elif match_pct >= 40:
            verdict["verdict"] = "LEAN_NO_HIRE"
            verdict["confidence"] = 0.62
        else:
            verdict["verdict"] = "NO_HIRE"
            verdict["confidence"] = 0.75
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the JUDGE in a structured hiring debate.
Render a FINAL VERDICT after listening to all rounds.

VERDICT OPTIONS (choose exactly one):
STRONG_HIRE | HIRE | LEAN_HIRE | LEAN_NO_HIRE | NO_HIRE | STRONG_NO_HIRE

Return ONLY valid JSON:
{{
    "verdict": "<option>",
    "confidence": <float 0-1>,
    "reasoning": "detailed reasoning",
    "advocate_strength": <float 0-1>,
    "critic_strength": <float 0-1>,
    "key_factors": ["factor1", ...],
    "conditions": ["condition1", ...]
}}"""),
            ("human", "EVIDENCE:\n{evidence}\n\nDEBATE:\n{debate}\n\nFAIRNESS:\n{fairness}\n\nReturn JSON only."),
        ])

        llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.2,
        )
        chain = prompt | llm | JsonOutputParser()
        debate_summary = [
            {"round": i + 1, "advocate": a, "critic": c}
            for i, (a, c) in enumerate(zip(state["advocate_args"], state["critic_args"]))
        ]
        verdict = await chain.ainvoke({
            "evidence": json.dumps(state["evidence"], indent=2, default=str),
            "debate": json.dumps(debate_summary, indent=2, default=str),
            "fairness": json.dumps(state["fairness_reviews"], indent=2, default=str),
        })

    log.add_entry(
        agent_name="Orchestrator", role="orchestrator",
        round_number=state["current_round"],
        content=json.dumps(verdict, indent=2, default=str),
        entry_type="verdict",
    )
    return {"verdict": verdict}


def build_debate_graph() -> StateGraph:
    graph = StateGraph(DebateState)
    graph.add_node("advocate", advocate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("fairness", fairness_node)
    graph.add_node("judge", judge_node)
    graph.add_edge(START, "advocate")
    graph.add_edge("advocate", "critic")
    graph.add_edge("critic", "fairness")
    graph.add_conditional_edges(
        "fairness",
        should_continue,
        {"continue": "advocate", "judge": "judge"},
    )
    graph.add_edge("judge", END)
    return graph.compile()


async def run_debate(evidence_bundle: dict, max_rounds: int | None = None) -> dict:
    log = ConversationLog()
    rounds = max_rounds or settings.DEBATE_ROUNDS

    log.add_entry(
        agent_name="System", role="orchestrator", round_number=0,
        content=f"Debate initiated. Max rounds: {rounds}.",
        entry_type="meta",
    )

    compiled = build_debate_graph()
    initial_state: DebateState = {
        "evidence": evidence_bundle,
        "current_round": 1,
        "max_rounds": rounds,
        "advocate_args": [],
        "critic_args": [],
        "fairness_reviews": [],
        "verdict": None,
        "conversation_log": log,
    }

    final_state = await compiled.ainvoke(initial_state)
    return {
        "verdict": final_state["verdict"],
        "advocate_args": final_state["advocate_args"],
        "critic_args": final_state["critic_args"],
        "fairness_reviews": final_state["fairness_reviews"],
        "conversation_log": log,
    }
