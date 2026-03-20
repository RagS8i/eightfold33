"""
Orchestrator — LangGraph StateGraph running the multi-agent debate.
Flow: advocate → critic → fairness → (loop N rounds) → judge
All agents use live LLM reasoning.
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

    context = "Opening round — make your strongest case for hiring."
    if state["critic_args"]:
        last_critic = state["critic_args"][-1]
        context = (
            f"Round {round_num}. The Critic argued:\n"
            f"{json.dumps(last_critic, indent=2, default=str)}\n\n"
            f"Directly rebut their specific claims using evidence."
        )
        if state["fairness_reviews"]:
            context += (
                f"\n\nFairness review noted:\n"
                f"{json.dumps(state['fairness_reviews'][-1], indent=2, default=str)}"
            )

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
    context = (
        f"Round {round_num}. The Advocate argued:\n"
        f"{json.dumps(last_advocate, indent=2, default=str)}\n\n"
        f"Directly challenge their specific claims using evidence."
    )
    if state["fairness_reviews"]:
        context += (
            f"\n\nFairness review noted:\n"
            f"{json.dumps(state['fairness_reviews'][-1], indent=2, default=str)}"
        )

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


async def judge_node(state: DebateState) -> dict:
    log: ConversationLog = state["conversation_log"]

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("No GEMINI_API_KEY set. Please add your API key in the sidebar.")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the JUDGE in a structured hiring debate.
After listening to all debate rounds, render a final hiring verdict.

Your verdict must be grounded in:
1. The raw evidence bundle (FAISS similarity scores, skill gaps, experience data)
2. The quality of arguments made by both agents
3. The fairness agent's fact-checks and recommendations

VERDICT OPTIONS (choose exactly one):
STRONG_HIRE    — Exceptional fit, hire with high confidence
HIRE           — Good fit, recommend hiring
LEAN_HIRE      — More strengths than weaknesses, lean toward hiring
LEAN_NO_HIRE   — More weaknesses than strengths, lean against hiring
NO_HIRE        — Poor fit, do not recommend hiring
STRONG_NO_HIRE — Very poor fit, strongly recommend against hiring

Rules:
1. Cite specific evidence in your reasoning.
2. Acknowledge the strongest argument from each side.
3. Your confidence should reflect genuine uncertainty where it exists.
4. List concrete conditions under which the verdict would change.

Return ONLY valid JSON, no markdown, no explanation:
{{
    "verdict": "<one of the six options>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "detailed paragraph citing specific evidence and debate arguments",
    "advocate_strength": <float 0.0-1.0>,
    "critic_strength": <float 0.0-1.0>,
    "key_factors": ["factor 1", "factor 2", ...],
    "conditions": ["condition that would improve verdict", "condition that would worsen verdict", ...]
}}"""),
        ("human", (
            "EVIDENCE BUNDLE:\n{evidence}\n\n"
            "FULL DEBATE (all rounds):\n{debate}\n\n"
            "FAIRNESS REVIEWS:\n{fairness}\n\n"
            "Return JSON only."
        )),
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
    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY in your .env file "
            "or paste it in the sidebar."
        )

    log = ConversationLog()
    rounds = max_rounds or settings.DEBATE_ROUNDS

    log.add_entry(
        agent_name="System", role="orchestrator", round_number=0,
        content=f"Debate initiated. Max rounds: {rounds}. Mode: Live Gemini LLM.",
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