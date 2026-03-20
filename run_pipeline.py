"""
CLI Pipeline Runner — runs the full evaluation against sample data
and prints the Markdown report to stdout.
"""

import asyncio
import json
import os
import sys

# ── Force mock mode ON before anything else loads ────────────────────────────
os.environ["USE_MOCK_LLM"] = "true"

# ── Load .env (will NOT override the above because we set it first) ───────────
from dotenv import load_dotenv
load_dotenv()  # no override=True, so our forced value above is preserved

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from core.anonymizer import anonymize
from core.skill_taxonomy import normalize_list
from core.vector_engine import get_vector_engine
from core.skill_graph import build_skill_graph, _extract_jd_skills, _extract_candidate_skills
from agents.orchestrator import run_debate
from report.generator import build_report, format_report_markdown


async def run_pipeline(jd: dict, candidate: dict) -> dict:
    """Core pipeline — reusable by both CLI and Streamlit."""

    anonymized = await anonymize(candidate)

    jd_skills = normalize_list(_extract_jd_skills(jd))
    cand_skills = normalize_list(_extract_candidate_skills(anonymized))

    engine = get_vector_engine()
    overall_sim = engine.compute_overall_similarity(jd_skills, cand_skills)
    skill_matches = engine.find_skill_matches(jd_skills, cand_skills)

    analysis = build_skill_graph(jd, anonymized, skill_matches)

    evidence_bundle = {
        "overall_similarity": round(overall_sim, 4),
        "match_percentage": analysis.match_percentage,
        "skill_matches": [
            {"jd_skill": m.jd_skill, "candidate_skill": m.candidate_skill, "similarity": m.similarity}
            for m in analysis.matched_skills
        ],
        "missing_skills": analysis.missing_skills,
        "extra_skills": analysis.extra_skills,
        "experience_analysis": [
            {"skill": e.skill, "required": e.required_years, "actual": e.actual_years,
             "meets": e.meets_requirement, "reasoning": e.reasoning}
            for e in analysis.experience_comparisons
        ],
        "graph_reasoning": analysis.reasoning,
        "jd_skills": jd_skills,
        "candidate_skills": cand_skills,
    }

    # Mock mode: instant, no API calls ever
    debate_result = await run_debate(evidence_bundle)

    report = build_report(jd, anonymized, evidence_bundle, debate_result)
    report["_skill_matches_raw"] = [
        {"jd_skill": m.jd_skill, "candidate_skill": m.candidate_skill, "similarity": m.similarity}
        for m in skill_matches
    ]
    return report


async def main():
    print("=" * 70)
    print("🤖 AGENTIC CANDIDATE EVALUATOR — CLI Pipeline")
    print("=" * 70)
    print()
    print("🎭 Mode: MOCK  (pre-canned AI responses, no API calls)")
    print()

    base = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base, "sample_data", "sample_jd.json")) as f:
        jd = json.load(f)
    with open(os.path.join(base, "sample_data", "sample_candidate.json")) as f:
        candidate = json.load(f)

    print(f"📄 Job:       {jd.get('title', 'Unknown')} @ {jd.get('company', 'N/A')}")
    print(f"👤 Candidate: {candidate.get('name', 'Unknown')}")
    print()
    print("🔒 Step 1: Anonymizing...")
    print("🔧 Step 2: Normalizing skills...")
    print("🔢 Step 3: FAISS similarities (local)...")
    print("📊 Step 4: Skill graph analysis...")
    print("🗣  Step 5: Multi-agent debate (mock)...")
    print("📋 Step 6: Building report...")
    print()

    report = await run_pipeline(jd, candidate)
    report_md = format_report_markdown(report)

    print("=" * 70)
    print(report_md)
    print("=" * 70)

    out_md   = os.path.join(base, "evaluation_report.md")
    out_json = os.path.join(base, "evaluation_report.json")
    out_log  = os.path.join(base, "debate_transcript.md")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report_md)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in report.items() if k != "conversation_log"}, f, indent=2, default=str)

    from agents.conversation_log import ConversationLog
    log = ConversationLog()
    for entry in report.get("immutable_log", []):
        log.add_entry(
            agent_name=entry["agent_name"], role=entry["role"],
            round_number=entry["round_number"], content=entry["content"],
            entry_type=entry["entry_type"],
        )
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(log.export_markdown())

    verdict = report.get("verdict", {})
    print(f"\n✅ Complete!")
    print(f"   Verdict:     {verdict.get('verdict', 'N/A')}  ({verdict.get('confidence', 0):.0%} confidence)")
    print(f"   Skill match: {report['skill_analysis']['match_percentage']:.1f}%")
    print(f"   Similarity:  {report['skill_analysis']['overall_similarity']:.3f}")
    print(f"\n💾 evaluation_report.md")
    print(f"💾 evaluation_report.json")
    print(f"📜 debate_transcript.md")


if __name__ == "__main__":
    asyncio.run(main())