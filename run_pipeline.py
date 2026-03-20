"""
CLI Pipeline Runner — runs the full evaluation against sample data
and prints the Markdown report to stdout.
"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.anonymizer import anonymize
from core.skill_taxonomy import normalize_list
from core.vector_engine import get_vector_engine
from core.skill_graph import build_skill_graph, _extract_jd_skills, _extract_candidate_skills
from agents.orchestrator import run_debate
from report.generator import build_report, format_report_markdown


async def run_pipeline(jd: dict, candidate: dict) -> dict:
    """Core pipeline logic — reusable by both CLI and Streamlit."""

    # Step 1: Anonymize
    anonymized = await anonymize(candidate)

    # Step 2: Normalize skills
    jd_skills = normalize_list(_extract_jd_skills(jd))
    cand_skills = normalize_list(_extract_candidate_skills(anonymized))

    # Step 3: FAISS similarity
    engine = get_vector_engine()
    overall_sim = engine.compute_overall_similarity(jd_skills, cand_skills)
    skill_matches = engine.find_skill_matches(jd_skills, cand_skills)

    # Step 4: Skill graph
    analysis = build_skill_graph(jd, anonymized, skill_matches)

    # Step 5: Evidence bundle
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
            {
                "skill": e.skill, "required": e.required_years,
                "actual": e.actual_years, "meets": e.meets_requirement,
                "reasoning": e.reasoning,
            }
            for e in analysis.experience_comparisons
        ],
        "graph_reasoning": analysis.reasoning,
        "jd_skills": jd_skills,
        "candidate_skills": cand_skills,
    }

    # Step 6: Multi-agent debate (with retry on rate limit)
    debate_result = None
    for attempt in range(3):
        try:
            debate_result = await run_debate(evidence_bundle)
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 40 * (attempt + 1)
                print(f"   ⚠️  Rate limited. Waiting {wait}s (attempt {attempt + 1}/3)...")
                await asyncio.sleep(wait)
                if attempt == 2:
                    raise
            else:
                raise

    # Step 7: Build report
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

    base = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base, "sample_data", "sample_jd.json")) as f:
        jd = json.load(f)
    with open(os.path.join(base, "sample_data", "sample_candidate.json")) as f:
        candidate = json.load(f)

    print(f"📄 Job:       {jd.get('title', 'Unknown')}")
    print(f"👤 Candidate: {candidate.get('name', 'Unknown')}")
    print()

    print("🔒 Step 1: Anonymizing...")
    print("🔧 Step 2: Normalizing skills...")
    print("🔢 Step 3: Computing FAISS similarities...")
    print("📊 Step 4: Building skill graph...")
    print("🗣  Step 5: Running multi-agent debate...")
    print("📋 Step 6: Generating report...")
    print()

    report = await run_pipeline(jd, candidate)
    report_md = format_report_markdown(report)

    print("=" * 70)
    print(report_md)
    print("=" * 70)

    # Save outputs
    out_md = os.path.join(base, "evaluation_report.md")
    out_json = os.path.join(base, "evaluation_report.json")
    out_log = os.path.join(base, "debate_transcript.md")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n💾 Report:     {out_md}")

    with open(out_json, "w", encoding="utf-8") as f:
        # Remove non-serializable objects before saving
        save_report = {k: v for k, v in report.items() if k != "conversation_log"}
        json.dump(save_report, f, indent=2, default=str)
    print(f"💾 JSON:       {out_json}")

    # Export transcript from the debate result
    from agents.orchestrator import run_debate as _  # noqa
    # The conversation log is inside the report's immutable_log
    from agents.conversation_log import ConversationLog
    log = ConversationLog()
    for entry in report.get("immutable_log", []):
        log.add_entry(
            agent_name=entry["agent_name"],
            role=entry["role"],
            round_number=entry["round_number"],
            content=entry["content"],
            entry_type=entry["entry_type"],
        )
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(log.export_markdown())
    print(f"📜 Transcript: {out_log}")

    verdict = report.get("verdict", {})
    print(f"\n✅ Done. Verdict: {verdict.get('verdict', 'N/A')} (confidence: {verdict.get('confidence', 0):.0%})")


if __name__ == "__main__":
    asyncio.run(main())
