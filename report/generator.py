"""
Report Generator — builds the final structured evaluation report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agents.conversation_log import ConversationLog


def build_report(
    jd: dict,
    anonymized_candidate: dict,
    evidence_bundle: dict,
    debate_result: dict,
) -> dict:
    log: ConversationLog = debate_result["conversation_log"]
    verdict = debate_result.get("verdict", {})

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "job_title": jd.get("title", jd.get("job_title", "Unknown Role")),
            "debate_rounds": len(debate_result.get("advocate_args", [])),
            "pipeline_version": "2.0.0",
        },
        "candidate_summary": {
            "profile": _summarize_candidate(anonymized_candidate),
            "note": "All personal identifiers have been redacted to prevent bias.",
        },
        "skill_analysis": {
            "overall_similarity": evidence_bundle.get("overall_similarity", 0),
            "match_percentage": evidence_bundle.get("match_percentage", 0),
            "matched_skills": [
                {"jd_skill": m["jd_skill"], "candidate_skill": m["candidate_skill"], "similarity": m["similarity"]}
                for m in evidence_bundle.get("skill_matches", [])
                if m.get("candidate_skill")
            ],
            "missing_skills": evidence_bundle.get("missing_skills", []),
            "extra_skills": evidence_bundle.get("extra_skills", []),
            "experience_analysis": evidence_bundle.get("experience_analysis", []),
            "reasoning": evidence_bundle.get("graph_reasoning", []),
        },
        "debate_summary": {
            "advocate": {"rounds": len(debate_result.get("advocate_args", [])), "arguments": debate_result.get("advocate_args", [])},
            "critic": {"rounds": len(debate_result.get("critic_args", [])), "arguments": debate_result.get("critic_args", [])},
            "fairness_reviews": debate_result.get("fairness_reviews", []),
        },
        "verdict": verdict,
        "immutable_log": log.export_dict(),
    }


def _summarize_candidate(candidate: dict) -> dict:
    safe_keys = [
        "skills", "technical_skills", "technologies", "tech_stack",
        "experience", "work_experience", "projects", "certifications",
        "education_level", "years_of_experience", "total_experience",
        "summary", "professional_summary", "capabilities", "competencies",
        "downsides", "weaknesses",
    ]
    return {key: candidate[key] for key in safe_keys if key in candidate}


def format_report_markdown(report: dict) -> str:
    lines: list[str] = []
    meta = report["metadata"]
    lines += [
        f"# 📋 Candidate Evaluation Report",
        f"**Role:** {meta['job_title']}",
        f"**Generated:** {meta['generated_at']}",
        f"**Debate Rounds:** {meta['debate_rounds']}",
        "",
    ]

    verdict = report.get("verdict", {})
    verdict_val = verdict.get("verdict", "UNKNOWN")
    confidence = verdict.get("confidence", 0)
    verdict_emoji = {
        "STRONG_HIRE": "🟢🟢", "HIRE": "🟢", "LEAN_HIRE": "🟡",
        "LEAN_NO_HIRE": "🟠", "NO_HIRE": "🔴", "STRONG_NO_HIRE": "🔴🔴",
    }.get(verdict_val, "⚪")

    lines += [
        f"## {verdict_emoji} Verdict: **{verdict_val}**",
        f"**Confidence:** {confidence:.0%}",
        "",
    ]
    if verdict.get("reasoning"):
        lines += [verdict["reasoning"], ""]
    if verdict.get("key_factors"):
        lines.append("### Key Factors")
        for f in verdict["key_factors"]:
            lines.append(f"- {f}")
        lines.append("")

    skill = report["skill_analysis"]
    lines += [
        "## 📊 Skill Analysis",
        f"- **Overall Similarity:** {skill['overall_similarity']:.2f}",
        f"- **Match Percentage:** {skill['match_percentage']:.1f}%",
        "",
    ]

    if skill["matched_skills"]:
        lines += ["### ✅ Matched Skills", "| JD Skill | Candidate Skill | Score |", "|---|---|---|"]
        for m in skill["matched_skills"]:
            bar = "█" * int(m["similarity"] * 10)
            lines.append(f"| {m['jd_skill']} | {m['candidate_skill']} | {m['similarity']:.2f} {bar} |")
        lines.append("")

    if skill["missing_skills"]:
        lines.append("### ❌ Missing Skills")
        for s in skill["missing_skills"]:
            lines.append(f"- {s}")
        lines.append("")

    if skill["extra_skills"]:
        lines.append("### ➕ Extra Skills")
        for s in skill["extra_skills"]:
            lines.append(f"- {s}")
        lines.append("")

    lines.append("## 🗣 Debate Summary")
    for i, (adv, crit) in enumerate(zip(
        report["debate_summary"]["advocate"]["arguments"],
        report["debate_summary"]["critic"]["arguments"],
    ), 1):
        lines.append(f"### Round {i}")
        lines.append("**🟢 Advocate:**")
        for arg in (adv.get("key_arguments") or [json.dumps(adv, default=str)]):
            lines.append(f"- {arg}")
        lines.append("**🔴 Critic:**")
        for arg in (crit.get("key_arguments") or crit.get("risks_identified") or [json.dumps(crit, default=str)]):
            lines.append(f"- {arg}")
        lines.append("")

    if report["debate_summary"]["fairness_reviews"]:
        lines.append("### ⚖️ Fairness Assessment")
        for i, review in enumerate(report["debate_summary"]["fairness_reviews"], 1):
            if isinstance(review, dict):
                if review.get("balance_assessment"):
                    lines.append(f"- **Balance:** {review['balance_assessment']}")
                if review.get("fairness_score") is not None:
                    lines.append(f"- **Fairness Score:** {review['fairness_score']:.0%}")
        lines.append("")

    lines += [
        "## 👤 Candidate Summary (Anonymized)",
        f"_{report['candidate_summary']['note']}_",
        "",
        "```json",
        json.dumps(report["candidate_summary"]["profile"], indent=2, default=str),
        "```",
        "",
        "## 🔒 Immutable Debate Log",
        f"_{len(report['immutable_log'])} entries recorded._",
    ]

    return "\n".join(lines)
