"""
Weighted Skill Graph — builds analysis from JD vs candidate skill matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import networkx as nx
from core.vector_engine import SkillMatch


@dataclass
class ExperienceComparison:
    skill: str
    required_years: float
    actual_years: float
    meets_requirement: bool
    reasoning: str


@dataclass
class SkillGraphAnalysis:
    matched_skills: list[SkillMatch]
    missing_skills: list[str]
    extra_skills: list[str]
    match_percentage: float
    experience_comparisons: list[ExperienceComparison]
    reasoning: list[str]
    graph: nx.DiGraph


def _extract_jd_skills(jd: dict) -> list[str]:
    skills: list[str] = []
    for key in ("skills", "required_skills", "preferred_skills", "must_have", "nice_to_have"):
        if isinstance(jd.get(key), list):
            for s in jd[key]:
                if isinstance(s, str):
                    skills.append(s)
                elif isinstance(s, dict):
                    skills.append(s.get("name", s.get("skill", "")))
    return [s for s in skills if s]


def _extract_candidate_skills(candidate: dict) -> list[str]:
    skills: list[str] = []
    for key in ("skills", "technical_skills", "technologies", "tech_stack", "competencies"):
        if isinstance(candidate.get(key), list):
            for s in candidate[key]:
                if isinstance(s, str):
                    skills.append(s)
                elif isinstance(s, dict):
                    skills.append(s.get("name", s.get("skill", "")))
    return [s for s in skills if s]


def _extract_experience_map(data: dict) -> dict[str, float]:
    exp_map: dict[str, float] = {}
    for key in ("skills", "required_skills", "technical_skills", "technologies",
                "preferred_skills", "must_have", "nice_to_have"):
        items = data.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                name = item.get("name", item.get("skill", ""))
                years = item.get("years", item.get("experience_years", item.get("yoe", 0)))
                if name:
                    try:
                        exp_map[name.lower()] = float(years)
                    except (ValueError, TypeError):
                        exp_map[name.lower()] = 0.0
    return exp_map


def build_skill_graph(
    jd: dict,
    candidate: dict,
    skill_matches: list[SkillMatch],
) -> SkillGraphAnalysis:
    G = nx.DiGraph()
    reasoning: list[str] = []

    jd_skills_raw = _extract_jd_skills(jd)
    cand_skills_raw = _extract_candidate_skills(candidate)
    jd_exp = _extract_experience_map(jd)
    cand_exp = _extract_experience_map(candidate)

    matched_jd = set()
    matched_cand = set()
    matched_list: list[SkillMatch] = []
    missing_list: list[str] = []

    for match in skill_matches:
        G.add_node(match.jd_skill, type="jd_requirement")
        if match.candidate_skill:
            G.add_node(match.candidate_skill, type="candidate_skill")
            G.add_edge(match.jd_skill, match.candidate_skill, weight=match.similarity)
            matched_jd.add(match.jd_skill.lower())
            matched_cand.add(match.candidate_skill.lower())
            matched_list.append(match)
            if match.similarity >= 0.9:
                reasoning.append(f"✅ Strong match: '{match.jd_skill}' ↔ '{match.candidate_skill}' (sim={match.similarity:.2f})")
            elif match.similarity >= 0.7:
                reasoning.append(f"🟡 Moderate match: '{match.jd_skill}' ↔ '{match.candidate_skill}' (sim={match.similarity:.2f})")
            else:
                reasoning.append(f"🟠 Weak match: '{match.jd_skill}' ↔ '{match.candidate_skill}' (sim={match.similarity:.2f})")
        else:
            missing_list.append(match.jd_skill)
            reasoning.append(f"❌ Missing: '{match.jd_skill}' — no matching skill found")

    extra_skills = [s for s in cand_skills_raw if s.lower() not in matched_cand]
    for extra in extra_skills:
        G.add_node(extra, type="extra_skill")
        reasoning.append(f"➕ Extra skill: '{extra}'")

    experience_comparisons: list[ExperienceComparison] = []
    for match in matched_list:
        jd_years = jd_exp.get(match.jd_skill.lower(), 0)
        cand_years = cand_exp.get(match.candidate_skill.lower(), 0)
        if jd_years > 0:
            meets = cand_years >= jd_years
            exp_reasoning = (
                f"📊 '{match.jd_skill}': requires {jd_years}yr, candidate has {cand_years}yr"
                f" → {'✅ meets' if meets else f'❌ shortfall of {jd_years - cand_years:.1f}yr'}"
            )
            reasoning.append(exp_reasoning)
            experience_comparisons.append(ExperienceComparison(
                skill=match.jd_skill,
                required_years=jd_years,
                actual_years=cand_years,
                meets_requirement=meets,
                reasoning=exp_reasoning,
            ))

    total_jd = len(jd_skills_raw) if jd_skills_raw else 1
    match_pct = (len(matched_list) / total_jd) * 100
    reasoning.insert(0, f"📈 Overall skill match: {match_pct:.1f}% ({len(matched_list)}/{total_jd} JD skills covered)")

    return SkillGraphAnalysis(
        matched_skills=matched_list,
        missing_skills=missing_list,
        extra_skills=extra_skills,
        match_percentage=round(match_pct, 1),
        experience_comparisons=experience_comparisons,
        reasoning=reasoning,
        graph=G,
    )
