"""
Skill Taxonomy Normalization.
"""

from __future__ import annotations
import re

SKILL_ALIASES: dict[str, str] = {
    "nodejs": "node.js", "node": "node.js", "reactjs": "react",
    "react.js": "react", "vuejs": "vue", "vue.js": "vue",
    "angularjs": "angular", "nextjs": "next.js", "expressjs": "express",
    "typescript": "typescript", "ts": "typescript", "js": "javascript",
    "python3": "python", "py": "python",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "pytorch": "pytorch", "torch": "pytorch",
    "tensorflow": "tensorflow", "tf": "tensorflow",
    "postgres": "postgresql", "psql": "postgresql",
    "mongo": "mongodb", "k8s": "kubernetes", "kube": "kubernetes",
    "aws": "aws", "amazon web services": "aws",
    "gcp": "google cloud", "google cloud platform": "google cloud",
    "golang": "go", "c++": "cpp", "cplusplus": "cpp",
    "c#": "csharp", "dotnet": ".net",
    "ci/cd": "ci/cd", "cicd": "ci/cd",
    "rest api": "rest", "restful": "rest",
    "ml": "machine learning", "dl": "deep learning",
    "nlp": "nlp", "natural language processing": "nlp",
    "langchain": "langchain", "langgraph": "langgraph",
    "fastapi": "fastapi", "django": "django", "flask": "flask",
    "docker compose": "docker compose", "terraform": "terraform",
    "redis": "redis", "kafka": "kafka", "spark": "spark",
    "graphql": "graphql", "grpc": "grpc",
    "git": "git", "github": "github", "gitlab": "gitlab",
    "linux": "linux", "bash": "bash", "shell": "bash",
}


def normalize(skill: str) -> str:
    cleaned = skill.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return SKILL_ALIASES.get(cleaned, cleaned)


def normalize_list(skills: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for skill in skills:
        canonical = normalize(skill)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def add_alias(alias: str, canonical: str) -> None:
    SKILL_ALIASES[alias.strip().lower()] = canonical.strip().lower()
