"""
FAISS Vector Similarity Engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import settings


@dataclass
class SkillMatch:
    jd_skill: str
    candidate_skill: str
    similarity: float


class VectorEngine:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors.astype(np.float32)

    def compute_overall_similarity(
        self,
        jd_texts: list[str],
        candidate_texts: list[str],
    ) -> float:
        if not jd_texts or not candidate_texts:
            return 0.0
        jd_vecs = self.embed(jd_texts)
        cand_vecs = self.embed(candidate_texts)
        jd_mean = jd_vecs.mean(axis=0, keepdims=True)
        cand_mean = cand_vecs.mean(axis=0, keepdims=True)
        faiss.normalize_L2(jd_mean)
        faiss.normalize_L2(cand_mean)
        return float(np.dot(jd_mean[0], cand_mean[0]))

    def find_skill_matches(
        self,
        jd_skills: list[str],
        candidate_skills: list[str],
        threshold: float = 0.3,
    ) -> list[SkillMatch]:
        if not jd_skills or not candidate_skills:
            return []
        jd_vecs = self.embed(jd_skills)
        cand_vecs = self.embed(candidate_skills)
        dim = cand_vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(cand_vecs)
        scores, indices = index.search(jd_vecs, k=1)
        matches: list[SkillMatch] = []
        for i, jd_skill in enumerate(jd_skills):
            sim_score = float(scores[i][0])
            best_idx = int(indices[i][0])
            candidate_skill = candidate_skills[best_idx] if sim_score >= threshold else ""
            matches.append(SkillMatch(
                jd_skill=jd_skill,
                candidate_skill=candidate_skill,
                similarity=round(sim_score, 4),
            ))
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches


_engine: VectorEngine | None = None


def get_vector_engine() -> VectorEngine:
    global _engine
    if _engine is None:
        _engine = VectorEngine()
    return _engine
