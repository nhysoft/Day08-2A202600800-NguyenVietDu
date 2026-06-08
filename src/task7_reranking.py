"""
Task 7 — Reranking Module.

- rerank_cross_encoder: keyword overlap + original score (không cần model ngoài)
- rerank_mmr: Maximal Marginal Relevance
- rerank_rrf: Reciprocal Rank Fusion
"""

import re
import math
from typing import Optional


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank bằng keyword overlap score kết hợp với original score.
    """
    query_terms = set(re.findall(r'\w+', query.lower()))

    scored = []
    for c in candidates:
        content_terms = set(re.findall(r'\w+', c["content"].lower()))
        overlap = len(query_terms & content_terms)
        overlap_ratio = overlap / max(len(query_terms), 1)
        new_score = c.get("score", 0.0) * 0.6 + overlap_ratio * 0.4
        scored.append({**c, "score": round(new_score, 6)})

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — relevance vs diversity trade-off.
    """
    if not candidates:
        return []

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            emb = candidates[idx].get("embedding", [])
            if not emb:
                relevance = candidates[idx].get("score", 0.0)
            else:
                relevance = _cosine_sim(query_embedding, emb)

            max_sim = 0.0
            for sel in selected_indices:
                sel_emb = candidates[sel].get("embedding", [])
                if emb and sel_emb:
                    max_sim = max(max_sim, _cosine_sim(emb, sel_emb))

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.
    RRF(d) = Σ 1 / (k + rank_r(d))
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2)
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")
