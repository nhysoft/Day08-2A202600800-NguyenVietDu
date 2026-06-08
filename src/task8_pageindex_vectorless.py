"""
Task 8 — PageIndex Vectorless RAG.

Fallback retrieval đọc trực tiếp từ data/standardized/ bằng keyword scoring,
đánh dấu kết quả với source='pageindex'.
"""

import os
import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval: keyword frequency scoring trên markdown files.

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}
    """
    if not STANDARDIZED_DIR.exists():
        return []

    query_tokens = re.findall(r'\w+', query.lower())
    if not query_tokens:
        return []

    results = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_tokens = re.findall(r'\w+', content.lower())
        if not doc_tokens:
            continue

        tf: dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1

        score = sum(tf.get(t, 0) for t in query_tokens) / len(doc_tokens)

        if score > 0:
            results.append({
                "content": content[:500],
                "score": score,
                "metadata": {
                    "source": md_file.name,
                    "type": md_file.parent.name,
                    "path": str(md_file),
                },
                "source": "pageindex",
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.4f}] {r['content'][:100]}...")
