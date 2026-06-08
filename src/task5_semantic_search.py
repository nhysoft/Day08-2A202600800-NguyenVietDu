"""
Task 5 — Semantic Search Module.

TF-IDF cosine similarity trên corpus chunks từ Task 4.
"""

import math
import re

_CORPUS: list[dict] = []


def _load_corpus() -> list[dict]:
    global _CORPUS
    if _CORPUS:
        return _CORPUS
    try:
        from .task4_chunking_indexing import load_documents, chunk_documents
        docs = load_documents()
        if docs:
            _CORPUS = chunk_documents(docs)
    except Exception:
        pass
    return _CORPUS


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def _cosine_tfidf(query_tokens: list[str], doc_tokens: list[str],
                  df: dict, N: int) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    def vec(tokens):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        result = {}
        for t, freq in tf.items():
            idf = math.log((N + 1) / (df.get(t, 0) + 1)) + 1
            result[t] = (freq / len(tokens)) * idf
        return result

    q_vec = vec(query_tokens)
    d_vec = vec(doc_tokens)

    dot = sum(q_vec.get(t, 0) * d_vec.get(t, 0) for t in q_vec)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
    d_norm = math.sqrt(sum(v * v for v in d_vec.values()))

    if q_norm == 0 or d_norm == 0:
        return 0.0
    return dot / (q_norm * d_norm)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng TF-IDF cosine similarity.

    Returns:
        List of {'content', 'score', 'metadata'} sorted descending.
    """
    corpus = _load_corpus()
    if not corpus:
        return []

    tokenized = [_tokenize(c["content"]) for c in corpus]
    N = len(tokenized)

    df: dict[str, int] = {}
    for tokens in tokenized:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    query_tokens = _tokenize(query)
    scores = [_cosine_tfidf(query_tokens, dt, df, N) for dt in tokenized]

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "content": corpus[idx]["content"],
            "score": scores[idx],
            "metadata": corpus[idx].get("metadata", {}),
        }
        for idx in top_indices
    ]


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
