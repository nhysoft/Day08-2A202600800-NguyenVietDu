"""
Task 6 — Lexical Search Module (BM25).

BM25 tự implement bằng stdlib, không cần rank-bm25.
Formula: score(q,d) = Σ IDF(qi) * tf(qi,d)*(k1+1) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
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


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Returns:
        List of {'content', 'score', 'metadata'} sorted descending.
    """
    corpus = _load_corpus()
    if not corpus:
        return []

    k1, b = 1.5, 0.75
    tokenized = [_tokenize(c["content"]) for c in corpus]
    N = len(tokenized)
    avg_dl = sum(len(t) for t in tokenized) / N if N > 0 else 1.0

    df: dict[str, int] = {}
    for tokens in tokenized:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    tf_list = []
    for tokens in tokenized:
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        tf_list.append(freq)

    query_tokens = _tokenize(query)
    scores = []

    for doc_tf, doc_tokens in zip(tf_list, tokenized):
        score = 0.0
        dl = len(doc_tokens)
        for term in query_tokens:
            if term not in df:
                continue
            f = doc_tf.get(term, 0)
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            tf_score = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_score
        scores.append(score)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "content": corpus[idx]["content"],
            "score": scores[idx],
            "metadata": corpus[idx].get("metadata", {}),
        }
        for idx in top_indices
        if scores[idx] > 0
    ]


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
