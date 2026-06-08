# RAG Evaluation Results

**Ngày đánh giá:** 2026-06-08 16:15
**Golden dataset:** 20 câu hỏi
**LLM Judge:** Gemini 2.0 Flash

---

## Overall Scores

| Config | Faithfulness | Answer Relevance | Context Recall | Context Precision |
|--------|-------------|-----------------|---------------|------------------|
| hybrid_rerank | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_no_rerank | 0.000 | 0.000 | 0.000 | 0.000 |

---

## A/B Comparison

So sánh **hybrid_rerank** vs **hybrid_no_rerank**:

- **faithfulness**: hybrid_rerank: 0.000 vs hybrid_no_rerank: 0.000 (+0.000) ⚠️
- **answer_relevance**: hybrid_rerank: 0.000 vs hybrid_no_rerank: 0.000 (+0.000) ⚠️
- **context_recall**: hybrid_rerank: 0.000 vs hybrid_no_rerank: 0.000 (+0.000) ⚠️
- **context_precision**: hybrid_rerank: 0.000 vs hybrid_no_rerank: 0.000 (+0.000) ⚠️

---

## Per-Item Breakdown

### hybrid_rerank

| # | Question | Expected Context | Faithfulness | Relevance | Recall | Precision |
|---|---------|----------------|-------------|----------|--------|----------|

### hybrid_no_rerank

| # | Question | Expected Context | Faithfulness | Relevance | Recall | Precision |
|---|---------|----------------|-------------|----------|--------|----------|

---

## Worst Performers


---

## Recommendations

Dựa trên kết quả evaluation, đề xuất cải tiến:

1. **Cải thiện chunk size**: Điều chỉnh CHUNK_SIZE để tối ưu context precision.
2. **Tối ưu embedding model**: Dùng model chuyên biệt cho tiếng Việt pháp lý.
3. **Cải thiện reranking**: Thử nghiệm cross-encoder thay vì heuristic.
4. **Mở rộng golden dataset**: Thêm nhiều edge cases để đánh giá chính xác hơn.
5. **Fine-tune BM25 parameters**: Điều chỉnh k1 và b cho văn bản pháp lý.
