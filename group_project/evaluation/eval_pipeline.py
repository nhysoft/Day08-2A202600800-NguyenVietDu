"""
RAG Evaluation Pipeline.

Đánh giá RAG pipeline với 4 metrics từ RAGAS framework:
  - Faithfulness: câu trả lời có bám đúng context không?
  - Answer Relevance: câu trả lời có đúng câu hỏi không?
  - Context Recall: retriever có lấy đủ evidence không?
  - Context Precision: trong context lấy về, bao nhiêu % thực sự hữu ích?

Sử dụng Gemini API làm LLM judge thông qua RAGAS + langchain-google-genai.
So sánh A/B: hybrid+rerank vs hybrid (không rerank).

Cài đặt:
    pip install ragas datasets langchain-google-genai python-dotenv
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Fix Windows cp1252 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Add src to path để import từ project
SRC_DIR = Path(__file__).parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# =============================================================================
# HELPER
# =============================================================================

def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG PIPELINE
# =============================================================================

def run_rag_pipeline(question: str, use_reranking: bool = True) -> dict:
    """
    Chạy RAG pipeline cho 1 câu hỏi.
    Trả về dict: {answer, sources, retrieval_source}
    """
    from task10_generation import generate_with_citation
    return generate_with_citation(question, use_reranking=use_reranking)


# =============================================================================
# RAGAS EVALUATION
# =============================================================================

def _setup_ragas():
    """
    Cấu hình RAGAS với Gemini 2.0 Flash làm LLM judge.

    Returns:
        list[RAGAS metric instances]: faithfulness, answer_relevancy,
                                       context_recall, context_precision
    """
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # LLM judge
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )
    evaluator = LangchainLLMWrapper(llm)

    # Embeddings (cần cho answer_relevancy)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
    )
    emb_wrapper = LangchainEmbeddingsWrapper(embeddings)

    # Gán LLM cho tất cả metrics
    for metric in [faithfulness, answer_relevancy, context_recall, context_precision]:
        metric.llm = evaluator
    # answer_relevancy cần thêm embeddings để tính cosine similarity
    answer_relevancy.embeddings = emb_wrapper

    return [faithfulness, answer_relevancy, context_recall, context_precision]


def evaluate_config(
    rag_fn, golden_dataset: list[dict], config_name: str = "default"
) -> dict:
    """
    Evaluate RAG pipeline với RAGAS metrics trên toàn bộ golden dataset.

    Args:
        rag_fn: function(question) -> {answer, sources, retrieval_source}
        golden_dataset: list of {question, expected_answer, expected_context}

    Returns:
        dict with scores per metric + per-item scores
    """
    from datasets import Dataset
    from ragas import evaluate

    # --- Step 1: Run RAG pipeline trên từng câu hỏi ---
    eval_items = []
    for i, item in enumerate(golden_dataset):
        q = item["question"]
        print(f"  [{i+1}/{len(golden_dataset)}] {q[:50]}...", end=" ")

        try:
            result = rag_fn(q)
            eval_items.append({
                "question": q,
                "answer": result.get("answer", ""),
                "contexts": [c["content"] for c in result.get("sources", [])],
                "ground_truth": item["expected_answer"],
                "expected_context": item["expected_context"],
            })
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    if not eval_items:
        print("  ⚠ Không có item nào chạy thành công!")
        return {
            "config": config_name,
            "summary": {},
            "items": [],
            "num_success": 0,
        }

    # --- Step 2: Build HuggingFace Dataset cho RAGAS ---
    data = {
        "question": [it["question"] for it in eval_items],
        "answer": [it["answer"] for it in eval_items],
        "contexts": [it["contexts"] for it in eval_items],
        "ground_truth": [it["ground_truth"] for it in eval_items],
    }
    dataset = Dataset.from_dict(data)

    # --- Step 3: Chạy RAGAS evaluation ---
    print(f"  → Running RAGAS evaluation ({len(eval_items)} items)...")
    metrics = _setup_ragas()
    result = evaluate(dataset, metrics=metrics)
    df = result.to_pandas()

    # --- Step 4: Trích xuất scores ---
    # Map RAGAS column names → internal names
    col_map = {
        "faithfulness": "faithfulness",
        "answer_relevance": "answer_relevancy",
        "context_recall": "context_recall",
        "context_precision": "context_precision",
    }

    item_results = []
    all_scores = {k: [] for k in col_map}

    for idx, row in df.iterrows():
        scores = {}
        for out_key, ragas_col in col_map.items():
            val = row.get(ragas_col)
            score = float(val) if val is not None and not pd.isna(val) else 0.0
            scores[out_key] = score
            all_scores[out_key].append(score)

        item_results.append({
            "question": row.get("question", eval_items[idx]["question"]),
            "expected_context": eval_items[idx]["expected_context"],
            **scores,
        })

    # Compute averages
    summary = {
        m: round(sum(v) / len(v), 3) if v else 0.0
        for m, v in all_scores.items()
    }

    return {
        "config": config_name,
        "summary": summary,
        "items": item_results,
        "num_success": len(eval_items),
    }


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs:
      - Config A: hybrid search + reranking
      - Config B: hybrid search (không rerank)
    """
    configs = {
        "hybrid_rerank": lambda q: run_rag_pipeline(q, use_reranking=True),
        "hybrid_no_rerank": lambda q: run_rag_pipeline(q, use_reranking=False),
    }

    results = {}
    for name, fn in configs.items():
        print(f"\n{'='*60}")
        print(f"  Config: {name}")
        print(f"{'='*60}")
        results[name] = evaluate_config(fn, golden_dataset, config_name=name)

    return results


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_results(comparison: dict):
    """Export evaluation results ra results.md."""
    content = "# RAG Evaluation Results\n\n"
    content += f"**Ngày đánh giá:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    content += f"**Golden dataset:** {len(load_golden_dataset())} câu hỏi\n"
    content += f"**LLM Judge:** Gemini 2.0 Flash via RAGAS\n\n"
    content += "---\n\n"

    # Overall scores per config
    content += "## Overall Scores\n\n"
    content += "| Config | Faithfulness | Answer Relevance | Context Recall | Context Precision |\n"
    content += "|--------|-------------|-----------------|---------------|------------------|\n"

    for config_name, result in comparison.items():
        s = result["summary"]
        if s:
            content += f"| {config_name} | {s['faithfulness']:.3f} | {s['answer_relevance']:.3f} | {s['context_recall']:.3f} | {s['context_precision']:.3f} |\n"

    content += "\n---\n\n"

    # A/B Comparison detail
    content += "## A/B Comparison\n\n"

    names = list(comparison.keys())
    if len(names) >= 2:
        a, b = names[0], names[1]
        a_scores = comparison[a]["summary"]
        b_scores = comparison[b]["summary"]

        if a_scores and b_scores:
            content += f"So sánh **{a}** vs **{b}**:\n\n"
            for metric in ["faithfulness", "answer_relevance", "context_recall", "context_precision"]:
                diff = a_scores[metric] - b_scores[metric]
                emoji = "✅" if diff > 0.05 else "⚠️" if abs(diff) <= 0.05 else "❌"
                content += f"- **{metric}**: {a}: {a_scores[metric]:.3f} vs {b}: {b_scores[metric]:.3f} ({diff:+.3f}) {emoji}\n"
            content += "\n"

    content += "---\n\n"

    # Per-item breakdown
    content += "## Per-Item Breakdown\n\n"
    for config_name, result in comparison.items():
        content += f"### {config_name}\n\n"
        content += "| # | Question | Expected Context | Faithfulness | Relevance | Recall | Precision |\n"
        content += "|---|---------|----------------|-------------|----------|--------|----------|\n"

        for idx, item in enumerate(result.get("items", []), 1):
            q_short = item["question"][:40]
            content += (
                f"| {idx} | {q_short}... | {item['expected_context']} "
                f"| {item['faithfulness']:.2f} | {item['answer_relevance']:.2f} "
                f"| {item['context_recall']:.2f} | {item['context_precision']:.2f} |\n"
            )
        content += "\n"

    content += "---\n\n"

    # Worst performers
    content += "## Worst Performers\n\n"
    worst_items = []
    for config_name, result in comparison.items():
        for item in result.get("items", []):
            avg = (item["faithfulness"] + item["answer_relevance"]
                   + item["context_recall"] + item["context_precision"]) / 4
            worst_items.append((avg, config_name, item))

    worst_items.sort(key=lambda x: x[0])
    for avg, config_name, item in worst_items[:5]:
        content += f"- **[{config_name}]** {item['question'][:60]}... — avg score: {avg:.2f}\n"

    content += "\n---\n\n"

    # Recommendations
    content += "## Recommendations\n\n"
    content += "Dựa trên kết quả evaluation, đề xuất cải tiến:\n\n"
    content += "1. **Cải thiện chunk size**: Điều chỉnh CHUNK_SIZE để tối ưu context precision.\n"
    content += "2. **Tối ưu embedding model**: Dùng model chuyên biệt cho tiếng Việt pháp lý.\n"
    content += "3. **Cải thiện reranking**: Thử nghiệm cross-encoder thay vì heuristic.\n"
    content += "4. **Mở rộng golden dataset**: Thêm nhiều edge cases để đánh giá chính xác hơn.\n"
    content += "5. **Fine-tune BM25 parameters**: Điều chỉnh k1 và b cho văn bản pháp lý.\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n✓ Results exported to: {RESULTS_PATH}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    # Chạy A/B comparison
    comparison = compare_configs(golden_dataset)

    # Export results
    export_results(comparison)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for config_name, result in comparison.items():
        s = result["summary"]
        if s:
            print(f"\n{config_name}:")
            print(f"  Faithfulness:      {s['faithfulness']:.3f}")
            print(f"  Answer Relevance:  {s['answer_relevance']:.3f}")
            print(f"  Context Recall:    {s['context_recall']:.3f}")
            print(f"  Context Precision: {s['context_precision']:.3f}")
            print(f"  Items evaluated:   {result['num_success']}/{len(golden_dataset)}")
