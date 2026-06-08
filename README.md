# RAG Pipeline v2 — Ngày 8

> **Họ tên:** Nguyễn Viết Du  
> **MSSV:** 2A202600800  
> **Chương trình:** Chương 2 — Ngày 8/15  
> **Chủ đề:** Pháp luật Việt Nam về ma tuý + Bài báo nghệ sĩ liên quan

---

## Tổng Quan

Xây dựng RAG pipeline end-to-end hoàn chỉnh: từ thu thập dữ liệu pháp luật và báo chí → xử lý → indexing → retrieval hybrid → generation có citation.

---

## Kết Quả Kiểm Thử

```
Ran 35 tests in ~1s

OK (skipped=1)
```

| Task | Nội dung | Điểm | Kết quả |
|------|----------|:----:|:-------:|
| Task 1 | Thu thập văn bản pháp luật | 3 | ✅ PASS |
| Task 2 | Crawl bài báo | 3 | ✅ PASS |
| Task 3 | Convert sang Markdown | 4 | ✅ PASS |
| Task 4 | Chunking & Indexing | 7 | ✅ PASS |
| Task 5 | Semantic Search | 6 | ✅ PASS |
| Task 6 | Lexical Search (BM25) | 6 | ✅ PASS |
| Task 7 | Reranking | 6 | ✅ PASS |
| Task 8 | PageIndex Vectorless | 4 | ✅ PASS |
| Task 9 | Retrieval Pipeline | 7 | ✅ PASS |
| Task 10 | Generation có Citation | 4 | ✅ PASS* |
| **Tổng** | | **50** | **34/35 PASS** |

> \* `generate_with_citation` skip khi không có `ANTHROPIC_API_KEY` — cần set `.env` để chạy đầy đủ.

---

## Cấu Trúc Thư Mục

```
Day08-2A202600800-NguyenVietDu/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── landing/
│   │   ├── legal/          ← Task 1: file PDF/DOCX pháp luật
│   │   └── news/           ← Task 2: file JSON bài báo crawl
│   └── standardized/
│       ├── legal/          ← Task 3: markdown từ PDF/DOCX
│       └── news/           ← Task 3: markdown từ HTML/JSON
├── src/
│   ├── __init__.py
│   ├── task1_collect_legal_docs.py
│   ├── task2_crawl_news.py
│   ├── task3_convert_markdown.py
│   ├── task4_chunking_indexing.py
│   ├── task5_semantic_search.py
│   ├── task6_lexical_search.py
│   ├── task7_reranking.py
│   ├── task8_pageindex_vectorless.py
│   ├── task9_retrieval_pipeline.py
│   └── task10_generation.py
└── tests/
    └── test_individual.py
```

---

## Chi Tiết Từng Task

### Task 1 — Thu Thập Văn Bản Pháp Luật

**Nguồn dữ liệu:** Tải trực tiếp các văn bản pháp luật về ma tuý từ cổng thông tin chính phủ.

**Dữ liệu thu thập được:**
- Luật Phòng, chống ma tuý 2021 (Luật số 73/2021/QH15)
- Bộ luật Hình sự — Chương XX: Các tội phạm về ma tuý
- Nghị định, thông tư hướng dẫn liên quan

**Lưu tại:** `data/landing/legal/` (≥3 file PDF/DOCX, mỗi file >1KB)

---

### Task 2 — Crawl Bài Báo

**Công cụ:** Crawl4AI / requests + BeautifulSoup

**Dữ liệu thu thập được:** ≥5 bài báo về nghệ sĩ Việt Nam liên quan tới ma tuý (Chi Dân, Miu Lê, Nikolai Đinh, ...)

**Mỗi file JSON có metadata:**
```json
{
  "url": "...",
  "title": "...",
  "content": "...",
  "crawled_at": "..."
}
```

**Lưu tại:** `data/landing/news/`

---

### Task 3 — Convert Sang Markdown

**Công cụ:** [MarkItDown](https://github.com/microsoft/markitdown) (Microsoft)

**Quá trình:**
- PDF/DOCX trong `data/landing/legal/` → `.md` trong `data/standardized/legal/`
- HTML/JSON trong `data/landing/news/` → `.md` trong `data/standardized/news/`

**Cách chạy:**
```bash
python src/task3_convert_markdown.py
```

---

### Task 4 — Chunking & Indexing

**Chiến lược đã chọn:**

| Thành phần | Lựa chọn | Lý do |
|------------|----------|-------|
| Chunker | `RecursiveCharacterTextSplitter` | An toàn, hoạt động tốt với cả văn bản pháp luật dài và bài báo ngắn |
| `CHUNK_SIZE` | 500 ký tự | Đủ ngữ cảnh cho 1 điều luật / 1 đoạn báo, không quá dài |
| `CHUNK_OVERLAP` | 50 ký tự | Giữ ngữ cảnh kết nối giữa các chunk liền kề |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual, nhẹ (420MB), hỗ trợ tiếng Việt tốt |
| Vector Store | ChromaDB | Local, không cần Docker, hỗ trợ metadata filtering |

**Cách chạy pipeline đầy đủ:**
```bash
python src/task4_chunking_indexing.py
```

---

### Task 5 — Semantic Search

**Phương pháp:** TF-IDF Cosine Similarity (chạy hoàn toàn trên RAM, không cần vector store)

**Cơ chế:**
1. Tokenize query và corpus bằng regex `\w+`
2. Tính TF-IDF vector cho từng document và query
3. Tính cosine similarity: `dot(q, d) / (|q| × |d|)`
4. Trả về top-k kết quả sorted descending

**Interface:**
```python
results = semantic_search("hình phạt tàng trữ ma tuý", top_k=5)
# [{'content': ..., 'score': 0.87, 'metadata': {...}}, ...]
```

---

### Task 6 — Lexical Search (BM25)

**Phương pháp:** BM25 tự implement bằng stdlib Python (không dùng thư viện ngoài)

**Công thức:**
```
score(q, d) = Σ IDF(qi) × tf(qi,d)×(k1+1) / (tf(qi,d) + k1×(1−b+b×|d|/avgdl))
```

**Tham số:** `k1=1.5` (term saturation), `b=0.75` (length normalization)

**Interface:**
```python
results = lexical_search("Điều 248 tàng trữ trái phép", top_k=5)
# [{'content': ..., 'score': 4.23, 'metadata': {...}}, ...]
```

---

### Task 7 — Reranking

**3 phương pháp đã implement:**

| Hàm | Mô tả |
|-----|-------|
| `rerank_cross_encoder()` | Keyword overlap + original score (pure Python, không cần model ngoài) |
| `rerank_mmr()` | Maximal Marginal Relevance — cân bằng relevance và diversity |
| `rerank_rrf()` | Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker |

**Công thức Cross-encoder (keyword-based):**
```
new_score = original_score × 0.6 + keyword_overlap_ratio × 0.4
```

**Interface:**
```python
reranked = rerank("hình phạt ma tuý", candidates, top_k=3)
```

---

### Task 8 — PageIndex Vectorless

**Phương pháp:** Keyword frequency scoring trực tiếp trên file markdown (không cần vector, không cần API ngoài)

**Cơ chế:**
- Đọc toàn bộ `.md` trong `data/standardized/`
- Tính `score = Σ tf(token) / len(doc)` cho mỗi query token
- Kết quả được đánh dấu `source: 'pageindex'`

**Interface:**
```python
results = pageindex_search("luật phòng chống ma tuý", top_k=3)
# [{'content': ..., 'score': ..., 'source': 'pageindex'}, ...]
```

---

### Task 9 — Retrieval Pipeline Hoàn Chỉnh

**Kiến trúc pipeline:**

```
Query
  ├──→ Semantic Search (TF-IDF)  ──┐
  │                                 ├──→ RRF Merge ──→ Rerank ──→ Results
  ├──→ Lexical Search (BM25)     ──┘        │
  │                                          │ score < threshold
  └──→ Fallback: PageIndex  ←───────────────┘
```

**Logic fallback:**
- Chạy semantic + lexical song song
- Merge bằng **Reciprocal Rank Fusion (RRF)**
- Rerank kết quả merged
- Nếu `best_score < 0.3` → fallback sang PageIndex

**Interface:**
```python
results = retrieve("hình phạt tàng trữ ma tuý", top_k=3)
# [{'content': ..., 'score': ..., 'source': 'hybrid' | 'pageindex'}, ...]
```

---

### Task 10 — Generation Có Citation

**Cấu hình LLM:**

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| Model | `claude-haiku-4-5` | Nhanh, đủ mạnh cho Q&A có context |
| `temperature` | 0.3 | RAG cần factual, giảm sáng tạo |
| `top_p` | 0.9 | Đủ diverse nhưng không quá random |
| `top_k` chunks | 5 | Đủ evidence, tránh lost in the middle |

**Document Reordering — tránh Lost in the Middle:**
```
Input  (by score desc): [C0, C1, C2, C3, C4]
Output (LLM-optimized): [C0, C2, C4, C3, C1]
                         ──────────  ────────
                         đầu (nhớ)   cuối (nhớ)
```

> LLM nhớ tốt thông tin ở đầu và cuối prompt, quên thông tin ở giữa.  
> Chunks quan trọng nhất đặt ở hai đầu, ít quan trọng hơn ở giữa.

**Pipeline:**
```
retrieve() → reorder_for_llm() → format_context() → LLM → answer có [citation]
```

**Interface:**
```python
result = generate_with_citation("Hình phạt tội tàng trữ ma tuý?")
# {
#   'answer': '...câu trả lời có [Luật PCMT 2021, Điều 248]...',
#   'sources': [...],
#   'retrieval_source': 'hybrid'
# }
```

---

## Cài Đặt & Chạy

### 1. Clone và cài dependencies

```bash
git clone <repo>
cd Day08-2A202600800-NguyenVietDu
pip install -r requirements.txt
```

### 2. Cấu hình môi trường

```bash
cp .env.example .env
```

Điền vào file `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Chạy toàn bộ test

```bash
python3 tests/test_individual.py
```

### 4. Chạy từng task riêng lẻ

```bash
python3 src/task4_chunking_indexing.py   # Build vector index
python3 src/task5_semantic_search.py     # Test semantic search
python3 src/task6_lexical_search.py      # Test BM25 search
python3 src/task9_retrieval_pipeline.py  # Test full pipeline
python3 src/task10_generation.py         # Test generation
```

---

## Công Nghệ Sử Dụng

| Thành phần | Công nghệ |
|-----------|-----------|
| Web crawling | Crawl4AI |
| Document conversion | MarkItDown (Microsoft) |
| Chunking | langchain-text-splitters |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector store | ChromaDB |
| BM25 | Tự implement (stdlib Python) |
| TF-IDF | Tự implement (stdlib Python) |
| Reranking | RRF + keyword overlap (pure Python) |
| Generation | Anthropic Claude Haiku |

