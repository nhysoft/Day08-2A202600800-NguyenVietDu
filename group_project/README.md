# Bài Tập Nhóm — RAG Chatbot về Pháp luật Ma tuý & Tin tức Nghệ sĩ

## Thông tin nhóm

| Thành viên | MSSV | Vai trò |
|-----------|------|---------|
| Phạm Triều Dương | 2A202600833 | Nhóm trưởng — Thiết kế kiến trúc, Pipeline chính |
| Nguyễn Viết Du | 2A202600800 | Thu thập dữ liệu pháp luật, Lexical Search |
| Nguyễn Thành Đạt | 2A202600626 | Crawl tin tức, Reranking |
| Nguyễn Ngọc Duy | 2A202600980 | Semantic Search, Chunking & Indexing |
| Nguyễn Võ Nguyên Huy | 2A202600672 | PageIndex Vectorless, Báo cáo nhóm |

---

## Lựa chọn sản phẩm: **RAG Chatbot** (Yêu cầu 1)

Nhóm chọn xây dựng **Chatbot tra cứu pháp luật ma tuý và tin tức liên quan** với giao diện Streamlit, trả lời có citation, hỗ trợ follow-up và hiển thị nguồn tham khảo.

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit App (app.py)                       │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────────────────────┐   │
│  │ Chat UI  │  │ Sidebar     │  │ Source Display               │   │
│  │ (message │  │ - Toggle    │  │ - Mở rộng/xem nguồn          │   │
│  │  history)│  │   rerank    │  │ - Metadata từng chunk         │   │
│  │          │  │ - Nguồn DL  │  │ - Score & loại               │   │
│  └────┬─────┘  └─────────────┘  └──────────────────────────────┘   │
└───────┼─────────────────────────────────────────────────────────────┘
        │ generate_with_citation(query, use_reranking)
┌───────▼─────────────────────────────────────────────────────────────┐
│                   Generation Layer (task10_generation.py)            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ Reorder docs │  │ Format       │  │ LLM Call (Gemini 2.5    │   │
│  │ (lost-in-the │→ │ context with │→ │ Flash / GPT-4o-mini)    │   │
│  │  middle fix) │  │ source label │  │ + citation generation   │   │
│  └──────────────┘  └──────────────┘  └─────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────────┘
                         │ retrieve(query, top_k, use_reranking)
┌────────────────────────▼────────────────────────────────────────────┐
│               Retrieval Pipeline (task9_retrieval_pipeline.py)       │
│                                                                     │
│  ┌────────────────┐      ┌──────────────────┐                       │
│  │ Semantic Search│      │ Lexical Search   │                       │
│  │ (task5)        │      │ (task6 — BM25)   │                       │
│  │  - Weaviate    │      │  - rank-bm25     │                       │
│  │  - Local pickle│      │  - LangChain     │                       │
│  │    fallback    │      │    splitter      │                       │
│  └───────┬────────┘      └────────┬─────────┘                       │
│          └──────────┬──────────────┘                                │
│                     ▼                                               │
│           ┌─────────────────┐                                       │
│           │  RRF Fusion     │                                       │
│           │  (Reciprocal    │                                       │
│           │   Rank Fusion)  │                                       │
│           └────────┬────────┘                                       │
│                    ▼                                                │
│           ┌─────────────────┐     ┌──────────────────────────────┐  │
│           │  Reranker       │──→  │ Cross-encoder (Jina/mock)   │  │
│           │  (task7)        │     │ hoặc MMR / RRF              │  │
│           └────────┬────────┘     └──────────────────────────────┘  │
│                    ▼          score < threshold?                    │
│           ┌─────────────────┐     ┌──────────────────────────────┐  │
│           │  Check          │────→│ PageIndex Vectorless         │  │
│           │  Threshold 0.3  │ YES │ (task8) — fallback BM25      │  │
│           └─────────────────┘     └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                         ▲
                         │ Embed & Store
┌────────────────────────┴────────────────────────────────────────────┐
│              Data Layer (task4_chunking_indexing.py)                 │
│                                                                     │
│  data/landing/  ──Task3──→  data/standardized/  ──Task4──→  Vector │
│  (PDF, JSON)      convert     (Markdown)           chunk+embed   Store│
│                    MarkItDown                      RecursiveChar   │
│                   (task3)                          TextSplitter    │
│                                                     size=500      │
│                                                     overlap=50    │
│                                                     Embedding:     │
│                                          gemini-embedding-001(768d)│
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Vector Store Options:                                        │   │
│  │  - Weaviate (docker/local) — preferred, hybrid search built-in│  │
│  │  - LocalVectorStore (pickle) — 100% local, no Docker needed   │  │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Công Nghệ Sử Dụng

| Layer | Công nghệ | Lý do chọn |
|-------|-----------|------------|
| UI | Streamlit | Đơn giản, nhanh, tích hợp Python tốt |
| LLM | Gemini 2.5 Flash (chính) / GPT-4o-mini (dự phòng) | Free tier Gemini, dùng làm judge luôn |
| Embedding | `gemini-embedding-001` (768 dim) | Multilingual, free, không cần self-host |
| Chunking | `RecursiveCharacterTextSplitter` (size=500, overlap=50) | An toàn, phổ biến, giữ ngữ cảnh điều luật |
| Semantic Search | Cosine similarity (local pickle) hoặc Weaviate | Weaviate hỗ trợ hybrid search built-in |
| Lexical Search | BM25 (`rank-bm25`, k1=1.5, b=0.75) | Chuẩn cho keyword matching |
| Reranking | Jina Reranker v2 (cross-encoder) / heuristic fallback | Cross-encoder multilingual, fallback word-overlap |
| Fusion | RRF (Reciprocal Rank Fusion, k=60) | Kết hợp semantic + lexical không cần tuning |
| Fallback | PageIndex / BM25 local | Khi hybrid search score thấp |

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Chi tiết |
|-----------|------|----------|----------|
| Phạm Triều Dương | 2A202600833 | Task 4 — Chunking & Indexing | Thiết kế chunking strategy (RecursiveCharacter, 500/50), tích hợp embedding (Gemini/OpenAI/local), Weaviate + local pickle fallback |
| | | Task 9 — Retrieval Pipeline | Tích hợp semantic + lexical → RRF fusion → rerank → threshold check → PageIndex fallback |
| | | Task 10 — Generation | Prompt engineering có citation, reorder chống lost-in-the-middle, Gemini + OpenAI provider |
| | | App — Streamlit Chatbot | Giao diện chat, toggle rerank, hiển thị sources, conversation memory, dedup sources |
| Nguyễn Viết Du | 2A202600800 | Task 1 — Thu thập văn bản pháp luật | Tải 3 văn bản: Luật Phòng chống ma tuý 2021, Nghị định 105/2021, Nghị định 28/2026 |
| | | Task 6 — Lexical Search (BM25) | Xây dựng BM25 index từ corpus, tokenizer tiếng Việt, tích hợp langchain-text-splitters |
| Nguyễn Thành Đạt | 2A202600626 | Task 2 — Crawl tin tức | Crawl 6 bài báo từ VnExpress, Lao Động, VietnamNet, Tiền Phong về nghệ sĩ và ma tuý |
| | | Task 7 — Reranking | Cross-encoder (Jina API), MMR, RRF implementations; fallback heuristic word-overlap |
| Nguyễn Ngọc Duy | 2A202600980 | Task 3 — Convert Markdown | Dùng MarkItDown để convert PDF → Markdown, crawl JSON → Markdown |
| | | Task 5 — Semantic Search | Query Weaviate hoặc local pickle bằng cosine similarity, hỗ trợ 3 embedding providers |
| Nguyễn Võ Nguyên Huy | 2A202600672 | Task 8 — PageIndex Vectorless | Fallback retrieval không cần vector store, BM25 fallback, PageIndex API integration |
| | | Báo cáo nhóm | Tổng hợp tài liệu, viết README, kiến trúc hệ thống |

---

## Hướng Dẫn Chạy

### Yêu cầu
- Python 3.10+
- API key: Gemini (bắt buộc), OpenAI (tuỳ chọn)

### Cài đặt

```bash
# Clone repo
git clone <repo-url>
cd day_08_rag_pipeline_v2

# Cài dependencies
pip install -r requirements.txt

# Cấu hình API key
cp .env.example .env
# Sửa .env: GEMINI_API_KEY=your_key_here
```

### Chạy indexing (nếu chưa có vector store)

```bash
python src/task4_chunking_indexing.py
```

### Chạy app

```bash
streamlit run group_project/app.py
```

### Sử dụng
1. Mở trình duyệt tại `http://localhost:8501`
2. Nhập câu hỏi về pháp luật ma tuý (vd: "Hình phạt cho tội tàng trữ trái phép chất ma tuý?")
3. Xem câu trả lời có citation + source documents
4. Bật/tắt **Reranking** ở sidebar để so sánh kết quả
5. Dùng **Follow-up questions** nhờ Streamlit session memory

---

## Kết Quả

- **3 văn bản pháp luật** gốc (PDF) + đã convert Markdown
- **6 bài báo** crawl từ các báo lớn + đã convert Markdown
- **RAG pipeline** hoàn chỉnh: semantic + lexical + rerank + fallback
- **Chatbot** Streamlit với citation, follow-up, source display
- **Citation** chính xác từng điều luật / tên báo

---

## Cấu Trúc Thư Mục

```
group_project/
├── README.md              ← Báo cáo nhóm (file này)
├── app.py                 ← Streamlit chatbot
├── evaluation/            ← Đánh giá (tham khảo)
│   ├── golden_dataset.json
│   ├── eval_pipeline.py
│   └── results.md
```

