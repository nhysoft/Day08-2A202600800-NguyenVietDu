"""
RAG Chatbot — Streamlit App.

Stack: Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display

Tính năng:
  - Trả lời có citation dựa trên task10
  - Follow-up questions (conversation memory)
  - Hiển thị source documents đã dùng
  - Tuỳ chọn bật/tắt reranking để so sánh

Chạy:
    streamlit run group_project/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Add src to path để import từ project
SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from task10_generation import generate_with_citation

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="RAG Chatbot — Pháp luật Ma tuý",
    page_icon="⚖️",
    layout="wide",
)

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_reranking" not in st.session_state:
    st.session_state.use_reranking = True

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("⚖️ RAG Chatbot")
    st.caption("Tra cứu pháp luật ma tuý & tin tức liên quan")

    st.divider()

    st.session_state.use_reranking = st.toggle(
        "Bật Reranking",
        value=st.session_state.use_reranking,
        help="Tắt để xem kết quả nếu không có rerank (phục vụ A/B test)",
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Xoá", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔁 Mới", use_container_width=True):
            st.rerun()

    st.divider()
    st.subheader("📚 Nguồn dữ liệu")
    st.markdown("""
- Luật Phòng chống ma tuý 2021
- Nghị định 57/2022/NĐ-CP
- Nghị định 105/2021/NĐ-CP
- Bộ luật Hình sự 2015
- Báo chí (VnExpress, Lao Động, VietnamNet, Tiền Phong)
    """)

    st.caption("Built with ❤️ using Streamlit + Gemini")

# =============================================================================
# HELPERS
# =============================================================================

def dedup_sources(sources: list[dict]) -> list[dict]:
    """Loại bỏ source trùng lặp dựa trên metadata source name."""
    seen = set()
    unique = []
    for s in sources:
        name = s.get("metadata", {}).get("source", "") or s.get("metadata", {}).get("file", "")
        if name and name not in seen:
            seen.add(name)
            unique.append(s)
        elif not name:
            unique.append(s)
    return unique


def get_top_k(sources: list[dict]) -> int:
    """Đếm số sources unique để hiển thị."""
    return len(dedup_sources(sources))


# =============================================================================
# CHAT HISTORY
# =============================================================================

# Hiển thị các tin nhắn cũ
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Xem {get_top_k(msg['sources'])} nguồn tham khảo", expanded=False):
                for src in dedup_sources(msg["sources"]):
                    meta = src.get("metadata", {})
                    source = meta.get("source", meta.get("file", "Unknown"))
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0.0)
                    content = src.get("content", "")

                    st.markdown(f"**📄 {source}** — `{doc_type}` — độ tương đồng: `{score:.3f}`")
                    st.text(content[:1000] + ("..." if len(content) > 1000 else ""))
                    st.divider()

# =============================================================================
# CHAT INPUT
# =============================================================================

if prompt := st.chat_input("Nhập câu hỏi về pháp luật ma tuý..."):
    # Lưu tin nhắn user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tra cứu tài liệu..."):
            try:
                result = generate_with_citation(
                    prompt,
                    use_reranking=st.session_state.use_reranking,
                )
                answer = result.get("answer", "")
                sources = result.get("sources", [])
            except Exception as e:
                answer = f"❌ **Lỗi xử lý:** {e}"
                sources = []

        # Hiển thị câu trả lời
        st.markdown(answer)

        # Hiển thị sources
        if sources:
            unique = dedup_sources(sources)
            st.divider()
            st.markdown(f"**📚 {len(unique)} nguồn tham khảo** — retrieval: `{result.get('retrieval_source', 'unknown')}`")

            for src in unique:
                meta = src.get("metadata", {})
                source = meta.get("source", meta.get("file", "Unknown"))
                doc_type = meta.get("type", "unknown")
                score = src.get("score", 0.0)
                content = src.get("content", "")

                with st.expander(f"📄 {source} — `{doc_type}` (độ tương đồng: `{score:.3f}`)"):
                    st.text(content)

        # Lưu vào session
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
