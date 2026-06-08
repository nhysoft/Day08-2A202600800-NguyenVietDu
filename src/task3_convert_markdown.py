"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Dùng MarkItDown (Microsoft) cho PDF/DOCX.
JSON bài báo thì extract content trực tiếp.
Output lưu vào data/standardized/ giữ nguyên cấu trúc thư mục.
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            if not result.text_content or len(result.text_content) < 200:
                print(f"  ⚠ Skip: {filepath.name} — PDF không có text layer (có thể là scan)")
                continue
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(result.text_content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path.name} ({len(result.text_content)} chars)")


def convert_news_articles():
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            content = header + data.get("content", "")

            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path.name} ({len(content)} chars)")


def convert_all():
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print(f"\n✓ Done! Output tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
