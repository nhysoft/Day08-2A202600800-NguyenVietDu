"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Tải 3 văn bản pháp luật PDF từ cổng chính phủ về data/landing/legal/.
"""

import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCS = [
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/01/luat120-2025.pdf",
        "filename": "luat-phong-chong-ma-tuy-2025.pdf",
    },
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/105.signed_02.pdf",
        "filename": "nghi-dinh-105-2021-phong-chong-ma-tuy.pdf",
    },
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/9/135-vbhn-vpqh.pdf",
        "filename": "bo-luat-hinh-su-hop-nhat-2017.pdf",
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, filename: str):
    filepath = DATA_DIR / filename
    if filepath.exists():
        print(f"  (đã có) {filename}")
        return
    print(f"  Đang tải {filename} ...")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    filepath.write_bytes(resp.content)
    print(f"  ✓ Saved {filename} ({filepath.stat().st_size // 1024} KB)")


def collect_legal_docs():
    setup_directory()
    for doc in LEGAL_DOCS:
        download_file(doc["url"], doc["filename"])


if __name__ == "__main__":
    collect_legal_docs()
