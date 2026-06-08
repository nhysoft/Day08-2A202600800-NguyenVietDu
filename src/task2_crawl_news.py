"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Dùng requests + BeautifulSoup thay vì crawl4ai (không cần cài thêm).
Output: data/landing/news/article_XX.json mỗi bài 1 file có trường `url`.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://tuoitre.vn/bat-nguoi-mau-an-tay-ca-si-chi-dan-co-tien-truc-phuong-do-lien-quan-ma-tuy-20241114114826655.htm",
    "https://thanhnien.vn/chi-dan-huu-tin-va-loat-sao-viet-gay-on-ao-vi-dinh-toi-ma-tuy-185241110141122628.htm",
    "https://thanhnien.vn/bat-ca-si-chi-dan-nguoi-mau-an-tay-nguyen-do-truc-phuong-lien-quan-ma-tuy-185241114120254879.htm",
    "https://tuoitre.vn/bo-cong-an-xuat-hien-mot-so-vu-can-bo-van-nghe-si-cau-thu-to-chuc-su-dung-ma-tuy-20241118150246384.htm",
    "https://ngoisao.vnexpress.net/nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-4816068.html",
    "https://thanhnien.vn/miu-le-va-loi-xin-loi-muon-mang-cua-loat-sao-viet-vuong-vao-ma-tuy-18526051513021689.htm",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def crawl_article(url: str) -> dict:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    title = ""
    if soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    elif soup.find("title"):
        title = soup.find("title").get_text(strip=True)

    # Lấy nội dung bài viết (thử các selector phổ biến)
    content = ""
    for selector in ["article", ".detail-content", ".article-body", ".content-detail", "main"]:
        tag = soup.select_one(selector)
        if tag:
            content = tag.get_text(separator="\n", strip=True)
            break
    if not content:
        content = soup.get_text(separator="\n", strip=True)[:5000]

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content": content,
    }


def crawl_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] {url}")
        try:
            article = crawl_article(url)
            filepath = DATA_DIR / f"article_{i:02d}.json"
            filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✓ Saved {filepath.name} ({len(article['content'])} chars)")
            success += 1
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")
        time.sleep(1)
    print(f"\nHoàn thành: {success}/{len(ARTICLE_URLS)} bài")


if __name__ == "__main__":
    crawl_all()
