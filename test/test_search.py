# tests/test_async_search_tool.py
import sys
import os
import urllib.request
import json
import pytest

# 避免和 vLLM 等库的多进程方式冲突
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# 让测试在任意位置都能 import 到你的源码
ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT_DIR)

from src.tools.search.backend.base import SearchBackend
from src.tools.search.backend.searxng import SearxngBackend
from src.tools.search.base_search import AsyncSearchTool


# ------------------------------
# 轻量 Mock 后端（少量用例）
# ------------------------------
class FakeBackend(SearchBackend):
    async def search(self, session, query: str, *, top_k: int):
        return [
            {
                "title": f"Title {i} for {query}",
                "url": f"https://example.com/{query}/{i}",
                "snippet": f"Snippet {i} about {query}",
            }
            for i in range(top_k)
        ]

    async def fetch_details(self, session, hits, *, max_length: int, concurrency: int, proxy: str | None):
        # 模拟详情抓取：给每个 hit 填充 content
        enriched = []
        for h in hits:
            enriched.append({**h, "content": f"Content fetched for {h.get('url')}"[:max_length]})
        return enriched


def test_search_run_one_with_details_fake_backend():
    print("Mocl single search")
    tool = AsyncSearchTool(
        backend=FakeBackend(),
        top_k=2,
        fetch_details=True,
        detail_concurrency=2,
        max_length=200,
    )
    text, meta = tool.run_one("unit-test")
    assert "[0] Title: Title 0 for unit-test" in text
    assert "https://example.com/unit-test/0" in text
    assert "Content fetched for https://example.com/unit-test/0" in text
    assert meta["query"] == "unit-test"
    assert len(meta["raw_hits"]) == 2


def test_search_run_batch_multiple_queries_fake_backend():
    print("Mock batch search")
    tool = AsyncSearchTool(
        backend=FakeBackend(),
        top_k=3,
        fetch_details=True,
        detail_concurrency=3,
        max_length=300,
    )
    texts, metas = tool.run_batch(["alpha", "beta"])
    assert len(texts) == len(metas) == 2
    assert "Title 0 for alpha" in texts[0]
    assert "Title 0 for beta" in texts[1]
    assert len(metas[0]["raw_hits"]) == 3
    assert len(metas[1]["raw_hits"]) == 3


# ------------------------------
# 真实本地 SearxNG 调用（无 summarize）
# ------------------------------
SEARXNG_BASE = "http://127.0.0.1:8888"



def test_search_real_searxng_basic_no_summary():
    backend = SearxngBackend(base_url=SEARXNG_BASE, proxy=None)
    tool = AsyncSearchTool(
        backend=backend,
        top_k=4,
        fetch_details=True,      # 不抓详情，避免访问外部网页；只用搜索返回的 title/snippet
        detail_concurrency=32,
        max_length=6000,
        enable_summarize=False,   # 明确不启用 summarize
        timeout=15,
    )
    text, meta = tool.run_one("python asyncio tutorial")
    print(text,meta)
    # 基本断言：不应失败，且格式包含 Title 字样
    assert text != "Search Failed"
    assert "[0] Title:" in text
    assert meta["query"] == "python asyncio tutorial"
    assert isinstance(meta.get("raw_hits"), list)
    
    texts, metas = tool.run_batch(["python learning"]*7)
    print(texts)
    
if __name__ == "__main__":
    print("Starting test")
    test_search_run_one_with_details_fake_backend()
    test_search_run_batch_multiple_queries_fake_backend()
    print("test real search")
    test_search_real_searxng_basic_no_summary()
    print("END")

