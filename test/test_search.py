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
from src.tools.base import ToolCallRequest


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
    result = tool.run_one(ToolCallRequest(
        index=0,
        name="search",
        content="python asyncio tutorial"
    ))
    print(result)

    

if __name__ == "__main__":
    print("Starting test")
    test_search_real_searxng_basic_no_summary()
    print("END")

