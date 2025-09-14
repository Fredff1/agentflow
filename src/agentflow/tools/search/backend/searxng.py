from __future__ import annotations
import asyncio
from typing import Optional, List, Dict, Any, Tuple
import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup

from .base import SearchBackend
# 可选日志：from utils.log_util import get_logger
# -------- Searxng Backend --------
class SearxngBackend(SearchBackend):
    def __init__(self, base_url: str, proxy: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.proxy = proxy

    async def search(self, session: aiohttp.ClientSession, query: str, *, top_k: int) -> List[Dict[str, Any]]:
        params = {"q": query, "format": "json"}
        kwargs: Dict[str, Any] = {"params": params}
        if self.proxy:
            kwargs["proxy"] = self.proxy
        try:
            async with session.get(f"{self.base_url}/search", **kwargs) as resp:
                resp.raise_for_status()
                js = await resp.json()
                hits = (js.get("results") or [])[:top_k]
                # standardize fields
                std_hits: List[Dict[str, Any]] = []
                for h in hits:
                    std_hits.append(
                        {
                            "title": (h.get("title") or "").strip() or "(no title)",
                            "url": h.get("url") or "",
                            "snippet": (h.get("content") or h.get("snippet") or "") or "",
                        }
                    )
                return std_hits
        except Exception:
            return []

    async def fetch_details(
        self,
        session: aiohttp.ClientSession,
        hits: List[Dict[str, Any]],
        *,
        max_length: int,
        concurrency: int,
        proxy: Optional[str],
    ) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        per_req_timeout = ClientTimeout(total=7, sock_connect=5, sock_read=5)

        async def _grab(url: str) -> str:
            if not url:
                return ""
            try:
                async with sem:
                    kwargs: Dict[str, Any] = {"headers": headers, "timeout": per_req_timeout}
                    if proxy and proxy.startswith(("http://", "https://")):
                        kwargs["proxy"] = proxy
                    async with session.get(url, **kwargs) as r:
                        r.raise_for_status()
                        if "text/html" not in r.headers.get("Content-Type", ""):
                            return ""
                        chunk = await r.content.read()
                soup = BeautifulSoup(chunk, "lxml")
                body = soup.body or soup
                return body.get_text("\n", strip=True)[:max_length]
            except Exception:
                return ""

        tasks = [asyncio.create_task(_grab(hit.get("url", ""))) for hit in hits]
        done, pending = await asyncio.wait(tasks, timeout=30)
        for t in pending:
            t.cancel()

        out: List[Dict[str, Any]] = []
        for hit, task in zip(hits, tasks):
            text = ""
            if task in done:
                try:
                    text = task.result() or ""
                except Exception:
                    text = ""
            out.append({**hit, "content": text})
        return out