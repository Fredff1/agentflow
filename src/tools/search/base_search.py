from __future__ import annotations
import asyncio
from typing import Optional, List, Dict, Any, Tuple
import aiohttp
from aiohttp import ClientTimeout

from .backend.base import SearchBackend
from ..base import BaseTool

def _run_coroutine_blocking(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # already inside an event loop → run in a dedicated thread loop
    import threading

    result_holder: Dict[str, Any] = {}
    exc_holder: Dict[str, BaseException] = {}

    def _target():
        try:
            result_holder["v"] = asyncio.run(coro)
        except BaseException as e:
            exc_holder["e"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()
    if "e" in exc_holder:
        raise exc_holder["e"]
    return result_holder.get("v")

class AsyncSearchTool(BaseTool):
    name = "search"
    description = "Async search tool with pluggable backends (blocking API)."

    def __init__(
        self,
        backend: SearchBackend,
        *,
        timeout: int = 60,
        top_k: int = 3,
        concurrent_limit: int = 2,
        trust_env: bool = False,
        proxy: Optional[str] = None,
        fetch_details: bool = True,
        detail_concurrency: int = 3,
        max_length: int = 10000,
        enable_summarize: bool = False,
        summarize_engine: Any = None,  
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config=config)
        self.backend = backend
        self.timeout = timeout
        self.top_k = top_k
        self.concurrent_limit = concurrent_limit
        self.trust_env = trust_env
        self.proxy = proxy
        self.fetch_details = fetch_details
        self.detail_concurrency = detail_concurrency
        self.max_length = max_length
        self.enable_summarize = enable_summarize
        self.summarize_engine = summarize_engine

    # --- public sync API (single) ---
    def run_one(self, content: Any, *, context: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        query = str(content)
        results = _run_coroutine_blocking(self._run_batch_async([query]))
        text = results[0]["formatted"]
        meta = results[0]["meta"]
        # optional summarization
        if self.enable_summarize and self.summarize_engine:
            summary, summarize_meta = self.summarize_engine.summarize(text, query)
            meta.update(summarize_meta or {})
            return summary, meta
        return text, meta

    # --- public sync API (batch) ---
    def run_batch(self, contents: List[Any], *, contexts: Optional[List[Any]] = None, **kwargs: Any):
        queries = [str(x) for x in contents]
        results = _run_coroutine_blocking(self._run_batch_async(queries))
        outs = [r["formatted"] for r in results]
        metas = [r["meta"] for r in results]

        if self.enable_summarize and self.summarize_engine:
            qs = queries
            raw_texts = outs
            summaries, summarize_metas = self.summarize_engine.batch_summarize(raw_texts, qs, user_prompts=None)
            outs = summaries
            for m, sm in zip(metas, summarize_metas):
                m.update(sm or {})
        return outs, metas

    async def _run_batch_async(self, queries: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        timeout = ClientTimeout(total=self.timeout)

        connector = aiohttp.TCPConnector(limit_per_host=self.concurrent_limit, ssl=False)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=self.trust_env, connector=connector) as session:
            # 1) search in parallel
            hits_lists = await asyncio.gather(
                *[self.backend.search(session, q, top_k=self.top_k) for q in queries],
                return_exceptions=True,
            )
            # 2) details (optional)
            if self.fetch_details:
                lengths: List[int] = []
                flattened: List[Dict[str, Any]] = []
                for hits in hits_lists:
                    if isinstance(hits, Exception) or not hits:
                        lengths.append(0)
                    else:
                        lengths.append(len(hits))
                        flattened.extend(hits)

                if flattened:
                    try:
                        enriched_flat = await self.backend.fetch_details(
                            session,
                            flattened,
                            max_length=self.max_length,
                            concurrency=self.detail_concurrency,  # 全局并发上限
                            proxy=self.proxy,
                        )
                        # 回切回每个 query
                        enriched_lists: List[List[Dict[str, Any]]] = []
                        offset = 0
                        for L in lengths:
                            if L == 0:
                                enriched_lists.append([])
                            else:
                                enriched_lists.append(enriched_flat[offset: offset + L])
                                offset += L
                        hits_lists = enriched_lists
                    except Exception:
                        # 兜底：若后端在大批量一次性抓详情时失败，退回“每个 query 并发、跨 query 串行”的旧策略
                        enriched_lists: List[List[Dict[str, Any]]] = []
                        for hits in hits_lists:
                            if isinstance(hits, Exception) or not hits:
                                enriched_lists.append([])
                            else:
                                enriched = await self.backend.fetch_details(
                                    session,
                                    hits,
                                    max_length=self.max_length,
                                    concurrency=self.detail_concurrency,
                                    proxy=self.proxy,
                                )
                                enriched_lists.append(enriched)
                        hits_lists = enriched_lists

            # 3) format each result
            for q, hits in zip(queries, hits_lists):
                if isinstance(hits, Exception) or not hits:
                    out.append({"formatted": "Search Failed", "meta": {"raw_hits": [], "query": q}})
                    continue
                formatted = self._format_hits(hits, q)
                out.append({"formatted": formatted, "meta": {"raw_hits": hits, "query": q}})
        return out

    def _format_hits(self, hits: List[Dict[str, Any]], query: str) -> str:
        if not hits:
            return "Search Failed"
        parts: List[str] = []
        share = max(1, self.max_length // max(1, len(hits)))
        for idx, h in enumerate(hits):
            title = (h.get("title") or "").strip() or "(no title)"
            url = h.get("url") or ""
            body = (h.get("content") or h.get("snippet") or "").replace("\n", " ")
            parts.append(f"[{idx}] Title: {title} Source: {url}\n{body[:share]}\n")
        text = "\n".join(parts)[: self.max_length]
        return text