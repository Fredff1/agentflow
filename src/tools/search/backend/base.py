from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import aiohttp


class SearchBackend:
    async def search(self, session: aiohttp.ClientSession, query: str, *, top_k: int) -> List[Dict[str, Any]]:
        """Return a list of hits: each hit is a dict with at least {title, url, snippet?}."""
        raise NotImplementedError

    async def fetch_details(
        self,
        session: aiohttp.ClientSession,
        hits: List[Dict[str, Any]],
        *,
        max_length: int,
        concurrency: int,
        proxy: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Optional: enrich hits with 'content'. Default: return hits."""
        return hits

