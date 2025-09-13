# base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

class BaseTool(ABC):
    """Minimal tool base: implement run_one; run_batch loops by default."""

    name: str = "tool"
    max_rounds: int = 3

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})

    @abstractmethod
    def run_one(self, content: Any, *, context: Any = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """Run the tool once. Return (result, meta)."""
        raise NotImplementedError

    def run_batch(
        self,
        contents: List[Any],
        *,
        contexts: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """Default batch = simple loop over run_one."""
        if contexts is None:
            contexts = [None] * len(contents)
        if len(contexts) != len(contents):
            raise ValueError("Length of 'contexts' must match 'contents'.")

        results: List[Any] = []
        metas: List[Dict[str, Any]] = []
        for content, ctx in zip(contents, contexts):
            out, meta = self.run_one(content, context=ctx, **kwargs)
            results.append(out)
            metas.append(meta)
        return results, metas

    def __call__(self, content: Any, **kwargs: Any):
        if isinstance(content, list):
            return self.run_batch(content, **kwargs)
        return self.run_one(content, **kwargs)

@dataclass
class ToolCall:
    index: int
    name: str
    content: str
    meta: Dict = field(default_factory=dict)
    
@dataclass
class ToolResult:
    tool_name: str
    request_content: str
    output: Any
    meta: Dict 
    error: Optional[Any]
    index: int
    call: ToolCall
    
    

class ToolParser:
    def parse(self, text: str, meta: Dict = None) -> list[ToolCall]: ...
    def parse_batch(self, texts: list[str], metas: List[Dict]=None) -> list[list[ToolCall]]:
        if not metas:
            metas = [None]*len(texts)
        return [self.parse(t, meta) for t, meta in zip(texts,metas)]