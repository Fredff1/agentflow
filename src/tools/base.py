# base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class ToolCallRequest:
    index: int
    name: str
    content: str
    meta: Dict = field(default_factory=dict)
    
@dataclass
class ToolCallResult:
    tool_name: str
    request_content: str
    output: Any
    meta: Dict 
    error: Optional[Any]
    index: int
    call: ToolCallRequest
    
    

class ToolParser:
    def parse(self, text: str, meta: Dict = None) -> list[ToolCallRequest]: ...
    def parse_batch(self, texts: list[str], metas: List[Dict]=None) -> list[list[ToolCallRequest]]:
        if not metas:
            metas = [None]*len(texts)
        return [self.parse(t, meta) for t, meta in zip(texts,metas)]

class BaseTool(ABC):
    """Minimal tool base: implement run_one; run_batch loops by default."""

    name: str = "tool"
    max_rounds: int = 3

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})

    @abstractmethod
    def run_one(self, call: ToolCallRequest, **kwargs: Any) -> ToolCallResult:
        """Run the tool once. Return (result, meta)."""
        raise NotImplementedError

    def run_batch(
        self,
        calls: List[ToolCallRequest],
        **kwargs: Any,
    ) -> List[ToolCallResult]:
        """Default batch = simple loop over run_one."""
        results: List[ToolCallResult] = []
        for call in calls:
            result = self.run_one(call, **kwargs)
            results.append(result)
        return results


