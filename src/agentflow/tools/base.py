# base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

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
    
    def make_result_str(self, result: ToolCallResult) -> str:...

class BaseTool(ABC):
    """Minimal tool base: implement run_one; run_batch loops by default with quota check."""

    name: str = "tool"

    def __init__(self, config: Optional[Dict[str, Any]] = None, max_rounds: int = 3) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self.max_rounds: int = int(max_rounds)  # 每个样本/轮的最大调用次数阈值（基于 meta.round_counter）

    # ---- 子类必须实现 ----
    @abstractmethod
    def run_one(self, call: ToolCallRequest, **kwargs: Any) -> ToolCallResult:
        """Run the tool once. Return ToolCallResult."""
        raise NotImplementedError

    # ---- 对外批处理（默认：for 循环 + 配额检查 + 兜底错误）----
    def run_batch(self, calls: List[ToolCallRequest], **kwargs: Any) -> List[ToolCallResult]:
        def _runner(allowed_calls: List[ToolCallRequest]) -> List[ToolCallResult]:
            out: List[ToolCallResult] = []
            for c in allowed_calls:
                try:
                    r = self.run_one(c, **kwargs)
                except Exception as e:
                    r = self._make_error_result(c, f"{type(e).__name__}: {e}")
                out.append(r)
            return out
        return self._apply_round_quota(calls, _runner)

    # ================== 通用工具：配额裁剪与结果构造 ==================

    def _apply_round_quota(
        self,
        calls: List[ToolCallRequest],
        runner: Callable[[List[ToolCallRequest]], List[ToolCallResult]],
    ) -> List[ToolCallResult]:
        if self.max_rounds and self.max_rounds > 0:
            pass
        # 分流
        allowed: List[ToolCallRequest] = []
        allowed_positions: List[int] = []
        results: List[Optional[ToolCallResult]] = [None] * len(calls)

        for i, call in enumerate(calls):
            if self._is_quota_exceeded(call):
                results[i] = self._make_exceeded_result(call)
            else:
                allowed.append(call)
                allowed_positions.append(i)

        # 实际执行
        if allowed:
            produced = runner(allowed)
            for pos, r in zip(allowed_positions, produced):
                results[pos] = r

        # 填充兜底（理论上不应出现 None）
        final: List[ToolCallResult] = []
        for i, r in enumerate(results):
            if r is None:
                final.append(self._make_error_result(calls[i], "InternalError: missing result"))
            else:
                final.append(r)
        return final

    def _is_quota_exceeded(self, call: ToolCallRequest) -> bool:
        """从 call.meta.round_counter 中读取该工具的使用次数，与 self.max_rounds 比较。"""
        if self.max_rounds <= 0:
            return False
        meta = getattr(call, "meta", None) or {}
        rc = meta.get("round_counter") or {}
        used = rc.get(self.name)
        try:
            if used is None:
                return False
            return int(used) >= self.max_rounds
        except Exception:
            return False

    def _make_exceeded_result(self, call: ToolCallRequest) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            request_content=getattr(call, "content", None),
            output="Tool Quota Exceed Failure",
            meta={**(getattr(call, "meta", {}) or {}), "skipped": True, "reason": f"per-sample max_rounds={self.max_rounds}"},
            error=f"skipped: per-sample limit {self.max_rounds}",
            index=getattr(call, "index", -1),
            call=call,
        )

    def _make_error_result(self, call: ToolCallRequest, error_msg: str) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            request_content=getattr(call, "content", None),
            output=None,
            meta={**(getattr(call, "meta", {}) or {}), "exception": True},
            error=error_msg,
            index=getattr(call, "index", -1),
            call=call,
        )


