from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

from .registry import ToolRegistry
from .base import ToolCall, ToolParser, ToolResult

class ToolCaller:
    
    def __init__(self, registry: ToolRegistry, parser: ToolParser):
        self.registry = registry
        self.parser = parser
        
    def call_batch(self, texts: List[str], metas: List[Dict], **kwargs) -> List[List[ToolResult]]:
        calls_per_text = self.parser.parse_batch(texts,metas)
        grouped: DefaultDict[str, List[Tuple[int,ToolCall]]] = defaultdict(list)
        for tid, calls in enumerate(calls_per_text):
            for c in calls:
                grouped[c.name].append((tid, c))
                
        result_grid: List[Dict[int, ToolResult]] = [dict() for _ in range(len(texts))]
                
        for tool_name, items in grouped.items():
            tool = self.registry.get(tool_name)
            if tool is None:
                raise RuntimeError(f"Tool with tag {tool_name} does not exist")
            indices = [text_idx for (text_idx, _) in items]
            contents = [call.content for (_, call) in items]
            contexts = [call.meta for (_, call) in items] 
            outputs, metas = tool.run_batch(contents, contexts=contexts)
            
        for (text_idx, call), out, meta in zip(items, outputs, metas):
            tool_result = ToolResult(
                tool_name=tool_name,
                request_content=call.content,
                output=out,
                meta=meta,
                error = None,
                index = call.index,
                call=call,
            )
            result_grid[text_idx][call.index] = tool_result
        
        final_results: List[List[ToolResult]] = []
        for tid, calls in enumerate(calls_per_text):
            if not calls:
                final_results.append([])  
                continue
            by_idx = result_grid[tid]
            final_results.append(list(by_idx.values()))
        return final_results
    
    def call_single(self, text: str, meta: Dict=None) -> List[ToolResult]:
        return self.call_batch([text],[meta])[0]