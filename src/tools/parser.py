from typing import Any, Dict, List, Optional, Tuple

from .base import ToolParser, ToolCall
from ..utils.tag_util import find_tags

class TagToolParser(ToolParser):
    
    def __init__(self, tool_tags: List[str] = ["search","python"]):
        super().__init__()
        self.tool_tags = tool_tags
        

    def parse(self, text: str, meta: Dict=None) -> List[ToolCall]:
        if not meta:
            meta = {}
        matches = find_tags(text,self.tool_tags)
        calls: List[ToolCall] = []
        for idx, match in enumerate(matches):
            call = ToolCall(
                index=idx,
                name = match.tag,
                content=match.body,
                meta=meta
            )
            calls.append(call)
        return calls

