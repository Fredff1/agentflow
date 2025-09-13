from typing import List, Dict, Optional

from.base import BaseTool

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}


    def register(self, name: str, tool: BaseTool) -> None:
        self._tools[name] = tool


    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)


    def __contains__(self, name: str) -> bool:
        return name in self._tools


    def available(self) -> List[str]:
        return sorted(self._tools.keys())