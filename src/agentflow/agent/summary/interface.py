from __future__ import annotations
from typing import Protocol, List, Tuple, Dict, Any, Optional, Union


from agentflow.common.messages import Message


SummaryItem = Union[str, List[Message]]

class SummarizerInterface(Protocol):

    def summarize(self, item: SummaryItem, **kwargs) -> Tuple[str, Dict[str, Any]]:
        ...

    def summarize_batch(
        self,
        items: List[SummaryItem],
        **kwargs
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        ...