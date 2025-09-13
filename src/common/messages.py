from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, List

MessageRole = Literal["system", "user", "assistant", "tool"]

@dataclass
class Message:
    role: MessageRole
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {"role":self.role,"content":self.content}
    
    @classmethod
    def to_chat_messages(cls, messages: List['Message']):
        msg_list = []
        for msg in messages:
            msg_list.append(msg.to_dict())
        return msg_list
            

def trans_messages_to_text(messages: List[Message], max_chars: Optional[int] = None) -> str:
    lines = []
    for m in messages:
        name = f" ({m.name})" if m.name else ""
        lines.append(f"[{m.role}{name}] {m.content}")
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + "\n...\n" + text[-half:]
    return text