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
    def from_dicts(cls, messages: List[Dict[str,str]]) ->List[Message]:
        msgs = []
        for msg in messages:
            msgs.append(Message(
                role=msg["role"],
                content=msg["content"]
            ))
        return msgs
    
    
def trans_messages_to_standard(
    messages: List['Message'], 
    tool_role_to_map: Literal["tool","assistant"]="tool"
) -> List[Dict[str,str]]:
    msg_list = []
    for msg in messages:
        msg_dict = msg.to_dict()
        if msg.role == "tool":
            msg_dict["role"]=tool_role_to_map
        msg_list.append(msg_dict)
    return msg_list
            

def trans_messages_to_text(messages: List[Message]) -> str:
    lines = []
    for m in messages:
        lines.append(f"{m.content}")
    text = "\n".join(lines)
    return text