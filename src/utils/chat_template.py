from __future__ import annotations
from typing import Any, List, Dict, Optional, Tuple, Union

MessageDict = Dict[str, Any] 

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

def is_chat_messages(obj: Any) -> bool:
    if not isinstance(obj, list):
        return False
    if len(obj) == 0:
        return True
    head = obj[0]
    if isinstance(head,list) and len(head)>0:
        head = head[0]
    if not isinstance(head, dict):
        return False
    return "role" in head and "content" in head

def default_trans_messages(messages: Union[List[MessageDict],List[List[MessageDict]]]) -> str:
    if len(messages)==0:
        return ""
    def _process_one(messages:List[MessageDict]):
        parts: List[str] = []
        for m in messages:
            name = f" ({m.get('name')})" if m.get("name") else ""
            parts.append(f"[{m.get('role')}{name}] {str(m.get('content',''))}")
        return "\n".join(parts)
    first = messages[0]
    if isinstance(first,dict):
        return _process_one(messages)
    else:
        results = [_process_one(msg) for msg in messages]
        return results

def safe_apply_chat_template(
    tokenizer: Any,
    messages: List[Any],
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    **kwargs
) -> Tuple[str, bool]:
    has_method = hasattr(tokenizer, "apply_chat_template")
    if has_method:
        try:
            text = tokenizer.apply_chat_template(
                messages=messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            return text, True
        except Exception as e:  # 模板存在但使用失败（模板变量缺失等）
            return default_trans_messages(messages), False, f"{type(e).__name__}: {e}"
    # 无模板：直接回退
    return default_trans_messages(messages), False