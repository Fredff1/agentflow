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

def default_trans_messages(messages: Union[List[MessageDict],List[List[MessageDict]]]) -> Union[str,List[str]]:
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
) -> Tuple[Union[str,List[str]]:, bool]:
    has_method = hasattr(tokenizer, "apply_chat_template")
    if has_method:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            return text, True
        except Exception as e:  # 模板存在但使用失败（模板变量缺失等）
            print(e)
            return default_trans_messages(messages), False
    # 无模板：直接回退
    return default_trans_messages(messages), False


class ChatTemplateDefaultsMixin:
    """为 backend 提供统一的 chat template 默认参数管理。"""
    _chat_template_defaults: Dict[str, Any]

    def __init__(self, *args, **kwargs):
        # 子类若有多重继承，记得 super().__init__() 放在最后或调用顺序合适的地方
        self._chat_template_defaults = {}
        super().__init__(*args, **kwargs)

    def set_chat_template_defaults(self, **defaults: Any) -> None:
        self._chat_template_defaults.update(defaults)

    def get_chat_template_defaults(self) -> Dict[str, Any]:
        return dict(self._chat_template_defaults)

    def reset_chat_template_defaults(self) -> None:
        self._chat_template_defaults.clear()