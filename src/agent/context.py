# src/agent/context.py
from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Tuple

# 你已有的 Message 定义：..common.messages.Message
from ..common.messages import Message
from ..tools.base import ToolCallResult  # 如暂时未用可删

RoundRole = Literal["user", "system", "assistant", "tool"]


@dataclass
class RoundBuffer:
    full: str = ""
    detail: List[Message] = field(default_factory=list)


@dataclass
class AgentContext:
  
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_index: int = 0

    head: str = ""                 
    text: str = ""                 
    current: str = ""              

    global_round: int = 0
    processes: List[RoundBuffer] = field(default_factory=list)
    round_counters: Dict[str, int] = field(default_factory=dict)  # 按工具/阶段统计可用
    tool_results: List[ToolCallResult] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # 可扩展位（后续做 RAG/记忆/追踪等直接在此加字段即可）
    # memory_snapshot: Optional[Dict[str, Any]] = None
    # retrieval_hints: List[str] = field(default_factory=list)
    # tags: List[str] = field(default_factory=list)
    # configurable: Dict[str, Any] = field(default_factory=dict)
    # framework_state: Dict[str, Any] = field(default_factory=dict)

    # ---------- 基础轮次维护 ----------

    def set_round(self, round_index: int) -> None:
        self.global_round = int(round_index)
        self._ensure_round_exists(self.global_round)

    def begin_new_round(self) -> int:
        self.global_round += 1
        self._ensure_round_exists(self.global_round)
        self.current = ""
        return self.global_round

    def _ensure_round_exists(self, idx: int) -> None:
        while len(self.processes) <= idx:
            self.processes.append(RoundBuffer())

    # ---------- 读写与追加 ----------

    def tail(self) -> str:
        """返回 text 中除去 head 的部分。"""
        return self.text[len(self.head) :] if len(self.text) >= len(self.head) else ""

    def append(
        self,
        text: str | None = None,
        *,
        role: RoundRole = "assistant",
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        commit_current: bool = False,
    ) -> Message:
        """
        追加一段文本到上下文中，同时在当前轮记录一条 Message。

        Args:
            text:  要追加的文本。若为 None，则使用 self.current 的内容（并清空 current）。
            role:  Message 角色，默认 assistant。
            name:  可选的 name（用于 tool 结果等）。
            metadata: 附加的元信息。
            commit_current: 若 True，先把 current 也一并提交。

        Returns:
            新增的 Message 对象。
        """
        self._ensure_round_exists(self.global_round)
        rb = self.processes[self.global_round]

        # 处理 current 提交
        if commit_current and self.current:
            self.text += self.current
            rb.full += self.current
            rb.detail.append(Message(role=role, content=self.current, name=name, metadata=metadata or {}))
            self.current = ""

        content = (text if text is not None else self.current) or ""
        # 将本次 content 写入 text / full，并记录明细
        if content:
            self.text += content
            rb.full += content
            msg = Message(role=role, content=content, name=name, metadata=metadata or {})
            rb.detail.append(msg)
            if text is None:  # 如果是把 current 当作正文提交，提交后清空
                self.current = ""
            return msg

        # 允许追加空消息（极少见），也给出占位
        msg = Message(role=role, content="", name=name, metadata=metadata or {})
        rb.detail.append(msg)
        return msg

    def pop(self) -> Optional[Dict[str, Any]]:
        """
        移除最后一条已记录的消息。若该消息的 content 正好是 text 的后缀，则从 text 末尾同步删除。
        返回结构：{"role": ..., "content": ..., "round": int}
        """
        for r in range(len(self.processes) - 1, -1, -1):
            detail = self.processes[r].detail
            if not detail:
                continue
            last_msg = detail.pop()
            content = last_msg.content or ""

            full_str = self.processes[r].full
            if content and full_str.endswith(content):
                self.processes[r].full = full_str[: len(full_str) - len(content)]
            else:
                idx = full_str.rfind(content) if content else -1
                if idx != -1 and idx + len(content) == len(full_str):
                    self.processes[r].full = full_str[:idx]

            if content:
                if self.text.endswith(content) and len(self.text) >= len(self.head) + len(content):
                    self.text = self.text[: len(self.text) - len(content)]
                else:
                    pass

            return {"role": last_msg.role, "content": content, "round": r}
        return None



    def append_user(self, text: str) -> Message:
        return self.append(text, role="user")

    def append_system(self, text: str) -> Message:
        return self.append(text, role="system")

    def append_assistant(self, text: str) -> Message:
        return self.append(text, role="assistant")

    def append_tool_observation(self, text: str, name: str = "observations", metadata: Optional[Dict[str, Any]] = None) -> Message:
        return self.append(text, role="tool", name=name, metadata=metadata or {})

    def update_round_counter(self, tool_result: ToolCallResult):
        tool_name = tool_result.tool_name
        if tool_name in self.round_counters.keys():
            self.round_counters[tool_name]+=1
        else:
            self.round_counters[tool_name]=1
        self.tool_results.append(tool_result)

    def last_message(self) -> Optional[Message]:
        for r in range(len(self.processes) - 1, -1, -1):
            if self.processes[r].detail:
                return self.processes[r].detail[-1]
        return None

    def flatten_messages(self) -> List[Message]:
        out: List[Message] = []
        for rb in self.processes:
            out.extend(rb.detail)
        return out


    @classmethod
    def from_head(cls, head: str, meta: Optional[Dict[str, Any]] = None) -> AgentContext:
        """以 head 初始化一个上下文：text=head，当前轮为 0。"""
        ctx = cls(head=head, text=head, meta=meta or {})
        ctx._ensure_round_exists(0)
        return ctx
