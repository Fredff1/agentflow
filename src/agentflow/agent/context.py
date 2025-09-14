# src/agent/context.py
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from agentflow.common.messages import Message
from agentflow.tools.base import ToolCallResult  # 若当前未用，可移除

RoundRole = Literal["user", "system", "assistant", "tool"]


@dataclass
class RoundBuffer:
    """单轮消息缓冲。"""
    messages: List[Message] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """按需拼接该轮消息文本（调试/日志用），不在状态里存字符串。"""
        parts = []
        for m in self.messages:
            name = f" ({m.name})" if getattr(m, "name", None) else ""
            parts.append(f"[{m.role}{name}] {m.content}")
        return "\n".join(parts)


@dataclass
class AgentContext:
    """
    轻量跨-step上下文（非对话态），完全以 messages 为一等公民。
    - prompt_messages: 初始上下文（如 system/user），外部传入后基本不改
    - rounds: 分轮追加的消息（模型输出/工具观察/系统提示等）
    - global_round: 当前轮索引
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_index: int = 0

    prompt_messages: List[Message] = field(default_factory=list)  # 初始 system/user...
    rounds: List[RoundBuffer] = field(default_factory=list)
    global_round: int = 0

    # 统计/记录
    round_counters: Dict[str, int] = field(default_factory=dict)
    tool_results: List[ToolCallResult] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -------- 轮次维护 --------
    def set_round(self, round_index: int) -> None:
        self.global_round = int(round_index)
        self._ensure_round_exists(self.global_round)

    def begin_new_round(self) -> int:
        self.global_round += 1
        self._ensure_round_exists(self.global_round)
        return self.global_round

    def _ensure_round_exists(self, idx: int) -> None:
        while len(self.rounds) <= idx:
            self.rounds.append(RoundBuffer())

    # -------- 读写与追加（message-first）--------
    def append(
        self,
        content: str,
        *,
        role: RoundRole = "assistant",
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """向当前轮追加一条消息。"""
        self._ensure_round_exists(self.global_round)
        msg = Message(role=role, content=content, name=name, metadata=metadata or {})
        self.rounds[self.global_round].messages.append(msg)
        return msg

    def append_user(self, text: str) -> Message:
        return self.append(text, role="user")

    def append_system(self, text: str) -> Message:
        return self.append(text, role="system")

    def append_assistant(self, text: str) -> Message:
        return self.append(text, role="assistant")

    def append_tool_observation(self, text: str, *, name: str = "observations", metadata: Optional[Dict[str, Any]] = None) -> Message:
        return self.append(text, role="tool", name=name, metadata=metadata)

    def append_messages(self, msgs: List[Message]) -> None:
        """批量把消息追加到当前轮。"""
        self._ensure_round_exists(self.global_round)
        self.rounds[self.global_round].messages.extend(msgs)

    def pop(self) -> Optional[Message]:
        """从最后一个非空轮中弹出最后一条消息。"""
        for r in range(len(self.rounds) - 1, -1, -1):
            if self.rounds[r].messages:
                return self.rounds[r].messages.pop()
        return None

    # -------- 访问视图 --------
    def current_round_buffer(self) -> RoundBuffer:
        self._ensure_round_exists(self.global_round)
        return self.rounds[self.global_round]

    def current_round_messages(self) -> List[Message]:
        return list(self.current_round_buffer().messages)

    def round_messages(self, round_index: int) -> List[Message]:
        self._ensure_round_exists(round_index)
        return list(self.rounds[round_index].messages)
    
    def all_round_messages(self) -> List[Message]:
        msg_list = []
        for round in self.rounds:
            msg_list.extend(round.messages)
        return msg_list

    def all_messages(self) -> List[Message]:
        """返回送入后端的完整消息序列：初始 + 各轮累积。"""
        out: List[Message] = list(self.prompt_messages)
        for rb in self.rounds:
            out.extend(rb.messages)
        return out

    def last_message(self) -> Optional[Message]:
        for r in range(len(self.rounds) - 1, -1, -1):
            msgs = self.rounds[r].messages
            if msgs:
                return msgs[-1]
        return None

    # -------- 统计/工具结果 --------
    def update_round_counter(self, tool_result: ToolCallResult) -> None:
        tool_name = getattr(tool_result, "tool_name", None) or tool_result.get("tool_name", "tool")
        new_counter = self.round_counters.get(tool_name, 0) + 1
        self.round_counters[tool_name] = new_counter
        self.meta["round_counter"][tool_name] = new_counter
        self.tool_results.append(tool_result)

    # -------- 构造器 --------
    @classmethod
    def from_messages(cls, messages: List[Message], meta: Optional[Dict[str, Any]] = None) -> AgentContext:
        ctx = cls(prompt_messages=list(messages), meta=meta or {})
        ctx._ensure_round_exists(0)  # 准备第 0 轮
        return ctx
