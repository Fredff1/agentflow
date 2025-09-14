import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)  

import re
import pytest

from typing import List, Dict, Tuple, Any, Optional

from agentflow.agent.basic import ToolDrivenAgent
from agentflow.agent.context import AgentContext
from agentflow.core.interfaces import CanGenerate
from agentflow.common.messages import Message
from agentflow.tools.base import BaseTool, ToolCallRequest, ToolCallResult, ToolParser
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.caller import ToolCaller
from agentflow.tools.parser import TagToolParser



class FakeSearchTool(BaseTool):
    name = "search"

    def run_one(self, call: ToolCallRequest, **kwargs: Any) -> ToolCallResult:
        q = str(call.content)
        return ToolCallResult(
            tool_name=self.name,
            request_content=q,
            output=f"RESULT_FOR({q})",
            meta={"echo": True},
            error=None,
            index=call.index,
            call=call,
        )


class FakeBackend(CanGenerate):
    """
    规则：
      - 如果初始 user 里含 'NEED_TOOL'，且消息中目前还没有 tool 消息，则返回触发工具的回复：<search>foo</search>
      - 如果历史消息中已经出现过 tool 消息，则返回终止：'Final Answer: from tool'
      - 如果初始 user 含 'FINAL_FIRST'，直接返回：'Final Answer: immediate'
      - 其他情况：返回普通文本 'thinking...'（既无工具也无终止）
    """
    def generate(self, prompts: List, extra: List[Dict] | None = None, **kwargs) -> Tuple[List[str], List[Dict]]:
        outs, metas = [], []
        for messages in prompts:
            # messages 是 list[dict]，按你目前的输入约定
            # 寻找初始 user 内容
            users = [m for m in messages if m.get("role") == "user"]
            user0 = (users[0]["content"] if users else "") if users else ""
            # 是否已有 tool 观察
            has_tool = any(m.get("role") == "tool" for m in messages)

            if "FINAL_FIRST" in user0:
                outs.append("Final Answer: immediate")
            elif "NEED_TOOL" in user0 and not has_tool:
                outs.append("<search>foo</search>")
            elif has_tool:
                outs.append("Final Answer: from tool")
            else:
                outs.append("thinking...")

            metas.append({"fake": True})
        return outs, metas


def default_finish_fn(ctx: AgentContext) -> bool:
    last = ctx.last_message()
    if not last or last.role != "assistant":
        return False
    return "Final Answer" in (last.content or "")


def make_agent(max_rounds: int = 4) -> ToolDrivenAgent:
    registry = ToolRegistry()
    registry.register(FakeSearchTool())  
    parser = TagToolParser()
    caller = ToolCaller(registry=registry, parser=parser)

    agent = ToolDrivenAgent(
        backend=FakeBackend(),
        tool_caller=caller,
        finish_fn=default_finish_fn,
        max_rounds=max_rounds,
    )
    return agent


def test_agent_tool_then_finish():
    agent = make_agent(max_rounds=4)
    prompts = [["""{"role":"user","content":"NEED_TOOL please"}"""]]  

    msgs = [[{"role": "user", "content": "NEED_TOOL please"}]]

    texts, metas = agent.generate(msgs)
    assert len(texts) == 1
    assert texts[0].startswith("Final Answer"), f"unexpected final: {texts[0]}"
    assert len(metas[0]["steps"]) >= 2
    assert "tool_results" in metas[0]["steps"][0]
    assert metas[0]["steps"][0]["tool_results"][0].output == "RESULT_FOR(foo)"


def test_agent_finish_immediately():
    agent = make_agent(max_rounds=4)
    msgs = [[{"role": "user", "content": "FINAL_FIRST"}]]

    texts, metas = agent.generate(msgs)
    assert len(texts) == 1
    assert texts[0] == "Final Answer: immediate"
    assert len(metas[0]["steps"]) == 1
    assert "tool_results" not in metas[0]["steps"][0]


def test_agent_no_tool_no_finish_uses_max_rounds():
    agent = make_agent(max_rounds=3)
    msgs = [[{"role": "user", "content": "NOTHING"}]]

    texts, metas = agent.generate(msgs)
    assert len(texts) == 1
    assert texts[0] == "thinking..."
    assert len(metas[0]["steps"]) == 3

if __name__ == "__main__":
    test_agent_finish_immediately()
    test_agent_no_tool_no_finish_uses_max_rounds()
    test_agent_tool_then_finish()