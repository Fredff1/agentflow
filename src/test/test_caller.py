import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)     

from agentflow.tools.caller import ToolCaller
from agentflow.tools.registry import ToolRegistry
from agentflow.tools.parser import TagToolParser
from agentflow.tools.base import ToolCallRequest
from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.tools.search.base_search import AsyncSearchTool
from agentflow.tools.search.backend.searxng import SearxngBackend


def test_one():
    registry=ToolRegistry()
    py_tool = PythonExecutionTool()
    search_tool = AsyncSearchTool(
        backend=SearxngBackend("http://127.0.0.1:8888")
    )
    registry.register(py_tool)
    registry.register(search_tool)
    caller = ToolCaller(
        registry=registry,
        parser=TagToolParser(["search","python"])
    )
    results = caller.call_single("<search>python debug</search>")
    print(results)
    results = caller.call_single("<python>print(100)</python>")
    print(results)
    
def test_batch():
    registry=ToolRegistry()
    py_tool = PythonExecutionTool()
    search_tool = AsyncSearchTool(
        backend=SearxngBackend("http://127.0.0.1:8888")
    )
    registry.register(py_tool)
    registry.register(search_tool)
    caller = ToolCaller(
        registry=registry,
        parser=TagToolParser(["search","python"])
    )
    batch_results = caller.call_batch(
        ["<search>python learning</search>",
         "<python>print(1001)</python>"]
    )
    print(batch_results)
    

if __name__ == "__main__":
    test_batch()
    test_one()