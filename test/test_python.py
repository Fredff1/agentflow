# tests/test_python_execution_tool.py

import sys
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)     

from src.tools.code.python_execution import PythonExecutionTool
from src.tools.base import ToolCallRequest

def test_python_tool_print_basic():
    tool = PythonExecutionTool(timeout_length=5)
    request = ToolCallRequest(
        index=0,
        name="search",
        content="print(1+2+3)"
    )
    result = tool.run_one(request)
    print(result)


def test_python_tool_forbidden_calls():
    tool = PythonExecutionTool(timeout_length=3)
    request = ToolCallRequest(
        index=0,
        name="search",
        content="input('x')"
    )
    result = tool.run_one(request)
    print(result)
    

def test_python_tool_timeout():
    tool = PythonExecutionTool(timeout_length=3)
    request = ToolCallRequest(
        index=0,
        name="search",
        content="import time\nprint('start')\ntime.sleep(20)\nprint('end')"
    )
    result = tool.run_one(request)
    print(result)

def test_python_tool_batch():
    tool = PythonExecutionTool(timeout_length=5)
    requests = [
        ToolCallRequest(
        index=0,
        name="search",
        content="print(sum(range(10)))"
    ),
        ToolCallRequest(
        index=0,
        name="search",
        content="x=2\ny=3\nprint(x**y)"
    )
    ]
    results = tool.run_batch(requests)
    print(results)
    
if __name__ == "__main__":
    test_python_tool_print_basic()
    test_python_tool_batch()
    test_python_tool_forbidden_calls()
    test_python_tool_timeout()
