# tests/test_python_execution_tool.py

import sys
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)     

from src.tools.code.python_execution import PythonExecutionTool

def test_python_tool_print_basic():
    tool = PythonExecutionTool(timeout_length=5)
    content, meta = tool.run_one("print(1+2+3)")
    print(content)
    assert meta["success"] is True
    assert "Execution Success" in meta["report"]
    assert "6" in content["result"]

def test_python_tool_forbidden_calls():
    tool = PythonExecutionTool(timeout_length=3)
    content, meta = tool.run_one("input('x')")
    assert meta["success"] is False
    assert "Execution Failed" in meta["report"]
    assert "Forbidden" in meta["report"] 

def test_python_tool_timeout():
    tool = PythonExecutionTool(timeout_length=1)
    content, meta = tool.run_one("import time\nprint('start')\ntime.sleep(20)\nprint('end')")
    assert meta["success"] is False
    assert "Execution Failed" in meta["report"]
    
    assert "Timeout" in meta["report"] or "timeout" in meta["report"]

def test_python_tool_batch():
    tool = PythonExecutionTool(timeout_length=5)
    codes = [
        "print(sum(range(10)))",         # 45
        "x=2\ny=3\nprint(x**y)",         # 8
    ]
    outs, metas = tool.run_batch(codes)
    assert len(outs) == len(metas) == 2
    assert all(m["success"] for m in metas)
    assert "45" in outs[0]["result"]
    assert "8" in outs[1]["result"]
    
if __name__ == "__main__":
    test_python_tool_print_basic()
    test_python_tool_batch()
    test_python_tool_forbidden_calls()
    test_python_tool_timeout()
