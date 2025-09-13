# python_execution_tool.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
import multiprocess
from multiprocess import Pool
from typing import Any, Dict, Optional, List, Tuple
from pebble import ProcessPool
from tqdm import tqdm
from concurrent.futures import TimeoutError as PebbleTimeoutError
from functools import partial
from timeout_decorator import timeout
from contextlib import redirect_stdout
import sympy
import math
from sympy import symbols, Eq, solve
from scipy import optimize

from ..base import BaseTool, ToolCallRequest, ToolCallResult
# 如果需要日志：from utils.log_util import get_logger


def _safe_traceback(exc: BaseException, limit: int = 2) -> str:
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    if len(lines) > limit:
        lines = lines[: limit - 1] + ["... (truncated)\n"]
    return "".join(lines)


# -------- Sandboxed runtimes --------
class GenericRuntime:
    GLOBAL_DICT: Dict[str, Any] = {}
    LOCAL_DICT: Optional[Dict[str, Any]] = None
    HEADERS: List[str] = []

    def __init__(self) -> None:
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None
        for code_piece in self.HEADERS:
            self.exec_code(code_piece)

    def exec_code(self, code_piece: str) -> None:
        pre_imports = """
import numpy as np
import sympy
import math
from sympy import symbols, Eq, solve
x, y, z = sympy.symbols('x y z')
"""
        if regex.search(r'(\s|^)?input\(', code_piece) or regex.search(r'(\s|^)?os\.system\(', code_piece):
            raise RuntimeError("Forbidden calls detected: input() / os.system()")

        exec(pre_imports, self._global_vars)
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        self._global_vars.update(var_dict)

    @property
    def answer(self) -> Any:
        return self._global_vars.get("answer")


class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        "datetime": datetime.datetime,
        "timedelta": dateutil.relativedelta.relativedelta,
        "relativedelta": dateutil.relativedelta.relativedelta,
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()


class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {"dict": CustomDict}


# -------- Executor --------
class PythonExecutor:
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expression: Optional[str] = None,
        capture_answer_from_stdout: bool = False,
        timeout_length: int = 5,
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expression = get_answer_expression
        self.capture_answer_from_stdout = capture_answer_from_stdout
        self.process_pool = Pool(multiprocess.cpu_count())
        self.timeout_length = timeout_length

    @staticmethod
    def execute(
        code: Any,
        capture_answer_from_stdout: Optional[bool] = None,
        runtime: Optional[Any] = None,
        answer_symbol: Optional[str] = None,
        answer_expression: Optional[str] = None,
        timeout_length: int = 10,
    ) -> Tuple[Any, str]:
        try:
            if isinstance(code, list):
                code = "\n".join(code)
            code = str(code).strip()

            if capture_answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    timeout(timeout_length)(runtime.exec_code)(code)
                program_io.seek(0)
                result = program_io.read()
            elif answer_symbol:
                timeout(timeout_length)(runtime.exec_code)(code)
                result = runtime._global_vars[answer_symbol]
            elif answer_expression:
                timeout(timeout_length)(runtime.exec_code)(code)
                result = timeout(timeout_length)(runtime.eval_code)(answer_expression)
            else:
                lines = code.split("\n")
                if len(lines) > 1:
                    exec_code = "\n".join(lines[:-1])
                    eval_code = lines[-1]
                    timeout(timeout_length)(runtime.exec_code)(exec_code)
                    result = timeout(timeout_length)(runtime.eval_code)(eval_code)
                else:
                    result = timeout(timeout_length)(runtime.eval_code)(code)

            report = "Execution Success"

            try:
                pickle.dumps(result)
            except (pickle.PicklingError, TypeError):
                try:
                    result = str(result)
                except Exception:
                    result = f"<unprintable object of type {type(result).__name__}>"

        except Exception as e:
            result = ""
            report = f"Execution Failed\n{type(e).__name__}: {str(e)}\nTraceback: {_safe_traceback(e)}"
        return result, report

    @staticmethod
    def _truncate(text: str, max_length: int = 400) -> str:
        if len(text) <= max_length:
            return text
        half = max_length // 2
        return text[:half] + "..." + text[-half:]

    def apply(self, code: str) -> Tuple[str, str]:
        return self.batch_apply([code])[0]

    def batch_apply(self, batch_code: List[str]) -> List[Tuple[str, str]]:
        all_code_snippets: List[List[str]] = [c.split("\n") for c in batch_code]

        all_exec_results: List[Tuple[Any, str]] = []
        with ProcessPool(max_workers=min(len(all_code_snippets), os.cpu_count())) as pool:
            executor = partial(
                self.execute,
                capture_answer_from_stdout=self.capture_answer_from_stdout,
                runtime=self.runtime,
                answer_symbol=self.answer_symbol,
                answer_expression=self.answer_expression,
                timeout_length=self.timeout_length,
            )
            future = pool.map(executor, all_code_snippets, timeout=self.timeout_length)
            iterator = future.result()

            progress_bar = tqdm(total=len(all_code_snippets), desc="Execute") if len(all_code_snippets) > 100 else None
            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except PebbleTimeoutError:
                    all_exec_results.append(("", "Timeout Error"))
                except Exception:
                    raise
                if progress_bar is not None:
                    progress_bar.update(1)
            if progress_bar is not None:
                progress_bar.close()

        batch_results: List[Tuple[str, str]] = []
        for (res, report) in all_exec_results:
            res_s = self._truncate(str(res).strip())
            rep_s = self._truncate(str(report).strip())
            batch_results.append((res_s, rep_s))
        return batch_results


# -------- Tool wrapper --------
class PythonExecutionTool(BaseTool):
    name = "python"
    description = "Execute Python code safely with optional stdout capture."

    def __init__(self, *, timeout_length: int = 5, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        self.executor = PythonExecutor(
            runtime=GenericRuntime(),
            get_answer_symbol=None,
            get_answer_expression=None,
            capture_answer_from_stdout=True,
            timeout_length=timeout_length,
        )

    def run_one(self, call: ToolCallRequest, **kwargs: Any) -> ToolCallResult:
        code_text = str(call.content)
        result, report = self.executor.apply(code_text)
        meta = {
            "success": report.startswith("Execution Success"),
            "report": report,
        }
        result = ToolCallResult(
            tool_name=self.name,
            request_content=call.content,
            output=f"Console: {result}\n Report: {report}",
            meta=meta,
            error=None,
            index =call.index,
            call=call,
        )
        return result

    def run_batch(self, calls: List[ToolCallRequest], **kwargs: Any) -> List[ToolCallResult]:
        results = self.executor.batch_apply([str(call.content) for call in calls])
        metas = [{"success": rep.startswith("Execution Success"), "report": rep} for (_, rep) in results]
        final_results: List[ToolCallResult] = []
        for (result, report), call, meta in zip(results,calls,metas):
            final_results.append(ToolCallResult(
                tool_name=self.name,
                request_content=call.content,
                output=f"Console: {result}\n Report: {report}",
                meta=meta,
                error=None,
                index=call.index,
                call=call,
            ))
        return final_results
