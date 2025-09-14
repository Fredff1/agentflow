import json
import jsonlines
import re

from typing import List, Dict, Any
from dataclasses import is_dataclass, asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
import collections.abc as cabc


class JsonUtil:

    @staticmethod
    def read_json(file_path:str,mode="r"):
        """Read json"""
        with open(file_path, mode, encoding='utf-8') as file:
            return json.load(file)

    @staticmethod
    def write_json(file_path:str,data:List[Dict],mode="w"):
        """Write json"""
        with open(file_path, mode, encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

            
    @staticmethod
    def parse_json(text:str)->List[Any] | None:
        """Read and parse all json in the given text

        Args:
            text (str): input text

        Returns:
            list[dict] | None: json item
        """
        all_matches = re.findall(r'\{.*?\}', text, re.DOTALL)
        if len(all_matches) >= 1:
            json_match=all_matches[0]
        else:
            json_match=None
        json_list=[]
        if json_match:
            for each_mach in all_matches:
                try:
                    # parse json object
                    json_data = json.loads(each_mach)
                    json_list.append(json_data)
                except:
                    pass
        return json_list
     

    @staticmethod
    def read_jsonlines(file_path,mode="r"):
        """Read jsonlines

        """
        data = []
        with open(file_path, mode, encoding='utf-8') as f:
            for line in jsonlines.Reader(f):
                data.append(line)
        return data

    @staticmethod
    def write_jsonlines(file_path, data,mode="a"):
        """write jsonlines

        """
        with open(file_path, mode, encoding="utf-8") as w:
            if isinstance(data,list):
                for i in data:
                    json.dump(i, w, ensure_ascii=False)
                    w.write('\n')
            else:
                json.dump(data, w, ensure_ascii=False)
                w.write('\n')
                
    @staticmethod      
    def json_sanitize(obj: Any, *, on_unknown: str = "drop", max_depth: int = 64) -> Any | None:
        """
        将任意对象清洗为“可被 json 序列化”的结构。
        - dict: 递归清洗，键统一转 str；不可序列化的键/值会被丢弃
        - list/tuple/set: 递归清洗；set 转 list；不可序列化项被丢弃
        - dataclass -> asdict；Enum -> value(可再递归)；Path/bytes/datetime 等做常见转换
        - 具有 .tolist() / .item() 的对象（如 numpy/pandas/torch）会优先尝试这些转换
        - 其他未知对象：on_unknown="drop" 丢弃；"stringify" 则转 str
        - 若顶层整体不可序列化，返回 None
        """
        _SKIP = object()

        def _inner(o: Any, depth: int) -> Any:
            if depth > max_depth:
                return _SKIP

            # 基础可序列化类型
            if isinstance(o, (str, int, float, bool)) or o is None:
                return o

            # 常见转换
            if isinstance(o, (bytes, bytearray)):
                return o.decode("utf-8", errors="replace")
            if isinstance(o, (date, datetime)):
                return o.isoformat()
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, Enum):
                return _inner(o.value, depth + 1)

            # dataclass
            if is_dataclass(o):
                o = asdict(o)

            # torch/numpy/pandas 等：duck-typing 转基础类型
            if hasattr(o, "detach") and callable(getattr(o, "detach")) and hasattr(o, "cpu"):
                try:
                    return _inner(o.detach().cpu(), depth + 1)
                except Exception:
                    pass
            if hasattr(o, "tolist"):
                try:
                    return _inner(o.tolist(), depth + 1)
                except Exception:
                    pass
            if hasattr(o, "item"):
                try:
                    return _inner(o.item(), depth + 1)
                except Exception:
                    pass

            # 映射类型
            if isinstance(o, cabc.Mapping):
                out = {}
                for k, v in o.items():
                    try:
                        sk = k if isinstance(k, str) else str(k)
                    except Exception:
                        continue
                    sv = _inner(v, depth + 1)
                    if sv is not _SKIP:
                        out[sk] = sv
                return out

            # 序列类型（包含 set / tuple / list）
            if isinstance(o, (list, tuple, set, frozenset)) or (isinstance(o, cabc.Sequence) and not isinstance(o, (str, bytes, bytearray))):
                out = []
                for it in list(o):  # set/frozenset 转列表
                    si = _inner(it, depth + 1)
                    if si is not _SKIP:
                        out.append(si)
                return out

            # 其他未知对象
            if on_unknown == "stringify":
                try:
                    return str(o)
                except Exception:
                    return _SKIP
            return _SKIP

        cleaned = _inner(obj, 0)
        return None if cleaned is _SKIP else cleaned

def load_dataset(path: str):
    if path.endswith(".json"):
        data = JsonUtil.read_json(path)
    elif path.endswith(".jsonl"):
        data = JsonUtil.read_jsonlines(path)
    else:
        raise ValueError("Dataset format not supported")
    return data