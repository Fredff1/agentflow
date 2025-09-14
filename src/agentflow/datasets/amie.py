from typing import Sequence, List, Dict, Any, Optional
from agentflow.utils.json_util import JsonUtil

def load_aime2025(input_paths: List[str],save_path: Optional[str] = None):
    all_data = []
    for pth in input_paths:
        all_data.extend(JsonUtil.read_jsonlines(pth))
    
    if save_path:
        JsonUtil.write_jsonlines(save_path,all_data)
    return all_data