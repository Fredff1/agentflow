import json
import jsonlines
import re

from typing import List, Dict, Any



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

def load_dataset(path: str):
    if path.endswith(".json"):
        data = JsonUtil.read_json(path)
    elif path.endswith(".jsonl"):
        data = JsonUtil.read_jsonlines(path)
    else:
        raise ValueError("Dataset format not supported")
    return data