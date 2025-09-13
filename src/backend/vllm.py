from typing import List, Tuple, Dict, Any
from logging import Logger


from vllm import LLM, SamplingParams

from ..core.interfaces import CanGenerate,SupportChatTemplate
from ..utils.log_util import get_logger

class VllmBackend(CanGenerate, SupportChatTemplate):
    
    def __init__(self, config: Dict[str,Any], logger: Logger = None, **kwargs):
        super().__init__()
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = get_logger(config, __name__)
        self._parse_config()
        self.sampling_params = SamplingParams(
            temperature=self.sampling_config.get("temperature",1),
            max_tokens=self.sampling_config.get("max_tokens",1024),
            top_p=self.sampling_config.get("top_p",1.0),
            top_k=self.sampling_config.get("top_k",20),
            stop=self.vllm_config.get("stop_tokens",[]),
            include_stop_str_in_output=True,
        )
        self.engine = LLM(
            model=self.backend_config["model_path"],
            dtype=self.backend_config.get("dtype","auto"),
            gpu_memory_utilization=self.vllm_config.get("gpu_memory_utilization",0.8),
            tensor_parallel_size=self.vllm_config["tensor_parallel_size"],
            trust_remote_code=True,
        )
        
        self.tokenizer = self.engine.get_tokenizer()
    
    
    def apply_chat_template(self, messages: List[Dict[str,str]], 
                            tokenize=False, 
                            add_generation_prompt=True, 
                            **additional_params) -> str:
        tokens = self.tokenizer.apply_chat_template(
            messages, 
            tokenize = tokenize,
            add_generation_prompt = add_generation_prompt,
            **additional_params)
        return tokens  
    
    def generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
        outputs = self.engine.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
        )
        texts = [output.outputs[0].text for output in outputs]
        return texts, [{"raw_output": output} for output in outputs]
    
    def _parse_config(self):
        backend_config = self.config["backend"]
        sampling_config = backend_config["sampling"]
        vllm_config = backend_config["vllm"]
        self.backend_name = "vllm"
        self.backend_config = backend_config
        self.sampling_config = sampling_config
        self.vllm_config = vllm_config
        