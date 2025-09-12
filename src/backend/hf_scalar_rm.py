# src/backends/hf_rm_backend.py
from typing import Sequence, List, Dict, Any, Optional
from logging import Logger
import torch
from transformers import AutoTokenizer, AutoModel
from ..utils.log_util import get_logger
from ..core.interfaces import SupportChatTemplate


class HFRMBackend(SupportChatTemplate):
    """
    仅支持【标量】奖励模型（Regression / num_labels=1）。
    - 配置来源：config["backend"]["rm"]（未提供的字段可回退到 backend 本体，如 model_path / dtype）
    - 推理路径：
        1) 若模型实现了 remote code 的 get_scores / get_score，则优先调用（如 InternLM2-7B-Reward）
        2) 否则走通用前向，要求 logits 的最后一维为 1；得到 raw 标量后可选 sigmoid，再裁剪到 [clip_low, clip_high]
    - 返回：List[float]，位于 [0,1]（若 apply_sigmoid=False，则返回可能不在 [0,1]，但最后仍会裁剪）
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        self.logger: Logger = logger if logger else get_logger(config, __name__)

        backend_config: Dict[str, Any] = config.get("backend", {})
        rm_config: Dict[str, Any] = backend_config.get("rm", {})
        sampling_config = backend_config.get("sampling", {})

        # 基本参数
        self.model_path: str = backend_config.get("model_path", "")
        self.device: str = rm_config.get("device", "cuda")
        self.torch_dtype_name: str = backend_config.get("dtype", "auto")
        self.max_length: int = int(sampling_config.get("max_length", 1024))
        self.padding_side: str = rm_config.get("padding_side", "right")
        self.truncation_side: str = rm_config.get("truncation_side", "left")  # 标量RM常保留结尾更重要

        # 标量归一化相关
        self.temperature: float = float(sampling_config.get("temperature", 1.0))
        self.apply_sigmoid: bool = bool(rm_config.get("apply_sigmoid", True))
        clip_range = rm_config.get("clip_range", [0.0, 1.0])
        self.clip_low: float = float(clip_range[0])
        self.clip_high: float = float(clip_range[1])

        # 批次
        self.batch_size: int = int(rm_config.get("batch_size", -1))

        # —— 加载 tokenizer / model —— #
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = self.padding_side
        # truncation_side 只有 fast tokenizer 支持；若不支持也不影响功能
        try:
            self.tokenizer.truncation_side = self.truncation_side
        except Exception:
            pass

        torch_dtype = "auto" if self.torch_dtype_name == "auto" else getattr(torch, self.torch_dtype_name)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

        # 记录一下 config 中的 num_labels（仅用于日志提示）
        self._cfg_num_labels = getattr(self.model.config, "num_labels", None)
        self.logger.info(
            f"[HFRMBackend] (scalar-only) config.num_labels={self._cfg_num_labels}, "
            f"padding_side={self.padding_side}, truncation_side={self.truncation_side}"
        )

    # --------- Chat 模板（保留接口以兼容你的上游调用） --------- #
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **additional_params
    ) -> str:
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **additional_params,
        )
        return tokens

    # --------- 内部：优先尝试 remote code 的 get_scores / get_score --------- #
    def _try_remote_scores(self, batch_texts: List[str]) -> Optional[torch.Tensor]:
        """
        如果模型实现了 remote code 的 get_scores/get_score，则使用之。
        该方法期望输入为 messages 列表，这里把每条纯文本包成：
            [{"role": "user", "content": <text>}]
        返回：Tensor[B] 或 None
        """
        model = self.model
        has_get_scores = hasattr(model, "get_scores")
        has_get_score = hasattr(model, "get_score")
        if not (has_get_scores or has_get_score):
            return None

        try:
            chats = [[{"role": "user", "content": t}] for t in batch_texts]
            if has_get_scores:
                scores = model.get_scores(self.tokenizer, chats)  # -> List[float]
            else:
                # 逐条 get_score
                scores = [model.get_score(self.tokenizer, c) for c in chats]
            # 统一后处理到 [0,1] 区间（若需要）
            scores_t = torch.tensor(scores, device=self.device, dtype=torch.float32)
            # remote code 一般已是“高分更好”的标量；根据需要可加温度和 Sigmoid，这里保持与你总流程一致
            if self.temperature != 1.0:
                scores_t = scores_t / max(1e-6, self.temperature)
            if self.apply_sigmoid:
                scores_t = torch.sigmoid(scores_t)
            scores_t = scores_t.clamp(self.clip_low, self.clip_high)
            return scores_t
        except Exception as e:
            self.logger.warning(f"[HFRMBackend] remote get_scores failed, fallback to logits. err={e}")
            return None

    # --------- 主函数：仅标量 RM 路径 --------- #
    @torch.no_grad()
    def rm_scores(
        self,
        sequences: Sequence[str],
        extra: List[Dict[str, Any]] | None = None,
        **kwargs: Any
    ) -> List[float]:
        """
        输入：Sequence[str]（已模板化的纯文本）；若你想喂 messages，请在上游用 apply_chat_template 变成字符串
        输出：标量分数列表，默认压缩至 [0,1]
        """
        if not sequences:
            return []

        normalized_scores: List[float] = []

        batch_size = self.batch_size if (self.batch_size and self.batch_size > 0) else len(sequences)
        for start in range(0, len(sequences), batch_size):
            batch_texts = list(sequences[start: start + batch_size])

            # 1) 先试 remote code（如 InternLM2-7B-Reward）
            # remote_scores = self._try_remote_scores(batch_texts)
            remote_scores = None
            if remote_scores is not None:
                normalized_scores.extend(remote_scores.detach().float().cpu().tolist())
                continue

            # 2) 回退到通用前向：要求 logits 的最后一维 == 1（标量）
            encodings = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            out = self.model(**encodings)
            logits = getattr(out, "logits", None)
            if logits is None:
                # 某些 remote code 可能用 tuple/list 形式返回
                if isinstance(out, (list, tuple)) and len(out) > 0:
                    logits = out[0]
                else:
                    raise RuntimeError("[HFRMBackend] Model forward has no 'logits' field.")

            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)  # [B] -> [B,1]

            if logits.size(-1) != 1:
                # 仅标量 RM；遇到多类头直接报错，避免误用
                raise RuntimeError(
                    f"[HFRMBackend] Expect scalar RM with logits[...,1], but got shape {list(logits.shape)}. "
                    f"Please load a regression RM (num_labels=1)."
                )

            raw = logits.squeeze(-1)  # [B]
            # 温度缩放
            if self.temperature != 1.0:
                raw = raw / max(1e-6, self.temperature)
            # Sigmoid（可关）
            scores = torch.sigmoid(raw) if self.apply_sigmoid else raw
            # 裁剪到期望区间
            scores = scores.clamp(self.clip_low, self.clip_high)

            normalized_scores.extend(scores.detach().float().cpu().tolist())

        return normalized_scores
