from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
import transformers
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from packaging.version import parse as vparse

if TYPE_CHECKING:
    from transformers.quantizers.auto import AutoQuantizationConfig

import sys

from .huggingface import HFLM

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import training.models.llama_monkey_patch

eval_logger = logging.getLogger(__name__)


@register_model("custom")
class CustomLM(HFLM):
    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: str | None = None,
        subfolder: str = "",
    ) -> None:
        """Return the model config for HuggingFace models."""
        self._config = transformers.LlamaConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            gguf_file=gguf_file,
            subfolder=subfolder,
        )

    def _create_model(
        self,
        pretrained: str,
        revision: str | None = "main",
        dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool | None = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: bool | None = False,
        gpus: int | None = None,
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        # PEFT, delta weights and quantization options
        peft: str | None = None,
        delta: str | None = None,
        autogptq: bool | str | None = False,
        gptqmodel: bool | None = False,
        gguf_file: str | None = None,
        quantization_config: AutoQuantizationConfig | None = None,
        subfolder: str = "",
        **kwargs,
    ) -> None:
        """Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """

        model_kwargs = kwargs or {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map"),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )
        if autogptq or gptqmodel or peft or delta:
            raise NotImplementedError()
        if model_kwargs.get("load_in_4bit"):
            assert vparse(transformers.__version__) >= vparse(
                "4.30.0"
            ), "load_in_4bit requires transformers >= 4.30.0"
            if compute_dtype := model_kwargs.get("bnb_4bit_compute_dtype"):
                model_kwargs["bnb_4bit_compute_dtype"] = get_dtype(compute_dtype)

        self._model = transformers.LlamaForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=get_dtype(dtype),
            trust_remote_code=trust_remote_code,
            gguf_file=gguf_file,
            quantization_config=quantization_config,
            subfolder=subfolder,
            **model_kwargs,
        )
