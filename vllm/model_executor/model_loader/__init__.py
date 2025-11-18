# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.exceptions import VLLMModelLoadError
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.utils.error_context import ErrorContext
from vllm.utils.known_issues import enhance_error
from vllm.model_executor.model_loader.bitsandbytes_loader import BitsAndBytesModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader
from vllm.model_executor.model_loader.runai_streamer_loader import (
    RunaiModelStreamerLoader,
)
from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader
from vllm.model_executor.model_loader.tensorizer_loader import TensorizerLoader
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name,
    get_model_architecture,
    get_model_cls,
)

logger = init_logger(__name__)

# Reminder: Please update docstring in `LoadConfig`
# if a new load format is added here
LoadFormats = Literal[
    "auto",
    "bitsandbytes",
    "dummy",
    "fastsafetensors",
    "gguf",
    "mistral",
    "npcache",
    "pt",
    "runai_streamer",
    "runai_streamer_sharded",
    "safetensors",
    "sharded_state",
    "tensorizer",
]
_LOAD_FORMAT_TO_MODEL_LOADER: dict[str, type[BaseModelLoader]] = {
    "auto": DefaultModelLoader,
    "bitsandbytes": BitsAndBytesModelLoader,
    "dummy": DummyModelLoader,
    "fastsafetensors": DefaultModelLoader,
    "gguf": GGUFModelLoader,
    "mistral": DefaultModelLoader,
    "npcache": DefaultModelLoader,
    "pt": DefaultModelLoader,
    "runai_streamer": RunaiModelStreamerLoader,
    "runai_streamer_sharded": ShardedStateLoader,
    "safetensors": DefaultModelLoader,
    "sharded_state": ShardedStateLoader,
    "tensorizer": TensorizerLoader,
}


def register_model_loader(load_format: str):
    """Register a customized vllm model loader.

    When a load format is not supported by vllm, you can register a customized
    model loader to support it.

    Args:
        load_format (str): The model loader format name.

    Examples:
        >>> from vllm.config.load import LoadConfig
        >>> from vllm.model_executor.model_loader import (
        ...     get_model_loader,
        ...     register_model_loader,
        ... )
        >>> from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        >>>
        >>> @register_model_loader("my_loader")
        ... class MyModelLoader(BaseModelLoader):
        ...     def download_model(self):
        ...         pass
        ...
        ...     def load_weights(self):
        ...         pass
        >>>
        >>> load_config = LoadConfig(load_format="my_loader")
        >>> type(get_model_loader(load_config))
        <class 'MyModelLoader'>
    """  # noqa: E501

    def _wrapper(model_loader_cls):
        if load_format in _LOAD_FORMAT_TO_MODEL_LOADER:
            logger.warning(
                "Load format `%s` is already registered, and will be "
                "overwritten by the new loader class `%s`.",
                load_format,
                model_loader_cls,
            )
        if not issubclass(model_loader_cls, BaseModelLoader):
            raise ValueError(
                "The model loader must be a subclass of `BaseModelLoader`."
            )
        _LOAD_FORMAT_TO_MODEL_LOADER[load_format] = model_loader_cls
        logger.info(
            "Registered model loader `%s` with load format `%s`",
            model_loader_cls,
            load_format,
        )
        return model_loader_cls

    return _wrapper


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    load_format = load_config.load_format
    if load_format not in _LOAD_FORMAT_TO_MODEL_LOADER:
        raise ValueError(f"Load format `{load_format}` is not supported")
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)


def get_model(
    *, vllm_config: VllmConfig, model_config: ModelConfig | None = None
) -> nn.Module:
    loader = get_model_loader(vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config

    try:
        return loader.load_model(vllm_config=vllm_config, model_config=model_config)
    except Exception as e:
        # Check if this is a known error pattern that we can enhance
        error_str = str(e).lower()

        # Model not found or authentication errors
        if any(
            pattern in error_str
            for pattern in [
                "not found",
                "404",
                "does not appear to have",
                "401",
                "unauthorized",
                "access denied",
                "permission denied",
                "gated repo",
            ]
        ):
            context = {
                "Model": model_config.model,
            }
            context.update(ErrorContext.get_system_context())
            if hasattr(model_config, "tokenizer") and model_config.tokenizer:
                context["Tokenizer"] = model_config.tokenizer
            if hasattr(model_config, "revision") and model_config.revision:
                context["Revision"] = model_config.revision

            # Determine if it's auth or not found
            if any(
                pattern in error_str
                for pattern in ["401", "unauthorized", "access denied", "gated"]
            ):
                raise VLLMModelLoadError(
                    f"Authentication error loading model '{model_config.model}'",
                    context=context,
                    solutions=[
                        "Login to HuggingFace: huggingface-cli login",
                        "Set HF_TOKEN environment variable with your token",
                        "Verify you have access to the model repository",
                        "For gated models, accept the license agreement on HuggingFace",
                    ],
                    docs_url="https://docs.vllm.ai/en/latest/models/supported_models.html",
                ) from e
            else:
                raise VLLMModelLoadError(
                    f"Model '{model_config.model}' not found",
                    context=context,
                    solutions=[
                        "Check the model name/path is correct",
                        "Ensure you're logged in: huggingface-cli login",
                        "For private models, set HF_TOKEN environment variable",
                        "Check network connectivity to HuggingFace Hub",
                        "Verify the model exists at the specified location",
                    ],
                    docs_url="https://docs.vllm.ai/en/latest/models/supported_models.html",
                ) from e

        # Re-raise other errors with enhancement attempt
        enhanced = enhance_error(e)
        if enhanced is not e:
            raise enhanced from e
        raise


__all__ = [
    "get_model",
    "get_model_loader",
    "get_architecture_class_name",
    "get_model_architecture",
    "get_model_cls",
    "register_model_loader",
    "BaseModelLoader",
    "BitsAndBytesModelLoader",
    "GGUFModelLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "RunaiModelStreamerLoader",
    "ShardedStateLoader",
    "TensorizerLoader",
]
