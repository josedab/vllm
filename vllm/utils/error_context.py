# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for capturing error context in vLLM."""

from typing import Any, Optional

import vllm


class ErrorContext:
    """Captures context for error reporting.

    Provides methods to gather system information, configuration details,
    and runtime state that can be included in error messages to help
    users diagnose and resolve issues.
    """

    @staticmethod
    def get_system_context() -> dict[str, Any]:
        """Get system information for error context.

        Returns:
            Dictionary containing system information including vLLM version,
            PyTorch version, CUDA version, and GPU details.
        """
        import torch

        context: dict[str, Any] = {
            "vLLM version": vllm.__version__,
            "PyTorch version": torch.__version__,
        }

        # Add CUDA information if available
        if torch.cuda.is_available():
            context["CUDA version"] = torch.version.cuda
            context["GPU count"] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                context["GPU"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                context["GPU memory"] = f"{props.total_memory / 1e9:.1f}GB"
        else:
            context["CUDA available"] = False

        return context

    @staticmethod
    def get_memory_context(device_id: int = 0) -> dict[str, Any]:
        """Get GPU memory information for error context.

        Args:
            device_id: The GPU device ID to query.

        Returns:
            Dictionary containing memory usage information.
        """
        import torch

        if not torch.cuda.is_available():
            return {"CUDA available": False}

        if device_id >= torch.cuda.device_count():
            return {"error": f"Invalid device ID: {device_id}"}

        try:
            props = torch.cuda.get_device_properties(device_id)
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)

            return {
                "GPU": torch.cuda.get_device_name(device_id),
                "Total memory": f"{props.total_memory / 1e9:.2f}GB",
                "Allocated": f"{allocated / 1e9:.2f}GB",
                "Reserved": f"{reserved / 1e9:.2f}GB",
                "Free (approx)": f"{(props.total_memory - reserved) / 1e9:.2f}GB",
            }
        except Exception as e:
            return {"error": f"Failed to get memory info: {e}"}

    @staticmethod
    def get_model_config_context(model_config: Any) -> dict[str, Any]:
        """Get model configuration for error context.

        Args:
            model_config: The ModelConfig object.

        Returns:
            Dictionary containing relevant model configuration.
        """
        context: dict[str, Any] = {}

        # Safely extract common attributes
        attrs = [
            ("model", "Model"),
            ("tokenizer", "Tokenizer"),
            ("dtype", "Dtype"),
            ("max_model_len", "Max model length"),
            ("quantization", "Quantization"),
        ]

        for attr_name, display_name in attrs:
            if hasattr(model_config, attr_name):
                value = getattr(model_config, attr_name)
                if value is not None:
                    context[display_name] = str(value)

        return context

    @staticmethod
    def get_parallel_config_context(parallel_config: Any) -> dict[str, Any]:
        """Get parallel configuration for error context.

        Args:
            parallel_config: The ParallelConfig object.

        Returns:
            Dictionary containing relevant parallel configuration.
        """
        context: dict[str, Any] = {}

        attrs = [
            ("tensor_parallel_size", "Tensor parallel size"),
            ("pipeline_parallel_size", "Pipeline parallel size"),
            ("data_parallel_size", "Data parallel size"),
        ]

        for attr_name, display_name in attrs:
            if hasattr(parallel_config, attr_name):
                value = getattr(parallel_config, attr_name)
                if value is not None:
                    context[display_name] = value

        return context

    @staticmethod
    def get_scheduler_config_context(scheduler_config: Any) -> dict[str, Any]:
        """Get scheduler configuration for error context.

        Args:
            scheduler_config: The SchedulerConfig object.

        Returns:
            Dictionary containing relevant scheduler configuration.
        """
        context: dict[str, Any] = {}

        attrs = [
            ("max_num_seqs", "Max sequences"),
            ("max_num_batched_tokens", "Max batched tokens"),
        ]

        for attr_name, display_name in attrs:
            if hasattr(scheduler_config, attr_name):
                value = getattr(scheduler_config, attr_name)
                if value is not None:
                    context[display_name] = value

        return context

    @staticmethod
    def get_cache_config_context(cache_config: Any) -> dict[str, Any]:
        """Get cache configuration for error context.

        Args:
            cache_config: The CacheConfig object.

        Returns:
            Dictionary containing relevant cache configuration.
        """
        context: dict[str, Any] = {}

        attrs = [
            ("block_size", "Block size"),
            ("gpu_memory_utilization", "GPU memory utilization"),
            ("cache_dtype", "Cache dtype"),
            ("num_gpu_blocks", "GPU blocks"),
        ]

        for attr_name, display_name in attrs:
            if hasattr(cache_config, attr_name):
                value = getattr(cache_config, attr_name)
                if value is not None:
                    if attr_name == "gpu_memory_utilization":
                        context[display_name] = f"{value:.0%}"
                    else:
                        context[display_name] = value

        return context

    @staticmethod
    def get_full_config_context(
        model_config: Optional[Any] = None,
        parallel_config: Optional[Any] = None,
        scheduler_config: Optional[Any] = None,
        cache_config: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Get combined configuration context for error reporting.

        Args:
            model_config: Optional ModelConfig object.
            parallel_config: Optional ParallelConfig object.
            scheduler_config: Optional SchedulerConfig object.
            cache_config: Optional CacheConfig object.

        Returns:
            Dictionary containing all available configuration context.
        """
        context: dict[str, Any] = {}

        if model_config is not None:
            context.update(ErrorContext.get_model_config_context(model_config))

        if parallel_config is not None:
            context.update(ErrorContext.get_parallel_config_context(parallel_config))

        if scheduler_config is not None:
            context.update(ErrorContext.get_scheduler_config_context(scheduler_config))

        if cache_config is not None:
            context.update(ErrorContext.get_cache_config_context(cache_config))

        return context


__all__ = ["ErrorContext"]
