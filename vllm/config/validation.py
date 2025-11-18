# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration validation utilities.

This module provides comprehensive configuration validation with actionable
error messages to help users identify and fix configuration issues.
"""

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails.

    This exception includes actionable messages to help users fix the issue.
    """

    pass


class ConfigurationWarning:
    """Container for configuration warnings."""

    def __init__(self, message: str, suggestion: str | None = None):
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\nSuggestion: {self.suggestion}"
        return self.message


def validate_config(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    enable_prefix_caching: bool | None = None,
    enable_chunked_prefill: bool | None = None,
    enforce_eager: bool = False,
    quantization: str | None = None,
    compilation_config: Any = None,
    **kwargs: Any,
) -> list[ConfigurationWarning]:
    """Validate configuration settings and return warnings.

    This function checks for common configuration issues and incompatibilities,
    returning a list of warnings with actionable suggestions.

    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        pipeline_parallel_size: Number of stages for pipeline parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_model_len: Maximum sequence length.
        max_num_seqs: Maximum number of sequences to batch.
        enable_prefix_caching: Whether prefix caching is enabled.
        enable_chunked_prefill: Whether chunked prefill is enabled.
        enforce_eager: Whether to disable CUDA graphs.
        quantization: Quantization method.
        compilation_config: Compilation configuration.
        **kwargs: Additional configuration parameters.

    Returns:
        List of ConfigurationWarning objects.

    Raises:
        ConfigurationError: If there are critical configuration errors.

    Example:
        >>> warnings = validate_config(
        ...     tensor_parallel_size=2,
        ...     gpu_memory_utilization=0.99
        ... )
        >>> for w in warnings:
        ...     print(w)
    """
    errors: list[str] = []
    warnings: list[ConfigurationWarning] = []

    # Check GPU memory utilization
    if gpu_memory_utilization > 0.95:
        warnings.append(ConfigurationWarning(
            f"GPU memory utilization is very high ({gpu_memory_utilization}). "
            "This may cause out-of-memory errors.",
            "Consider reducing to 0.9 or lower for stability."
        ))

    if gpu_memory_utilization < 0.5:
        warnings.append(ConfigurationWarning(
            f"GPU memory utilization is low ({gpu_memory_utilization}). "
            "You may be underutilizing GPU memory.",
            "Consider increasing to 0.8-0.9 for better throughput."
        ))

    # Check parallelism settings
    total_parallelism = tensor_parallel_size * pipeline_parallel_size
    if total_parallelism > 8:
        warnings.append(ConfigurationWarning(
            f"Total parallelism is high ({total_parallelism} = {tensor_parallel_size}x{pipeline_parallel_size}). "
            "This may have diminishing returns.",
            "Consider if this level of parallelism is necessary for your model size."
        ))

    # Check compilation settings with parallelism
    if compilation_config is not None:
        comp_mode = getattr(compilation_config, 'mode', None)
        if comp_mode is not None:
            # CompilationMode.FULL == 2
            mode_value = comp_mode.value if hasattr(comp_mode, 'value') else comp_mode
            if mode_value == 2 and tensor_parallel_size > 1:
                errors.append(
                    "FULL compilation mode is not supported with tensor_parallel_size > 1. "
                    "Use PIECEWISE mode (level 3) instead, or reduce tensor_parallel_size to 1."
                )

    # Check prefix caching recommendations
    if enable_prefix_caching is False:
        warnings.append(ConfigurationWarning(
            "Prefix caching is disabled. This feature improves performance "
            "for prompts with shared prefixes at minimal cost.",
            "Consider enabling prefix caching with enable_prefix_caching=True."
        ))

    # Check chunked prefill with large sequences
    if max_model_len is not None and max_model_len > 8192:
        if enable_chunked_prefill is False:
            warnings.append(ConfigurationWarning(
                f"Chunked prefill is disabled with large max_model_len ({max_model_len}). "
                "This may cause high latency for the first token.",
                "Consider enabling chunked prefill with enable_chunked_prefill=True."
            ))

    # Check enforce_eager in production
    if enforce_eager and kwargs.get('profile') != 'development':
        warnings.append(ConfigurationWarning(
            "Eager execution is enabled. This disables CUDA graphs "
            "and may significantly reduce performance.",
            "Only use enforce_eager=True for debugging. "
            "Set to False for production workloads."
        ))

    # Check batch size settings
    if max_num_seqs is not None:
        if max_num_seqs < 8:
            warnings.append(ConfigurationWarning(
                f"max_num_seqs is very small ({max_num_seqs}). "
                "This limits batching efficiency.",
                "Consider increasing max_num_seqs for better throughput, "
                "unless low latency is critical."
            ))
        elif max_num_seqs > 512:
            warnings.append(ConfigurationWarning(
                f"max_num_seqs is very large ({max_num_seqs}). "
                "This may cause memory issues or scheduling overhead.",
                "Consider reducing to 256 or less unless you have specific requirements."
            ))

    # Check quantization recommendations
    if quantization is None and max_model_len is not None and max_model_len > 4096:
        warnings.append(ConfigurationWarning(
            "Running with long sequences without quantization may require "
            "significant GPU memory.",
            "Consider using quantization (e.g., quantization='awq' or 'gptq') "
            "to reduce memory usage."
        ))

    # Raise errors if any critical issues found
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise ConfigurationError(error_message)

    # Log warnings
    for warning in warnings:
        logger.warning("%s", warning)

    return warnings


def validate_profile_overrides(
    profile: str | None,
    overrides: dict[str, Any],
) -> list[ConfigurationWarning]:
    """Validate that profile overrides make sense.

    Some overrides may conflict with the profile's intended use case.

    Args:
        profile: The configuration profile being used.
        overrides: User-provided overrides to the profile.

    Returns:
        List of warnings about potentially conflicting overrides.
    """
    warnings: list[ConfigurationWarning] = []

    if profile is None:
        return warnings

    profile_lower = profile.lower() if isinstance(profile, str) else profile.value

    # Check for conflicting overrides
    if profile_lower == "high_throughput":
        if overrides.get("max_num_seqs") and overrides["max_num_seqs"] < 64:
            warnings.append(ConfigurationWarning(
                f"Using HIGH_THROUGHPUT profile but max_num_seqs is low ({overrides['max_num_seqs']}). "
                "This may limit throughput.",
                "Consider using LOW_LATENCY profile for small batch sizes, "
                "or increase max_num_seqs."
            ))
        if overrides.get("enforce_eager") is True:
            warnings.append(ConfigurationWarning(
                "Using HIGH_THROUGHPUT profile with enforce_eager=True. "
                "This contradicts the profile's optimization goal.",
                "Remove enforce_eager override or use DEVELOPMENT profile."
            ))

    elif profile_lower == "low_latency":
        if overrides.get("max_num_seqs") and overrides["max_num_seqs"] > 128:
            warnings.append(ConfigurationWarning(
                f"Using LOW_LATENCY profile but max_num_seqs is high ({overrides['max_num_seqs']}). "
                "Large batches may increase latency.",
                "Consider using HIGH_THROUGHPUT profile for large batch sizes, "
                "or reduce max_num_seqs."
            ))

    elif profile_lower == "memory_constrained":
        if overrides.get("gpu_memory_utilization") and overrides["gpu_memory_utilization"] > 0.9:
            warnings.append(ConfigurationWarning(
                f"Using MEMORY_CONSTRAINED profile but gpu_memory_utilization is high "
                f"({overrides['gpu_memory_utilization']}). "
                "This may cause OOM errors.",
                "Consider reducing gpu_memory_utilization or using a different profile."
            ))

    elif profile_lower == "development":
        if overrides.get("enforce_eager") is False:
            warnings.append(ConfigurationWarning(
                "Using DEVELOPMENT profile with enforce_eager=False. "
                "CUDA graphs may make debugging harder.",
                "Keep enforce_eager=True for easier debugging, "
                "or use a different profile for production."
            ))

    return warnings


def check_deprecated_env_vars() -> list[str]:
    """Check for deprecated environment variables and return warnings.

    Returns:
        List of warning messages for deprecated env vars that are set.
    """
    import os

    warnings = []

    # Map of deprecated env vars to their replacements/alternatives
    deprecated_vars = {
        "VLLM_USE_V1": (
            "VLLM_USE_V1 is deprecated. "
            "V1 engine is now the default. Use VLLM_USE_V0=1 to use V0 engine."
        ),
        "VLLM_NCCL_SO_PATH": (
            "VLLM_NCCL_SO_PATH is deprecated. "
            "vLLM now automatically locates the NCCL library."
        ),
    }

    for var, message in deprecated_vars.items():
        if os.environ.get(var):
            warnings.append(message)
            logger.warning("Deprecated environment variable: %s", message)

    return warnings
