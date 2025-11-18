# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration profiles for common use cases.

This module provides predefined configuration profiles that bundle together
optimal settings for common use cases like high throughput, low latency,
or memory-constrained environments.

Example:
    >>> from vllm import LLM
    >>> from vllm.config.profiles import ConfigProfile
    >>>
    >>> # Simple: Use a profile
    >>> llm = LLM(
    ...     model="meta-llama/Llama-2-7b-hf",
    ...     profile=ConfigProfile.HIGH_THROUGHPUT
    ... )
    >>>
    >>> # Advanced: Override specific settings
    >>> llm = LLM(
    ...     model="meta-llama/Llama-2-7b-hf",
    ...     profile=ConfigProfile.HIGH_THROUGHPUT,
    ...     max_model_len=8192  # Override just this
    ... )
"""

from enum import Enum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class ConfigProfile(str, Enum):
    """Predefined configuration profiles for common use cases.

    Attributes:
        HIGH_THROUGHPUT: Optimized for maximum tokens/second with larger batches.
            Best for batch processing and high-volume production workloads.
        LOW_LATENCY: Optimized for minimum time-to-first-token and fast responses.
            Best for interactive applications and real-time chat.
        MEMORY_CONSTRAINED: Optimized to minimize GPU memory usage.
            Best for running large models on limited hardware.
        DEVELOPMENT: Balanced settings with verbose logging for debugging.
            Best for development, testing, and experimentation.
    """

    HIGH_THROUGHPUT = "high_throughput"
    LOW_LATENCY = "low_latency"
    MEMORY_CONSTRAINED = "memory_constrained"
    DEVELOPMENT = "development"


# Profile definitions with their configuration settings
# These settings are applied as defaults and can be overridden by user
PROFILE_SETTINGS: dict[ConfigProfile, dict[str, Any]] = {
    ConfigProfile.HIGH_THROUGHPUT: {
        # Maximize batch size and memory usage for throughput
        "max_num_seqs": 256,
        "gpu_memory_utilization": 0.95,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        # Note: compilation_config will be set as a dict to be processed later
        "_compilation_level": 3,  # PIECEWISE optimization
    },
    ConfigProfile.LOW_LATENCY: {
        # Smaller batches, optimizations for fast first token
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "_compilation_level": 3,  # PIECEWISE for balance
    },
    ConfigProfile.MEMORY_CONSTRAINED: {
        # Conservative memory settings
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.8,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "swap_space": 8,  # More swap space to offload
        "cpu_offload_gb": 0,  # Start without CPU offload, user can enable
        "_compilation_level": 0,  # No compilation to save memory
    },
    ConfigProfile.DEVELOPMENT: {
        # Balanced settings for development with helpful defaults
        "max_num_seqs": 128,
        "gpu_memory_utilization": 0.85,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,  # Simpler behavior for debugging
        "enforce_eager": True,  # Easier to debug without CUDA graphs
        "_compilation_level": 0,  # No compilation for faster startup
    },
}


def get_profile_settings(profile: ConfigProfile | str | None) -> dict[str, Any]:
    """Get the settings dictionary for a given profile.

    Args:
        profile: The configuration profile to retrieve settings for.
            Can be a ConfigProfile enum or string name.

    Returns:
        Dictionary of configuration settings for the profile.
        Returns empty dict if profile is None.

    Raises:
        ValueError: If the profile name is not recognized.

    Example:
        >>> settings = get_profile_settings(ConfigProfile.HIGH_THROUGHPUT)
        >>> print(settings["max_num_seqs"])
        256
    """
    if profile is None:
        return {}

    # Convert string to enum if needed
    if isinstance(profile, str):
        try:
            profile = ConfigProfile(profile)
        except ValueError:
            valid_profiles = [p.value for p in ConfigProfile]
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Valid profiles are: {valid_profiles}"
            ) from None

    if profile not in PROFILE_SETTINGS:
        raise ValueError(f"Profile {profile} is not defined")

    settings = PROFILE_SETTINGS[profile].copy()
    logger.info("Using configuration profile: %s", profile.value)

    return settings


def apply_profile_settings(
    profile: ConfigProfile | str | None,
    user_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Apply profile settings with user overrides.

    Profile settings serve as defaults that can be overridden by explicit
    user-provided values. This allows users to start with a profile and
    customize specific settings.

    Args:
        profile: The configuration profile to use as base settings.
        user_overrides: Dictionary of user-specified settings that will
            override profile defaults.

    Returns:
        Merged dictionary with profile settings as base and user overrides
        applied on top.

    Example:
        >>> settings = apply_profile_settings(
        ...     ConfigProfile.HIGH_THROUGHPUT,
        ...     {"max_model_len": 8192}  # Override just this
        ... )
    """
    # Get base settings from profile
    settings = get_profile_settings(profile)

    # User overrides take precedence
    # Only override with non-None values
    for key, value in user_overrides.items():
        if value is not None:
            settings[key] = value

    return settings


def get_profile_description(profile: ConfigProfile | str) -> str:
    """Get a human-readable description of a profile.

    Args:
        profile: The profile to describe.

    Returns:
        A description of the profile's intended use case and characteristics.
    """
    if isinstance(profile, str):
        profile = ConfigProfile(profile)

    descriptions = {
        ConfigProfile.HIGH_THROUGHPUT: (
            "Optimized for maximum tokens/second with larger batches. "
            "Best for batch processing and high-volume production workloads. "
            "Uses high GPU memory utilization (95%), larger batch sizes (256 sequences), "
            "and enables prefix caching and chunked prefill."
        ),
        ConfigProfile.LOW_LATENCY: (
            "Optimized for minimum time-to-first-token and fast responses. "
            "Best for interactive applications and real-time chat. "
            "Uses smaller batch sizes (32 sequences) for faster individual response times, "
            "while still enabling optimizations like prefix caching."
        ),
        ConfigProfile.MEMORY_CONSTRAINED: (
            "Optimized to minimize GPU memory usage. "
            "Best for running large models on limited hardware. "
            "Uses conservative GPU memory utilization (80%), moderate batch sizes (64), "
            "and increased swap space. Disables compilation to save memory."
        ),
        ConfigProfile.DEVELOPMENT: (
            "Balanced settings with verbose logging for debugging. "
            "Best for development, testing, and experimentation. "
            "Uses eager execution for easier debugging, moderate settings, "
            "and disables compilation for faster startup times."
        ),
    }

    return descriptions.get(profile, f"No description available for {profile}")


def list_profiles() -> list[dict[str, Any]]:
    """List all available profiles with their descriptions and settings.

    Returns:
        List of dictionaries containing profile information.

    Example:
        >>> profiles = list_profiles()
        >>> for p in profiles:
        ...     print(f"{p['name']}: {p['description'][:50]}...")
    """
    result = []
    for profile in ConfigProfile:
        result.append({
            "name": profile.value,
            "enum": profile,
            "description": get_profile_description(profile),
            "settings": PROFILE_SETTINGS[profile].copy(),
        })
    return result
