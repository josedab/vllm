# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM engine module with V0 deprecation support.

This module provides engine creation utilities with deprecation warnings
for the legacy V0 engine. V1 is the default and recommended engine.
"""

import os
import warnings
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine

logger = init_logger(__name__)

# V0 Engine Deprecation Constants
V0_DEPRECATION_VERSION = "v1.0.0"
V0_MIGRATION_GUIDE_URL = "https://docs.vllm.ai/en/latest/migration/v0-to-v1.html"
V0_REMOVAL_URL = "https://docs.vllm.ai/en/latest/migration/v0-removal.html"


def create_engine(
    engine_args: "EngineArgs",
    usage_context: "UsageContext | None" = None,
    use_v1: bool | None = None,
    **kwargs,
) -> "LLMEngine":
    """Create an LLM engine with deprecation handling for V0.

    This function creates an LLMEngine instance. The V0 engine is deprecated
    and will be removed in v1.0.0. All users should migrate to V1.

    Args:
        engine_args: Engine configuration arguments.
        usage_context: Usage context for telemetry.
        use_v1: If explicitly set to False, will show deprecation warning.
                V1 is always used as V0 has been removed.
        **kwargs: Additional arguments passed to the engine.

    Returns:
        LLMEngine: The created engine instance.

    Raises:
        RuntimeError: If use_v1 is explicitly False (V0 no longer available).
    """
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine

    if usage_context is None:
        usage_context = UsageContext.ENGINE_CONTEXT

    # Check environment variable for V0 request
    env_use_v1 = os.getenv("VLLM_USE_V1", "1")

    # Determine if user is trying to use V0
    if use_v1 is False or env_use_v1 == "0":
        # V0 has been removed - show error with migration guidance
        raise RuntimeError(
            f"V0 engine has been removed as of {V0_DEPRECATION_VERSION}. "
            f"Please use V1 engine. "
            f"See {V0_REMOVAL_URL} for migration guidance."
        )

    # V1 is always used
    return LLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=usage_context,
        **kwargs
    )


def warn_v0_feature_usage(feature_name: str, stacklevel: int = 3) -> None:
    """Emit a deprecation warning for V0-only feature usage.

    Args:
        feature_name: Name of the V0-only feature being used.
        stacklevel: Stack level for the warning.
    """
    warnings.warn(
        f"'{feature_name}' is a V0-only feature and is deprecated. "
        f"It will be removed in {V0_DEPRECATION_VERSION}. "
        f"See {V0_MIGRATION_GUIDE_URL} for migration guidance.",
        DeprecationWarning,
        stacklevel=stacklevel
    )


def warn_v0_parameter_usage(
    param_name: str,
    alternative: str | None = None,
    stacklevel: int = 3
) -> None:
    """Emit a deprecation warning for V0-only parameter usage.

    Args:
        param_name: Name of the deprecated parameter.
        alternative: Alternative parameter or approach to use.
        stacklevel: Stack level for the warning.
    """
    msg = (
        f"Parameter '{param_name}' is V0-only and deprecated. "
        f"It will be removed in {V0_DEPRECATION_VERSION}."
    )

    if alternative:
        msg += f" Please use '{alternative}' instead."

    msg += f" See {V0_MIGRATION_GUIDE_URL} for details."

    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)


# Feature parity tracking for V0 -> V1 migration
V0_FEATURES = {
    "continuous_batching": True,  # Supported in V1
    "prefix_caching": True,  # Supported in V1
    "speculative_decoding": True,  # Supported in V1
    "lora": True,  # Supported in V1
    "multimodal": True,  # Supported in V1
    "best_of": False,  # V0-only, not in V1
    "swap_space": True,  # Supported but different implementation
}


def check_v1_feature_support(feature: str) -> bool:
    """Check if a feature is supported in V1.

    Args:
        feature: Feature name to check.

    Returns:
        bool: True if feature is supported in V1.
    """
    return V0_FEATURES.get(feature, False)


def get_unsupported_v1_features() -> list[str]:
    """Get list of V0 features not supported in V1.

    Returns:
        list[str]: List of unsupported feature names.
    """
    return [f for f, supported in V0_FEATURES.items() if not supported]


__all__ = [
    "create_engine",
    "warn_v0_feature_usage",
    "warn_v0_parameter_usage",
    "check_v1_feature_support",
    "get_unsupported_v1_features",
    "V0_DEPRECATION_VERSION",
    "V0_MIGRATION_GUIDE_URL",
    "V0_REMOVAL_URL",
    "V0_FEATURES",
]
