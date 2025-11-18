# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enhanced exception hierarchy for vLLM with actionable error messages."""

from typing import Any, Optional


class VLLMError(Exception):
    """Base class for vLLM exceptions with enhanced formatting.

    Provides structured error messages with context, solutions, and
    documentation links to help users quickly diagnose and resolve issues.
    """

    def __init__(
        self,
        message: str,
        *,
        context: Optional[dict[str, Any]] = None,
        solutions: Optional[list[str]] = None,
        docs_url: Optional[str] = None,
    ):
        """Initialize a VLLMError with enhanced information.

        Args:
            message: The main error message.
            context: Dictionary of relevant context information.
            solutions: List of suggested solutions.
            docs_url: URL to relevant documentation.
        """
        self.message = message
        self.context = context or {}
        self.solutions = solutions or []
        self.docs_url = docs_url
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format the error message with context, solutions, and docs link."""
        parts = [self.message]

        if self.context:
            parts.append("\nContext:")
            for key, value in self.context.items():
                parts.append(f"  {key}: {value}")

        if self.solutions:
            parts.append("\nSolutions:")
            for i, solution in enumerate(self.solutions, 1):
                parts.append(f"  {i}. {solution}")

        if self.docs_url:
            parts.append(f"\nDocumentation: {self.docs_url}")

        return "\n".join(parts)


class VLLMMemoryError(VLLMError):
    """GPU memory related errors.

    Raised when there is insufficient GPU memory for model loading,
    KV cache allocation, or other memory-intensive operations.
    """
    pass


class VLLMConfigurationError(VLLMError):
    """Configuration validation errors.

    Raised when configuration parameters are invalid, incompatible,
    or cannot be satisfied by the available hardware.
    """
    pass


class VLLMModelLoadError(VLLMError):
    """Model loading errors.

    Raised when a model cannot be loaded due to missing files,
    authentication issues, or incompatible configurations.
    """
    pass


class VLLMDistributedError(VLLMError):
    """Distributed computing errors.

    Raised when there are issues with distributed setup, communication
    timeouts, or multi-GPU/multi-node configurations.
    """
    pass


class VLLMTokenizerError(VLLMError):
    """Tokenizer related errors.

    Raised when tokenizer loading or configuration fails.
    """
    pass


class VLLMQuantizationError(VLLMError):
    """Quantization related errors.

    Raised when quantization configuration is invalid or
    the requested quantization method is not supported.
    """
    pass


# Convenience aliases for backwards compatibility and shorter names
MemoryError = VLLMMemoryError  # Note: shadows builtin, use VLLMMemoryError preferred
ConfigurationError = VLLMConfigurationError
ModelLoadError = VLLMModelLoadError
DistributedError = VLLMDistributedError
TokenizerError = VLLMTokenizerError
QuantizationError = VLLMQuantizationError


__all__ = [
    "VLLMError",
    "VLLMMemoryError",
    "VLLMConfigurationError",
    "VLLMModelLoadError",
    "VLLMDistributedError",
    "VLLMTokenizerError",
    "VLLMQuantizationError",
    # Aliases
    "ConfigurationError",
    "ModelLoadError",
    "DistributedError",
    "TokenizerError",
    "QuantizationError",
]
