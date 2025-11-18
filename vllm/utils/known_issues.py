# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Database of known issues and their solutions for enhanced error messages."""

import re
from typing import Optional

from vllm.exceptions import (
    VLLMConfigurationError,
    VLLMDistributedError,
    VLLMError,
    VLLMMemoryError,
    VLLMModelLoadError,
)

# Base URL for vLLM documentation
DOCS_BASE_URL = "https://docs.vllm.ai/en/latest"

# Known issues database with patterns and solutions
KNOWN_ISSUES: dict[str, dict] = {
    "CUDA_OOM": {
        "pattern": r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError",
        "message": "GPU memory exhausted during execution",
        "exception_class": VLLMMemoryError,
        "solutions": [
            "Reduce gpu_memory_utilization (e.g., --gpu-memory-utilization 0.8)",
            "Reduce max_num_seqs to limit concurrent sequences",
            "Reduce max_model_len to decrease KV cache requirements",
            "Enable quantization (e.g., --quantization fp8)",
            "Enable KV cache quantization (e.g., --kv-cache-dtype fp8_e4m3)",
            "Use tensor parallelism across multiple GPUs",
        ],
        "docs": "troubleshooting/memory.html",
    },
    "NCCL_TIMEOUT": {
        "pattern": r"NCCL.*timeout|ncclTimeout|NCCL error",
        "message": "Distributed communication timeout",
        "exception_class": VLLMDistributedError,
        "solutions": [
            "Increase NCCL timeout: export NCCL_TIMEOUT=1800",
            "Check network connectivity between nodes",
            "Ensure all GPUs are accessible and functional",
            "Verify CUDA versions match across nodes",
            "Check firewall settings for NCCL ports",
        ],
        "docs": "serving/distributed_serving.html",
    },
    "MODEL_NOT_FOUND": {
        "pattern": r"(model|repository).*not found|404.*not found|does not appear to have",
        "message": "Model or repository not found",
        "exception_class": VLLMModelLoadError,
        "solutions": [
            "Verify the model name/path is correct",
            "For HuggingFace models, ensure you're logged in: huggingface-cli login",
            "For private models, set HF_TOKEN environment variable",
            "Check network connectivity to HuggingFace Hub",
            "Verify the model exists at the specified location",
        ],
        "docs": "models/supported_models.html",
    },
    "AUTH_ERROR": {
        "pattern": r"401|unauthorized|authentication|access denied|permission denied",
        "message": "Authentication or permission error",
        "exception_class": VLLMModelLoadError,
        "solutions": [
            "Login to HuggingFace: huggingface-cli login",
            "Set HF_TOKEN environment variable with your token",
            "Verify you have access to the model repository",
            "For gated models, accept the license agreement on HuggingFace",
        ],
        "docs": "models/loading_models.html",
    },
    "TP_SIZE_MISMATCH": {
        "pattern": r"tensor.parallel.*exceed|not.*divisible|invalid.*tensor.*parallel",
        "message": "Tensor parallel configuration error",
        "exception_class": VLLMConfigurationError,
        "solutions": [
            "Ensure tensor_parallel_size <= number of available GPUs",
            "Model dimensions must be divisible by tensor_parallel_size",
            "Try tensor_parallel_size that divides the model's hidden dimension",
        ],
        "docs": "serving/distributed_serving.html",
    },
    "CONTEXT_LENGTH": {
        "pattern": r"context.length|max.*position|exceeds.*maximum|sequence.*too long",
        "message": "Context length exceeded",
        "exception_class": VLLMConfigurationError,
        "solutions": [
            "Reduce max_model_len to fit within model's supported context",
            "Use a model with longer context support",
            "Enable rope_scaling for extended context (if supported)",
            "Truncate input to fit within model limits",
        ],
        "docs": "models/engine_args.html",
    },
    "DTYPE_MISMATCH": {
        "pattern": r"dtype.*mismatch|expected.*got|incompatible.*dtype",
        "message": "Data type mismatch error",
        "exception_class": VLLMConfigurationError,
        "solutions": [
            "Ensure consistent dtype across model and inputs",
            "Use --dtype auto to let vLLM select appropriate dtype",
            "Check model supports the specified dtype",
        ],
        "docs": "models/engine_args.html",
    },
    "TOKENIZER_ERROR": {
        "pattern": r"tokenizer.*error|tokenizer.*not found|failed.*tokenizer",
        "message": "Tokenizer loading error",
        "exception_class": VLLMModelLoadError,
        "solutions": [
            "Verify tokenizer files exist in model directory",
            "Try specifying tokenizer explicitly with --tokenizer",
            "For custom tokenizers, ensure correct format (sentencepiece, tiktoken)",
            "Check for tokenizer.json or tokenizer.model files",
        ],
        "docs": "models/loading_models.html",
    },
    "QUANTIZATION_ERROR": {
        "pattern": r"quantization.*error|quantiz.*not.*support|bitsandbytes|awq|gptq",
        "message": "Quantization error",
        "exception_class": VLLMConfigurationError,
        "solutions": [
            "Ensure quantization method is supported for your model",
            "Check required packages are installed (e.g., bitsandbytes, auto-gptq)",
            "Verify model was quantized correctly",
            "Try a different quantization method",
        ],
        "docs": "quantization/supported_hardware.html",
    },
    "PORT_IN_USE": {
        "pattern": r"address.*in use|port.*in use|bind.*failed",
        "message": "Port already in use",
        "exception_class": VLLMConfigurationError,
        "solutions": [
            "Use a different port with --port",
            "Kill the process using the port",
            "Check for other vLLM instances running",
        ],
        "docs": "serving/openai_compatible_server.html",
    },
}


def enhance_error(error: Exception) -> VLLMError:
    """Enhance an error with known issue information.

    Matches the error message against known issue patterns and returns
    a VLLMError with helpful solutions and documentation links.

    Args:
        error: The original exception.

    Returns:
        A VLLMError with enhanced information, or a generic VLLMError
        wrapping the original if no pattern matches.
    """
    error_str = str(error)

    for issue_id, issue in KNOWN_ISSUES.items():
        if re.search(issue["pattern"], error_str, re.IGNORECASE):
            exception_class = issue.get("exception_class", VLLMError)
            docs_url = f"{DOCS_BASE_URL}/{issue['docs']}" if issue.get("docs") else None

            return exception_class(
                issue["message"],
                context={"Original error": str(error)[:200]},  # Truncate long errors
                solutions=issue["solutions"],
                docs_url=docs_url,
            )

    # No pattern matched, return a generic enhanced error
    return VLLMError(
        str(error),
        context={"Error type": type(error).__name__},
    )


def get_issue_info(issue_id: str) -> Optional[dict]:
    """Get information about a known issue by ID.

    Args:
        issue_id: The issue identifier (e.g., "CUDA_OOM").

    Returns:
        Dictionary with issue information, or None if not found.
    """
    return KNOWN_ISSUES.get(issue_id)


def list_known_issues() -> list[str]:
    """Get a list of all known issue IDs.

    Returns:
        List of issue identifiers.
    """
    return list(KNOWN_ISSUES.keys())


__all__ = [
    "KNOWN_ISSUES",
    "DOCS_BASE_URL",
    "enhance_error",
    "get_issue_info",
    "list_known_issues",
]
