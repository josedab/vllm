# SPDX-License-Identifier: Apache-2.0
"""Memory estimation for auto-tuning."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MemoryEstimate:
    """Memory estimation breakdown."""

    weights: int  # Model weights in bytes
    kv_cache: int  # KV cache in bytes
    activations: int  # Activation memory in bytes
    overhead: int  # System overhead in bytes
    total: int  # Total memory in bytes

    @property
    def weights_gb(self) -> float:
        """Weights memory in GB."""
        return self.weights / (1024**3)

    @property
    def kv_cache_gb(self) -> float:
        """KV cache memory in GB."""
        return self.kv_cache / (1024**3)

    @property
    def activations_gb(self) -> float:
        """Activations memory in GB."""
        return self.activations / (1024**3)

    @property
    def total_gb(self) -> float:
        """Total memory in GB."""
        return self.total / (1024**3)


@dataclass
class ModelConfig:
    """Model configuration for memory estimation."""

    num_parameters: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    dtype: str = "bfloat16"
    max_position_embeddings: int = 4096

    @classmethod
    def from_hf_config(cls, config: Any) -> "ModelConfig":
        """Create from HuggingFace config.

        Args:
            config: HuggingFace model config.

        Returns:
            ModelConfig instance.
        """
        # Get number of KV heads (for GQA models)
        num_kv_heads = getattr(
            config,
            "num_key_value_heads",
            getattr(config, "num_attention_heads", 32),
        )

        # Get head dimension
        hidden_size = getattr(config, "hidden_size", 4096)
        num_attention_heads = getattr(config, "num_attention_heads", 32)
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

        # Estimate number of parameters
        num_layers = getattr(config, "num_hidden_layers", 32)
        intermediate_size = getattr(
            config, "intermediate_size", hidden_size * 4
        )
        vocab_size = getattr(config, "vocab_size", 32000)

        # Calculate parameters
        num_params = estimate_parameters(
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
        )

        return cls(
            num_parameters=num_params,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_position_embeddings=getattr(
                config, "max_position_embeddings", 4096
            ),
        )


def estimate_parameters(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
) -> int:
    """Estimate total number of parameters.

    Args:
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension.
        intermediate_size: FFN intermediate dimension.
        vocab_size: Vocabulary size.
        num_attention_heads: Number of attention heads.
        num_kv_heads: Number of KV heads.

    Returns:
        Estimated number of parameters.
    """
    head_dim = hidden_size // num_attention_heads

    # Embedding layers
    embedding_params = vocab_size * hidden_size * 2  # input + output embeddings

    # Per-layer parameters
    # Attention: Q, K, V projections + output projection
    q_params = hidden_size * (num_attention_heads * head_dim)
    kv_params = hidden_size * (num_kv_heads * head_dim) * 2  # K and V
    o_params = (num_attention_heads * head_dim) * hidden_size
    attention_params = q_params + kv_params + o_params

    # FFN: up, gate, down projections
    ffn_params = hidden_size * intermediate_size * 3  # up, gate, down

    # Layer norms
    ln_params = hidden_size * 4  # 2 layer norms * (weight + bias)

    layer_params = attention_params + ffn_params + ln_params
    total_layer_params = num_layers * layer_params

    # Final layer norm
    final_ln_params = hidden_size * 2

    return embedding_params + total_layer_params + final_ln_params


def get_bytes_per_param(dtype: str) -> int:
    """Get bytes per parameter for dtype.

    Args:
        dtype: Data type string.

    Returns:
        Bytes per parameter.
    """
    dtype_bytes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "fp8": 1,
        "int4": 0.5,
    }
    return dtype_bytes.get(dtype, 2)


def estimate_activations(
    model_config: ModelConfig,
    batch_size: int,
    seq_len: int,
) -> int:
    """Estimate activation memory.

    Args:
        model_config: Model configuration.
        batch_size: Batch size.
        seq_len: Sequence length.

    Returns:
        Estimated activation memory in bytes.
    """
    # Estimate activation memory per token
    # This includes attention scores, intermediate activations, etc.
    bytes_per_param = get_bytes_per_param(model_config.dtype)

    # Attention activations
    # Shape: (batch, num_heads, seq_len, seq_len)
    attention_activations = (
        batch_size
        * model_config.num_attention_heads
        * seq_len
        * seq_len
        * bytes_per_param
    )

    # FFN activations
    # Shape: (batch, seq_len, intermediate_size)
    ffn_activations = (
        batch_size
        * seq_len
        * model_config.intermediate_size
        * bytes_per_param
    )

    # Residual stream and layer norm
    residual_activations = (
        batch_size
        * seq_len
        * model_config.hidden_size
        * 4  # Multiple residual copies
        * bytes_per_param
    )

    # Sum per layer and multiply by number of layers
    per_layer = attention_activations + ffn_activations + residual_activations
    total_activations = per_layer * model_config.num_layers

    return int(total_activations)


def estimate_memory(
    model_config: ModelConfig,
    batch_size: int,
    seq_len: int,
    quantization: Optional[str] = None,
) -> MemoryEstimate:
    """Estimate GPU memory requirements.

    Args:
        model_config: Model configuration.
        batch_size: Maximum batch size.
        seq_len: Maximum sequence length.
        quantization: Quantization method (e.g., "awq", "gptq", "fp8").

    Returns:
        MemoryEstimate with detailed breakdown.
    """
    # Determine effective dtype for weights
    if quantization in ("awq", "gptq"):
        weight_bytes = 0.5  # 4-bit quantization
    elif quantization == "fp8":
        weight_bytes = 1
    elif quantization == "int8":
        weight_bytes = 1
    else:
        weight_bytes = get_bytes_per_param(model_config.dtype)

    # Model weights
    weights_memory = int(model_config.num_parameters * weight_bytes)

    # KV cache
    # Each token needs K and V for each layer
    kv_dtype = "bfloat16" if quantization != "fp8" else "fp8"
    kv_bytes_per_param = get_bytes_per_param(kv_dtype)

    kv_per_token = (
        2  # K and V
        * model_config.num_layers
        * model_config.num_kv_heads
        * model_config.head_dim
        * kv_bytes_per_param
    )
    kv_memory = batch_size * seq_len * kv_per_token

    # Activations
    activation_memory = estimate_activations(model_config, batch_size, seq_len)

    # Overhead (CUDA context, gradients if training, etc.)
    overhead = int(0.1 * (weights_memory + kv_memory))

    total = weights_memory + kv_memory + activation_memory + overhead

    return MemoryEstimate(
        weights=weights_memory,
        kv_cache=kv_memory,
        activations=activation_memory,
        overhead=overhead,
        total=total,
    )


def calculate_max_batch_size(
    model_config: ModelConfig,
    available_memory: int,
    seq_len: int,
    quantization: Optional[str] = None,
    memory_utilization: float = 0.9,
) -> int:
    """Calculate maximum batch size that fits in memory.

    Args:
        model_config: Model configuration.
        available_memory: Available GPU memory in bytes.
        seq_len: Sequence length.
        quantization: Quantization method.
        memory_utilization: Target memory utilization (0-1).

    Returns:
        Maximum batch size.
    """
    usable_memory = int(available_memory * memory_utilization)

    # Binary search for max batch size
    low, high = 1, 1024
    max_batch = 1

    while low <= high:
        mid = (low + high) // 2
        estimate = estimate_memory(model_config, mid, seq_len, quantization)

        if estimate.total <= usable_memory:
            max_batch = mid
            low = mid + 1
        else:
            high = mid - 1

    return max_batch


def calculate_max_model_len(
    model_config: ModelConfig,
    available_memory: int,
    batch_size: int,
    quantization: Optional[str] = None,
    memory_utilization: float = 0.9,
) -> int:
    """Calculate maximum model length that fits in memory.

    Args:
        model_config: Model configuration.
        available_memory: Available GPU memory in bytes.
        batch_size: Batch size.
        quantization: Quantization method.
        memory_utilization: Target memory utilization (0-1).

    Returns:
        Maximum model length.
    """
    usable_memory = int(available_memory * memory_utilization)

    # Start with model's max position embeddings
    max_len = model_config.max_position_embeddings

    # Binary search for max length
    low, high = 512, max_len
    result = 512

    while low <= high:
        mid = (low + high) // 2
        estimate = estimate_memory(model_config, batch_size, mid, quantization)

        if estimate.total <= usable_memory:
            result = mid
            low = mid + 1
        else:
            high = mid - 1

    return min(result, max_len)
