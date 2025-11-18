# Attention Backend Selection Guide

This document describes how vLLM selects attention backends and how users can configure them for optimal performance.

## Overview

vLLM supports multiple attention backends optimized for different hardware platforms and use cases. The system automatically selects the best backend based on your hardware and configuration, but you can also manually override this selection.

## Available Backends

### Core Backends (P0 - Highest Priority)

| Backend | Platform | Description | Key Features |
|---------|----------|-------------|--------------|
| **FlashAttention** | NVIDIA CUDA | Default for NVIDIA GPUs | Full feature support, FP8, cascade attention |
| **FlashInfer** | NVIDIA CUDA | Advanced features | TensorRT-LLM integration, FP8, NVFP4 |
| **ROCm Attention** | AMD ROCm | Default for AMD GPUs | Optimized for MI series |
| **Pallas** | TPU | Default for Google TPUs | XLA-optimized |

### Secondary Backends (P1 - Standard Priority)

| Backend | Platform | Description | Key Features |
|---------|----------|-------------|--------------|
| **Triton** | CUDA/ROCm | Portable, customizable | Works across platforms, FP8 |
| **xFormers** | CUDA | Legacy support | Wide head size support |

### Fallback Backends (P2 - Low Priority)

| Backend | Platform | Description | Key Features |
|---------|----------|-------------|--------------|
| **CPU** | CPU | CPU fallback | Float32 support |
| **torch.sdpa** | All | PyTorch native | Universal compatibility |

## Feature Capabilities Matrix

Each backend declares its capabilities through the `CAPABILITIES` dictionary:

| Feature | Flash Attn | FlashInfer | Triton | xFormers | ROCm | Pallas | CPU |
|---------|------------|------------|--------|----------|------|--------|-----|
| Paged Attention | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Prefix Caching | Yes | Yes | No | No | Yes | No | No |
| Sliding Window | Yes | Yes | Yes | No | No | No | No |
| Speculative Decoding | Yes | Yes | Yes | Yes | Yes | No | No |
| Chunked Prefill | Yes | Yes | Yes | Yes | Yes | No | No |
| Multi-step Decoding | Yes | Yes | No | No | No | No | No |

## Automatic Selection

vLLM automatically selects the best backend based on:

1. **Platform detection** - CUDA, ROCm, TPU, CPU
2. **Hardware capabilities** - Compute capability, available memory
3. **Model requirements** - Head size, dtype, attention type
4. **Feature requirements** - What features the model needs

### Selection Priority

Backends are selected by priority (higher = preferred):
- P0 backends (priority 100): Platform-native, full-featured
- P1 backends (priority 50): Good alternatives with some limitations
- P2 backends (priority 10): Fallbacks for compatibility

## Manual Configuration

### Environment Variable

Set the backend using the `VLLM_ATTENTION_BACKEND` environment variable:

```bash
# Use FlashAttention
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Use FlashInfer
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Use Triton
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

### Programmatic Selection

```python
from vllm.attention.selector import global_force_attn_backend
from vllm.attention.backends.registry import AttentionBackendEnum

# Force a specific backend
global_force_attn_backend(AttentionBackendEnum.FLASH_ATTN)
```

### Context Manager

```python
from vllm.attention.selector import global_force_attn_backend_context_manager
from vllm.attention.backends.registry import AttentionBackendEnum

with global_force_attn_backend_context_manager(AttentionBackendEnum.FLASHINFER):
    # Use FlashInfer in this context
    pass
```

## Feature-Based Selection

You can select a backend based on required features:

```python
from vllm.attention.selector import get_backend_by_features

# Get the highest priority backend that supports these features
backend = get_backend_by_features({
    "prefix_caching",
    "sliding_window",
})

if backend:
    print(f"Selected: {backend.get_name()}")
else:
    print("No backend supports all required features")
```

## Checking Backend Capabilities

### Query All Backends

```python
from vllm.attention.selector import get_backend_capabilities_summary

summary = get_backend_capabilities_summary()
for name, info in summary.items():
    print(f"{name}:")
    print(f"  Priority: {info['priority']}")
    print(f"  Capabilities: {info['capabilities']}")
```

### Query Specific Backend

```python
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

# Check specific capabilities
print(f"Supports prefix caching: {FlashAttentionBackend.supports_prefix_caching()}")
print(f"Supports sliding window: {FlashAttentionBackend.supports_sliding_window()}")

# Check all capabilities
caps = FlashAttentionBackend.get_capabilities()
print(f"All capabilities: {caps}")

# Check if it supports a set of features
features = {"prefix_caching", "sliding_window"}
print(f"Supports required features: {FlashAttentionBackend.supports_features(features)}")
```

## Deprecation Policy

Some backends may be deprecated over time. When using a deprecated backend:

1. A warning will be logged with the deprecation timeline
2. The system will automatically migrate to the replacement backend
3. Deprecated backends are removed after one major version

### Current Deprecations

Check `vllm.attention.selector.DEPRECATED_BACKENDS` for the current list of deprecated backends and their replacements.

## Troubleshooting

### Backend Not Available

If your preferred backend is not available:

1. Check hardware compatibility (compute capability, platform)
2. Verify required libraries are installed (flash-attn, flashinfer, xformers)
3. Check dtype compatibility (some backends don't support FP32)

### Performance Issues

If performance is not optimal:

1. Use P0 backends when possible (FlashAttention, FlashInfer, ROCm)
2. Enable features like prefix caching and chunked prefill
3. Check that the selected backend matches your hardware

### Debugging Selection

```python
import logging
logging.getLogger("vllm.attention.selector").setLevel(logging.DEBUG)
```

This will show detailed logs about backend selection decisions.

## Extending with Custom Backends

You can register custom backends:

```python
from vllm.attention.backends.registry import register_backend, AttentionBackendEnum
from vllm.attention.backends.abstract import AttentionBackend

@register_backend(AttentionBackendEnum.CUSTOM)
class MyCustomBackend(AttentionBackend):
    CAPABILITIES = {
        "paged_attention": True,
        "prefix_caching": True,
        # ... other capabilities
    }
    selection_priority = 75  # Between P0 and P1

    @staticmethod
    def get_name():
        return "MY_CUSTOM"

    # ... implement other required methods
```

## Best Practices

1. **Use automatic selection** - Let vLLM choose the best backend for your hardware
2. **Check capabilities first** - Before forcing a backend, verify it supports your needs
3. **Test with your workload** - Different backends may perform better for different use cases
4. **Stay updated** - New backends and improvements are added regularly
5. **Monitor deprecations** - Update configurations when backends are deprecated

## Related Documentation

- [Paged Attention](paged_attention.md)
- [Prefix Caching](prefix_caching.md)
- [CUDA Graphs](cuda_graphs.md)
