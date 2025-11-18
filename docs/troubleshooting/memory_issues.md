# Memory Issues Troubleshooting

This guide covers common memory-related issues in vLLM and how to resolve them.

## Out of Memory (OOM) Errors

### Symptoms

- `RuntimeError: CUDA out of memory`
- `torch.cuda.OutOfMemoryError`
- Process killed by OOM killer
- Slow performance due to memory swapping

### Diagnosis

#### Check GPU Memory Usage

```python
import torch

# Current memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

#### Monitor via Metrics

If using the API server, check KV cache metrics:

```bash
curl http://localhost:8000/metrics | grep -E "(kv_cache|memory)"
```

Key metrics to watch:
- `vllm:gpu_cache_usage_perc` - Percentage of KV cache used
- `vllm:cpu_cache_usage_perc` - CPU cache usage (if enabled)

#### Check System Memory

```bash
# GPU memory
nvidia-smi

# System memory
free -h
```

### Solutions

#### Solution 1: Reduce GPU Memory Utilization

Reduce the fraction of GPU memory allocated to vLLM:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.85  # Default is 0.9
)
```

For the API server:

```bash
vllm serve model-name --gpu-memory-utilization 0.85
```

#### Solution 2: Reduce Maximum Sequence Length

Limit the maximum context length:

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_model_len=2048  # Reduce from model's max
)
```

This is especially effective for models with very long context windows (e.g., 128K tokens).

#### Solution 3: Reduce Concurrent Requests

Limit the number of sequences processed simultaneously:

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_num_seqs=128  # Default is 256
)
```

#### Solution 4: Enable Quantization

Use model quantization to reduce memory footprint:

```python
# FP8 quantization (recommended for H100/A100)
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="fp8"
)

# AWQ quantization (for other GPUs)
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="awq"
)

# Or use pre-quantized models
llm = LLM(model="TheBloke/Llama-2-7B-AWQ")
```

#### Solution 5: Enable KV Cache Quantization

Reduce KV cache memory by 50%:

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_cache_dtype="fp8_e4m3"  # Reduces KV cache by 50%
)
```

#### Solution 6: Use Tensor Parallelism

Distribute the model across multiple GPUs:

```python
# Distribute across 2 GPUs
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2
)
```

For larger models, use more GPUs:

```python
# 8-way tensor parallelism for very large models
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=8
)
```

#### Solution 7: Enforce Eager Mode

Disable CUDA graphs to reduce memory overhead (at some performance cost):

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enforce_eager=True
)
```

### Memory Estimation

#### Model Weights Memory

Approximate GPU memory for model weights:

| Model Size | FP16 Memory | FP8 Memory | INT4 Memory |
|------------|-------------|------------|-------------|
| 7B         | ~14 GB      | ~7 GB      | ~3.5 GB     |
| 13B        | ~26 GB      | ~13 GB     | ~6.5 GB     |
| 70B        | ~140 GB     | ~70 GB     | ~35 GB      |

#### KV Cache Memory

KV cache memory per token (approximate):

```
Memory per token = 2 * num_layers * hidden_size * 2 bytes (FP16)
```

For Llama 2 7B (32 layers, 4096 hidden):
- FP16: ~512 KB per token
- FP8: ~256 KB per token

### Prevention

1. **Test memory requirements** before production deployment
2. **Monitor KV cache usage** to understand actual memory pressure
3. **Set appropriate max_model_len** for your use case
4. **Use the right quantization** for your hardware
5. **Consider chunked prefill** for long prompts

### Related Documentation

- [Conserving Memory](../configuration/conserving_memory.md)
- [Quantization Guide](../configuration/optimization.md)
- [Parallelism and Scaling](../serving/parallelism_scaling.md)
- [General Troubleshooting](../usage/troubleshooting.md)
