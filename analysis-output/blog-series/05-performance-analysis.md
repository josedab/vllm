# Performance Analysis and Optimization

**Part 5 of the vLLM Technical Blog Series**
**Commit SHA:** `67745d189fd981ee824bde35666a3737a962c031`

---

## What You'll Learn

- vLLM's performance characteristics and bottlenecks
- Quantization options and trade-offs
- Compilation and CUDA graph optimization
- Benchmarking and profiling techniques
- Production tuning strategies

---

## Introduction

vLLM achieves state-of-the-art LLM serving throughput through careful optimization at every level—from custom CUDA kernels to intelligent scheduling. In this post, we'll explore these optimizations, understand their trade-offs, and learn how to tune vLLM for your specific workload.

---

## Performance Characteristics

### The Two Phases of LLM Inference

LLM inference has two distinct phases with different characteristics:

**Prefill (Prompt Processing):**
- Compute-bound
- Processes all input tokens in parallel
- High arithmetic intensity

**Decode (Token Generation):**
- Memory-bandwidth-bound
- Generates one token at a time
- Low arithmetic intensity

```
Phase      | Bottleneck  | Optimization Strategy
-----------|-------------|----------------------
Prefill    | Compute     | Batch prompts, use tensor cores
Decode     | Memory BW   | Batch requests, quantize weights
```

### Typical Performance Numbers

On A100 80GB with Llama-2-7B:

| Metric | Value | Notes |
|--------|-------|-------|
| Throughput | 2,000-4,000 tok/s | Depends on batch size |
| TTFT | 20-50ms | Time to first token |
| ITL | 10-20ms | Inter-token latency |
| Memory | 90-95% | KV cache utilization |

---

## Quantization

Quantization reduces model precision to decrease memory usage and increase throughput.

### Available Methods

vLLM supports 25+ quantization methods. Key options:

| Method | Precision | Speedup | Accuracy Impact |
|--------|-----------|---------|-----------------|
| BF16 (baseline) | 16-bit | 1x | None |
| FP8 | 8-bit | 1.5-2x | ~0.5% |
| AWQ | 4-bit | 2-3x | ~1% |
| GPTQ | 4-bit | 2-3x | ~1% |
| GPTQ-Marlin | 4-bit | 3-4x | ~1% |

### FP8 Quantization

The best balance of speed and accuracy:

```python
from vllm import LLM

# Use FP8 quantized model
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="fp8",
    # Or load pre-quantized:
    # model="neuralmagic/Llama-2-7b-fp8"
)
```

FP8 provides:
- ~2x memory reduction
- ~1.5-2x throughput increase
- <0.5% accuracy loss

### 4-bit Quantization

For maximum efficiency:

```python
# AWQ quantization
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq"
)

# GPTQ with Marlin kernels (fastest)
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq_marlin"
)
```

### KV Cache Quantization

Compress the KV cache to fit more concurrent requests:

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_cache_dtype="fp8_e4m3",  # or "fp8_e5m2"
)
```

This gives ~50% memory reduction for the KV cache with minimal accuracy impact.

---

## Compilation and CUDA Graphs

### torch.compile

vLLM uses torch.compile for kernel fusion and optimization:

```python
from vllm import LLM
from vllm.config import CompilationLevel

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    compilation_config={
        "level": CompilationLevel.PIECEWISE,  # or FULL
        "use_cudagraph": True,
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
    }
)
```

### CUDA Graphs

CUDA graphs capture a sequence of GPU operations and replay them with minimal CPU overhead:

```
Without CUDA Graphs:
CPU → GPU → CPU → GPU → CPU → GPU → ...
     ↑      ↑      ↑
  Kernel launch overhead

With CUDA Graphs:
CPU → [Captured Graph] → CPU
      GPU → GPU → GPU
          ↑
    Single launch, all kernels
```

**Results:** 30-50% latency reduction for decode phase.

### Compilation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| NONE | No compilation | Debugging |
| DYNAMO_TRACE_ONCE | Single trace, dynamic shapes | Flexibility |
| PIECEWISE | Per-layer graphs | Balance |
| FULL | End-to-end graph | Best performance |

```python
# For production (best performance)
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    compilation_config={"level": "FULL", "use_cudagraph": True}
)

# For development (flexibility)
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    compilation_config={"level": "NONE"}
)
```

---

## Speculative Decoding

Generate multiple tokens per forward pass:

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
    }
)
```

**Methods:**
- **ngram**: Uses prompt patterns (no extra model)
- **draft_model**: Smaller model proposes tokens
- **medusa/eagle**: Additional prediction heads

**Results:** 1.5-3x latency reduction for suitable workloads.

---

## Benchmarking

### Built-in Benchmarks

```bash
# Throughput benchmark
python benchmarks/throughput.py \
    --model meta-llama/Llama-2-7b-hf \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 1000

# Latency benchmark with profiling
python benchmarks/latency.py \
    --model meta-llama/Llama-2-7b-hf \
    --batch-size 1 \
    --input-len 128 \
    --output-len 128 \
    --profile
```

### Key Metrics

```python
# Calculate throughput
total_tokens = num_prompts * (input_len + output_len)
throughput = total_tokens / elapsed_time

# Calculate latency percentiles
ttft_p50 = np.percentile(ttft_samples, 50)
ttft_p99 = np.percentile(ttft_samples, 99)
```

### Profiling with Torch Profiler

```python
import torch
from vllm import LLM

llm = LLM(model="meta-llama/Llama-2-7b-hf")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    llm.generate(["Hello, world!"])

prof.export_chrome_trace("trace.json")
# Open in chrome://tracing
```

### vLLM Layerwise Profiler

```python
from vllm.profiler.layerwise_profile import layerwise_profile

with layerwise_profile() as prof:
    llm.generate(["Hello, world!"])

# Print per-layer timing
prof.results.print_model_table()
```

---

## Bottleneck Analysis

### Common Bottlenecks

**1. KV Cache Memory**

Symptom: Low batch sizes, OOM errors

```python
# Check via metrics
# vllm:kv_cache_usage_perc > 95%

# Solutions:
llm = LLM(
    model="...",
    gpu_memory_utilization=0.95,  # More memory for KV
    kv_cache_dtype="fp8_e4m3",    # Compress KV cache
    max_model_len=2048,           # Reduce max context
)
```

**2. Prefill Compute**

Symptom: High TTFT, GPU compute at 100%

```python
# Solutions:
llm = LLM(
    model="...",
    enable_chunked_prefill=True,  # Break up long prompts
    max_num_batched_tokens=4096,  # Limit prefill batch
)
```

**3. Decode Memory Bandwidth**

Symptom: Low ITL, memory bandwidth saturated

```python
# Solutions:
llm = LLM(
    model="...",
    quantization="fp8",           # Reduce weight size
    max_num_seqs=256,             # Increase batch size
)
```

**4. Scheduling Overhead**

Symptom: CPU at 100%, low GPU utilization

```python
# V1 engine has lower overhead
# Use multiprocess mode
```

---

## Production Tuning Guide

### High Throughput Configuration

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",

    # Maximize batching
    max_num_seqs=256,
    max_num_batched_tokens=8192,

    # Memory efficiency
    gpu_memory_utilization=0.95,
    quantization="fp8",
    kv_cache_dtype="fp8_e4m3",

    # Compilation
    compilation_config={
        "level": "PIECEWISE",
        "use_cudagraph": True,
    },

    # Caching
    enable_prefix_caching=True,
)
```

### Low Latency Configuration

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",

    # Smaller batches for lower latency
    max_num_seqs=32,
    max_num_batched_tokens=2048,

    # Full compilation for minimum overhead
    compilation_config={
        "level": "FULL",
        "use_cudagraph": True,
    },

    # Speculative decoding
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
    },

    # Chunked prefill to avoid blocking
    enable_chunked_prefill=True,
)
```

### Memory Constrained Configuration

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",

    # Aggressive quantization
    quantization="gptq_marlin",
    kv_cache_dtype="fp8_e4m3",

    # Conservative memory
    gpu_memory_utilization=0.85,
    max_model_len=2048,

    # CPU offloading
    swap_space=16,  # GB
)
```

---

## Performance Monitoring

### Prometheus Metrics to Watch

```python
# Latency
vllm:time_to_first_token_seconds_bucket  # TTFT histogram
vllm:inter_token_latency_seconds_bucket  # ITL histogram

# Throughput
vllm:prompt_tokens  # Input tokens processed
vllm:generation_tokens  # Output tokens generated

# Resource usage
vllm:kv_cache_usage_perc  # Memory pressure
vllm:num_requests_running  # Concurrent requests
vllm:num_requests_waiting  # Queue depth

# Cache efficiency
vllm:prefix_cache_hits / vllm:prefix_cache_queries
```

### Alerting Thresholds

```yaml
# Example Prometheus alerts
- alert: HighKVCacheUsage
  expr: vllm:kv_cache_usage_perc > 0.95
  for: 5m
  annotations:
    summary: "KV cache usage above 95%"

- alert: HighQueueDepth
  expr: vllm:num_requests_waiting > 100
  for: 2m
  annotations:
    summary: "Request queue growing"
```

---

## Trade-off Summary

| Optimization | Throughput | Latency | Accuracy | Memory |
|-------------|------------|---------|----------|--------|
| FP8 Quantization | +50-100% | +20-30% | -0.5% | -50% |
| 4-bit Quantization | +100-200% | +30-50% | -1% | -75% |
| KV Cache FP8 | +20-30% | Neutral | -0.1% | -50% (KV) |
| CUDA Graphs | +10-20% | -30-50% | Neutral | +5% |
| Speculative | +50-100% | -40-60% | Neutral | +10% |
| Prefix Caching | +100%+ | -50%+ | Neutral | +5% |

---

## Key Takeaways

1. **Prefill is compute-bound, decode is memory-bound** - optimize accordingly

2. **FP8 is the sweet spot** for most production workloads

3. **CUDA graphs** provide significant latency reduction

4. **Speculative decoding** can dramatically improve latency for suitable workloads

5. **Prefix caching** is a free win for repeated prompts

6. **Monitor KV cache usage** - it's often the bottleneck

7. **Profile before optimizing** - understand your specific bottleneck

---

## Next Steps

In [Part 6: Distributed Inference](06-distributed-inference.md), we'll explore scaling vLLM across multiple GPUs and nodes using tensor, pipeline, and expert parallelism.

---

## Code References

- **Benchmarks:** [benchmarks/](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/benchmarks/)
- **Quantization:** [vllm/model_executor/layers/quantization/](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/model_executor/layers/quantization/)
- **Compilation:** [vllm/compilation/](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/compilation/)
- **Profiler:** [vllm/profiler/](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/profiler/)

---

*This is Part 5 of the vLLM Technical Blog Series based on commit `67745d189fd981ee824bde35666a3737a962c031`.*
