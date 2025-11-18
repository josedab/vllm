# Performance Issues Troubleshooting

This guide covers common performance issues in vLLM and how to diagnose and resolve them.

## Low Throughput

### Symptoms

- Fewer tokens per second than expected
- High latency per request
- GPU utilization lower than expected

### Diagnosis

#### Check GPU Utilization

```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Or use nvtop for better visualization
nvtop
```

Target: GPU utilization should be >90% for optimal throughput.

#### Check vLLM Metrics

```bash
# Get current metrics
curl http://localhost:8000/metrics | grep -E "(throughput|latency|queue)"
```

Key metrics:
- `vllm:avg_generation_throughput_toks_per_s` - Tokens generated per second
- `vllm:avg_prompt_throughput_toks_per_s` - Prompt tokens processed per second
- `vllm:num_requests_waiting` - Queue depth (should be >0 for max throughput)

#### Enable Statistics Logging

```bash
export VLLM_LOG_STATS_INTERVAL=1.0
```

This logs queue sizes and throughput every second.

### Solutions

#### Solution 1: Increase Batch Size

Ensure enough concurrent requests to maximize GPU utilization:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_num_seqs=256,  # Increase concurrent sequences
    max_num_batched_tokens=8192  # Increase batch token limit
)
```

#### Solution 2: Enable Prefix Caching

Reuse KV cache for common prefixes:

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True
)
```

Especially effective for:
- System prompts
- Few-shot examples
- RAG applications with shared context

#### Solution 3: Enable Chunked Prefill

Process long prompts in chunks for better batching:

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_chunked_prefill=True,
    max_num_batched_tokens=2048
)
```

#### Solution 4: Use Speculative Decoding

Enable draft model for faster decoding:

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="meta-llama/Llama-2-7b-hf",
    num_speculative_tokens=5
)
```

#### Solution 5: Optimize Memory Utilization

Higher memory utilization = larger batches = better throughput:

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.95  # Increase from default 0.9
)
```

#### Solution 6: Use CUDA Graphs

Ensure CUDA graphs are enabled (default):

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enforce_eager=False  # Default, enables CUDA graphs
)
```

Note: CUDA graphs increase startup time but improve steady-state performance.

## High Latency

### Time to First Token (TTFT)

High TTFT usually indicates prompt processing bottlenecks.

#### Diagnosis

Check if the issue is:
1. **Long prompts** - Check prompt token count
2. **Queue depth** - Monitor `num_requests_waiting`
3. **Memory pressure** - Check for preemptions

#### Solutions

1. **Enable chunked prefill** for long prompts
2. **Reduce queue depth** if latency-sensitive
3. **Use smaller batches** for lower latency (at cost of throughput)

```python
# Latency-optimized configuration
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_num_seqs=64,  # Lower for reduced latency
    enable_chunked_prefill=True
)
```

### Inter-Token Latency

High latency between tokens usually indicates decode bottlenecks.

#### Diagnosis

1. Check for preemptions in metrics
2. Monitor KV cache usage
3. Look for memory pressure

#### Solutions

1. **Reduce concurrent requests**
2. **Use speculative decoding**
3. **Enable KV cache quantization** to fit more sequences

## Model Loading Issues

### Slow Startup

#### Diagnosis

- Check if downloading vs loading
- Monitor disk and network I/O
- Check memory usage during loading

#### Solutions

1. **Pre-download models**:
   ```bash
   huggingface-cli download meta-llama/Llama-2-7b-hf
   ```

2. **Use local SSD storage** instead of network filesystems

3. **Enable tensor loading optimization**:
   ```python
   llm = LLM(
       model="meta-llama/Llama-2-7b-hf",
       load_format="safetensors"  # Faster than pickle
   )
   ```

## Distributed Performance

### Poor Scaling

#### Symptoms

- 2 GPUs not giving 2x performance
- High communication overhead

#### Diagnosis

Check NVLink/network bandwidth:

```bash
# Check NVLink topology
nvidia-smi topo -m
```

#### Solutions

1. **Use NVLink** when available (faster than PCIe)

2. **Optimize tensor parallel size**:
   - For 7B models: 1 GPU
   - For 13B models: 1-2 GPUs
   - For 70B models: 4-8 GPUs

3. **Use pipeline parallelism** for very large models:
   ```python
   llm = LLM(
       model="very-large-model",
       tensor_parallel_size=4,
       pipeline_parallel_size=2
   )
   ```

## Benchmarking

### Recommended Benchmarking Approach

```bash
# Use vLLM's built-in benchmark
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf &

# Run benchmark
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-2-7b-hf \
    --request-rate 10 \
    --num-prompts 1000
```

### Key Metrics to Track

| Metric | Latency-Optimized | Throughput-Optimized |
|--------|-------------------|---------------------|
| P50 TTFT | <100ms | <500ms |
| P99 TTFT | <500ms | <2000ms |
| Tokens/sec | Variable | >1000 |
| GPU Util | Variable | >90% |

## Environment Variables

Useful environment variables for performance tuning:

```bash
# Enable statistics logging
export VLLM_LOG_STATS_INTERVAL=1.0

# Tune CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable detailed profiling
export VLLM_LOGGING_LEVEL=DEBUG
```

## Related Documentation

- [Optimization Guide](../configuration/optimization.md)
- [Benchmarking Guide](../contributing/benchmarks.md)
- [Speculative Decoding](../features/spec_decode.md)
- [Prefix Caching](../features/automatic_prefix_caching.md)
- [General Troubleshooting](../usage/troubleshooting.md)
