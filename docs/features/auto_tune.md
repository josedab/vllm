# Automatic Configuration Tuning

vLLM provides an automatic configuration tuning system that analyzes your model, hardware, and workload to recommend optimal settings. This reduces time-to-production and improves out-of-box performance.

## Quick Start

### Using the CLI

The simplest way to use auto-tuning is through the CLI:

```bash
# Basic analysis
vllm tune meta-llama/Llama-2-7b-hf

# Optimize for throughput
vllm tune meta-llama/Llama-2-70b-hf --optimize-for throughput

# Optimize for latency
vllm tune mistralai/Mixtral-8x7B-v0.1 --optimize-for latency

# Save configuration to file
vllm tune meta-llama/Llama-2-70b-hf --output config.yaml
```

### Using the Python API

```python
from vllm import LLM, AutoTune

# Basic: Auto-tune for balanced performance
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    auto_tune=True
)

# With optimization hints
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    auto_tune=AutoTune(
        optimize_for="throughput",  # or "latency", "memory"
        expected_batch_size=100,
        expected_input_length=500,
        expected_output_length=200,
    )
)
```

## Analyzing Without Creating LLM

You can analyze a model without instantiating the LLM:

```python
from vllm.autotune import AutoTuner

tuner = AutoTuner()
result = tuner.analyze(
    model="meta-llama/Llama-2-70b-hf",
    optimize_for="throughput",
    hints={
        "expected_batch_size": 100,
        "expected_input_length": 500,
    }
)

# View recommendations
print(result.explain())

# Get configuration dict
config = result.to_engine_args()
print(config)
```

## Optimization Targets

### Throughput

Maximizes tokens processed per second. Best for:
- Batch processing
- High-volume API services
- Cost optimization

```python
auto_tune=AutoTune(optimize_for="throughput")
```

### Latency

Minimizes time-to-first-token and inter-token latency. Best for:
- Interactive applications
- Real-time chat
- Low-latency requirements

```python
auto_tune=AutoTune(optimize_for="latency")
```

### Memory

Minimizes GPU memory usage. Best for:
- Running on smaller GPUs
- Multi-model deployments
- Development environments

```python
auto_tune=AutoTune(optimize_for="memory")
```

## Configuration Parameters

Auto-tune analyzes and recommends values for:

| Parameter | Description |
|-----------|-------------|
| `tensor_parallel_size` | Number of GPUs for tensor parallelism |
| `max_num_seqs` | Maximum concurrent sequences (batch size) |
| `max_model_len` | Maximum context length |
| `gpu_memory_utilization` | Target GPU memory usage |
| `quantization` | Quantization method (fp8, awq, gptq) |
| `enable_prefix_caching` | Enable KV cache reuse |

## Workload Hints

Provide hints about your expected workload for better recommendations:

```python
AutoTune(
    expected_batch_size=100,      # Typical number of concurrent requests
    expected_input_length=500,    # Average input tokens
    expected_output_length=200,   # Average output tokens
)
```

## Hardware Detection

Auto-tune automatically detects your GPU hardware and applies hardware-specific optimizations:

- **A100 80GB**: Flash Attention, FP8 quantization
- **H100**: FlashInfer attention, FP8 quantization
- **RTX 4090**: Flash Attention, AWQ quantization
- **V100**: Flash Attention, GPTQ quantization

## Performance Estimates

Auto-tune provides estimated performance metrics:

```python
result = tuner.analyze(model="meta-llama/Llama-2-70b-hf")

print(f"Estimated throughput: {result.estimated_throughput:.0f} tok/s")
print(f"Estimated TTFT: {result.estimated_ttft * 1000:.1f}ms")
print(f"Estimated memory: {result.estimated_memory:.1f}GB")
print(f"Confidence: {result.confidence:.0%}")
```

Note: Estimates are based on theoretical models and actual performance may vary.

## CLI Reference

```
vllm tune <model> [options]

Arguments:
  model                      Model name or path to analyze

Options:
  --optimize-for {throughput,latency,memory}
                             Optimization target (default: throughput)
  --expected-batch-size INT  Expected batch size
  --expected-input-length INT
                             Expected input length (default: 500)
  --expected-output-length INT
                             Expected output length (default: 200)
  --output, -o FILE          Save configuration to YAML file
```

## Example Output

```
$ vllm tune meta-llama/Llama-2-70b-hf --optimize-for throughput

Model: meta-llama/Llama-2-70b-hf
GPU: 8x NVIDIA A100 80GB

Recommendations:
  tensor_parallel_size: 8 (model requires 8 GPUs for memory)
  max_num_seqs: 256 (maximize batching for throughput)
  max_model_len: 4096 (maximum context that fits in memory)
  quantization: fp8 (2x memory savings, <0.5% quality loss)
  attention_backend: FLASH_ATTN (fastest for A100-80GB)
  enable_prefix_caching: True
  gpu_memory_utilization: 0.9

Estimated Performance:
  Throughput: 4,200 tok/s
  TTFT: 35.0ms
  Memory: 72.0GB
  Confidence: 80%

To use these recommendations:
  vllm serve meta-llama/Llama-2-70b-hf --tensor-parallel-size 8 --max-num-seqs 256 --max-model-len 4096 --quantization fp8
```

## Best Practices

1. **Start with auto-tune**: Use auto-tune as a starting point, then fine-tune based on actual performance.

2. **Provide accurate hints**: Better workload hints lead to better recommendations.

3. **Test recommendations**: Always benchmark the recommended configuration on your actual workload.

4. **Consider your priorities**: Choose the right optimization target for your use case.

5. **Monitor performance**: Use vLLM metrics to monitor actual vs. estimated performance.

## Limitations

- Performance estimates are theoretical and may differ from actual performance
- Hardware profiles may not cover all GPU models
- Complex model architectures may not be fully supported
- Multi-model deployments require manual configuration

## Troubleshooting

### "Generic" hardware profile selected

If auto-tune selects the "generic" profile, your GPU may not be recognized. You can still use the recommendations, but confidence will be lower.

### Memory overestimate

If the model doesn't fit despite recommendations, try:
- Reducing `gpu_memory_utilization`
- Using quantization
- Increasing `tensor_parallel_size`

### Low throughput

If actual throughput is lower than estimated:
- Increase `max_num_seqs`
- Enable chunked prefill
- Check for CPU bottlenecks
