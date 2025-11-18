# vllm tune

Analyze a model and hardware to recommend optimal vLLM configuration.

## Synopsis

```bash
vllm tune <model> [options]
```

## Description

The `tune` command analyzes the specified model's architecture, available GPU hardware, and workload hints to generate configuration recommendations for optimal performance. This helps reduce the time spent on manual configuration tuning and improves out-of-box performance.

## Arguments

### model

Model name or path to analyze. Can be a HuggingFace model name or a local path.

Examples:
- `meta-llama/Llama-2-7b-hf`
- `mistralai/Mixtral-8x7B-v0.1`
- `/path/to/local/model`

## Options

### --optimize-for {throughput,latency,memory}

Optimization target for the recommendations.

- **throughput**: Maximize tokens processed per second (default)
- **latency**: Minimize time-to-first-token and inter-token latency
- **memory**: Minimize GPU memory usage

### --expected-batch-size INT

Expected batch size (number of concurrent requests). If not specified, auto-tune will choose an appropriate value based on the optimization target.

### --expected-input-length INT

Expected average input sequence length in tokens. Default: 500.

### --expected-output-length INT

Expected average output sequence length in tokens. Default: 200.

### --output, -o FILE

Save the recommended configuration to a YAML file. The file can be used with other vLLM commands.

## Examples

### Basic analysis

```bash
vllm tune meta-llama/Llama-2-7b-hf
```

### Optimize for throughput

```bash
vllm tune meta-llama/Llama-2-70b-hf --optimize-for throughput
```

### Optimize for latency

```bash
vllm tune mistralai/Mixtral-8x7B-v0.1 --optimize-for latency
```

### Save configuration to file

```bash
vllm tune meta-llama/Llama-2-70b-hf --output config.yaml
```

### With workload hints

```bash
vllm tune meta-llama/Llama-2-7b-hf \
    --expected-batch-size 100 \
    --expected-input-length 1000 \
    --expected-output-length 500
```

## Output

The command outputs:

1. **Model information**: Model name being analyzed
2. **Hardware detection**: Detected GPUs and their count
3. **Recommendations**: Suggested configuration parameters with explanations
4. **Performance estimates**: Estimated throughput, TTFT, and memory usage
5. **Usage example**: Command to apply the recommendations

Example output:

```
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

## See Also

- [Auto-Tune Feature Guide](../features/auto_tune.md)
- [vllm serve](serve.md)
- [Engine Arguments](../configuration/engine_args.md)
