# Configuration Profiles

vLLM provides predefined configuration profiles for common use cases. Profiles bundle together optimized settings so you don't have to manually tune dozens of parameters.

## Quick Start

```python
from vllm import LLM
from vllm.config import ConfigProfile

# Use a profile for your use case
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    profile=ConfigProfile.HIGH_THROUGHPUT
)
```

## Available Profiles

### HIGH_THROUGHPUT

Optimized for maximum tokens per second with larger batches. Best for:
- Batch processing
- High-volume production workloads
- Background processing jobs

**Settings:**
- `max_num_seqs`: 256
- `gpu_memory_utilization`: 0.95
- `enable_prefix_caching`: True
- `enable_chunked_prefill`: True
- Compilation level: PIECEWISE

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    profile=ConfigProfile.HIGH_THROUGHPUT
)
```

### LOW_LATENCY

Optimized for minimum time-to-first-token and fast responses. Best for:
- Interactive applications
- Real-time chat
- Streaming responses

**Settings:**
- `max_num_seqs`: 32
- `gpu_memory_utilization`: 0.9
- `enable_prefix_caching`: True
- `enable_chunked_prefill`: True
- Compilation level: PIECEWISE

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    profile=ConfigProfile.LOW_LATENCY
)
```

### MEMORY_CONSTRAINED

Optimized to minimize GPU memory usage. Best for:
- Running large models on limited hardware
- Development on consumer GPUs
- Multi-model deployments

**Settings:**
- `max_num_seqs`: 64
- `gpu_memory_utilization`: 0.8
- `enable_prefix_caching`: True
- `enable_chunked_prefill`: True
- `swap_space`: 8 GiB
- Compilation level: None (saves memory)

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    profile=ConfigProfile.MEMORY_CONSTRAINED
)
```

### DEVELOPMENT

Balanced settings with verbose logging for debugging. Best for:
- Development and testing
- Debugging issues
- Experimentation

**Settings:**
- `max_num_seqs`: 128
- `gpu_memory_utilization`: 0.85
- `enable_prefix_caching`: True
- `enforce_eager`: True (easier debugging)
- Compilation level: None (faster startup)

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    profile=ConfigProfile.DEVELOPMENT
)
```

## Overriding Profile Settings

You can override any profile setting with explicit parameters:

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    profile=ConfigProfile.HIGH_THROUGHPUT,
    max_model_len=8192,  # Override just this
    max_num_seqs=128     # Use smaller batch size
)
```

User-provided values always take precedence over profile defaults.

## Using Profiles from CLI

You can also use profiles from the command line:

```bash
vllm serve meta-llama/Llama-2-7b-hf --profile high_throughput
```

Available profile names:
- `high_throughput`
- `low_latency`
- `memory_constrained`
- `development`

## Listing Available Profiles

You can programmatically list all available profiles:

```python
from vllm.config import list_profiles

for profile in list_profiles():
    print(f"{profile['name']}: {profile['description'][:80]}...")
```

## Configuration Validation

vLLM provides validation for your configuration with actionable error messages:

```python
from vllm.config import validate_config

# Validate settings and get warnings
warnings = validate_config(
    tensor_parallel_size=2,
    gpu_memory_utilization=0.99,  # Will generate a warning
    max_num_seqs=4                # Will generate a warning
)

for warning in warnings:
    print(warning)
```

## Best Practices

1. **Start with a profile** that matches your use case, then tune individual settings as needed.

2. **Monitor memory usage** when using HIGH_THROUGHPUT profile, as it uses 95% GPU memory.

3. **Use DEVELOPMENT profile** during development for easier debugging, then switch to a production profile.

4. **Check warnings** from the validation system to catch configuration issues early.

5. **Override conservatively** - profiles are designed to work well together, so only override settings you specifically need to change.

## Migration from Manual Configuration

If you're currently using manual configuration:

**Before:**
```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.9,
    max_num_seqs=256,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    # ... many more options
)
```

**After:**
```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    profile=ConfigProfile.HIGH_THROUGHPUT
)
```

The profile system provides sensible defaults while still allowing full customization when needed.
