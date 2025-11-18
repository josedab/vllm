# Migrating from V0 to V1

This guide helps you migrate from the deprecated V0 engine to V1. The V0 engine is deprecated and will be removed in v1.0.0. V1 provides significant performance improvements (1.7x throughput) and cleaner code.

## Quick Migration

For most users, migration is straightforward since the public API is unchanged:

```python
# Before (V0) - implicit
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B")

# After (V1) - same API!
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B")  # V1 is default
```

V1 is now the default engine. No code changes are required for basic usage.

## Environment Variables

If you were explicitly using V0:

```bash
# Old (V0)
export VLLM_USE_V1=0

# New (V1) - remove or set to 1
export VLLM_USE_V1=1  # or unset
```

## Breaking Changes

### 1. `best_of` Parameter (V0-only)

The `best_of` sampling parameter is not supported in V1:

```python
# V0 (deprecated)
sampling_params = SamplingParams(n=1, best_of=5)

# V1 - use n with custom selection
sampling_params = SamplingParams(n=5)
outputs = llm.generate(prompts, sampling_params)
# Select best output yourself based on your criteria
best_output = select_best(outputs)
```

### 2. Engine Internal Access

If you accessed engine internals:

```python
# V0 (deprecated)
scheduler = engine.scheduler
block_manager = engine.scheduler.block_manager

# V1 - use engine_core
engine_core = engine.engine_core
# Note: Internal APIs have changed
```

### 3. Scheduler Output Format

The scheduler output format has changed:

```python
# For compatibility, use the compat layer
from vllm.compat.v0 import get_scheduler_output_v0_format

v0_output = get_scheduler_output_v0_format(v1_output)
```

### 4. Configuration Changes

Some configuration options have been renamed or removed:

| V0 Option | V1 Option | Notes |
|-----------|-----------|-------|
| `swap_space` | `swap_space` | Still supported but implementation differs |
| Internal scheduler config | N/A | V1 has different scheduler architecture |

## Compatibility Layer

For users who need time to migrate, a compatibility layer is available:

```python
from vllm.compat.v0 import (
    get_scheduler_output_v0_format,
    wrap_v1_engine_with_v0_interface,
)

# Wrap V1 engine with V0-like interface
engine = LLMEngine.from_engine_args(args)
v0_compat_engine = wrap_v1_engine_with_v0_interface(engine)

# Access V0-style attributes (with deprecation warnings)
scheduler = v0_compat_engine.scheduler
```

**Warning**: The compatibility layer is temporary and will be removed in a future version.

## Performance Improvements

V1 provides significant improvements:

- **1.7x throughput improvement** - Better batching and scheduling
- **Lower latency** - Optimized token generation
- **Better memory efficiency** - Improved KV cache management
- **Cleaner codebase** - Easier to maintain and extend

## API Server

The OpenAI-compatible API server works the same way:

```bash
# Start server (V1 is default)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B
```

All OpenAI API endpoints remain compatible.

## Feature Parity

You can check feature parity using the provided script:

```bash
python scripts/v0_v1_parity.py --verbose
```

This shows which V0 features are supported in V1.

### Fully Supported Features

- Continuous batching
- Prefix caching
- Speculative decoding
- LoRA adapters
- Multimodal models
- Tensor/Pipeline parallelism
- All quantization methods
- Streaming
- Structured output (JSON, regex, grammar)
- Async API
- OpenAI API compatibility
- Embeddings and pooling

### V0-Only Features (Deprecated)

- `best_of` parameter - Generate multiple, return best

## Troubleshooting

### Error: "V0 engine has been removed"

```python
RuntimeError: V0 engine has been removed as of v1.0.0...
```

**Solution**: Remove `use_v1=False` or `VLLM_USE_V1=0`. V1 is now required.

### Warning: "V0-only feature"

```python
DeprecationWarning: 'best_of' is a V0-only feature and is deprecated...
```

**Solution**: Remove usage of `best_of`. Use `n` parameter instead.

### Internal API Changes

If your code depends on internal APIs:

1. Check if the feature is available through public APIs
2. Use the compatibility layer temporarily
3. File an issue if you need functionality exposed publicly

## Timeline

| Version | Date | Action |
|---------|------|--------|
| v0.6.0 | Current | Deprecation warnings |
| v0.7.0 | +2 months | V1 becomes default |
| v0.8.0 | +4 months | Loud deprecation warnings |
| v1.0.0 | +6 months | V0 code removed |

## Getting Help

- **GitHub Issues**: Report migration problems at [vllm-project/vllm](https://github.com/vllm-project/vllm/issues)
- **Discord**: Join the vLLM community for real-time help
- **Documentation**: Check the latest docs for V1 features

## Migration Checklist

- [ ] Remove `use_v1=False` or `VLLM_USE_V1=0`
- [ ] Remove `best_of` parameter usage
- [ ] Update internal API access to use V1 patterns
- [ ] Run tests to verify functionality
- [ ] Check for deprecation warnings in logs
- [ ] Run `scripts/v0_v1_parity.py` to verify features

## Examples

### Basic Generation

```python
from vllm import LLM, SamplingParams

# Create LLM (V1 is default)
llm = LLM(model="meta-llama/Llama-3.1-8B")

# Generate
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["Hello, world!"], sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### Async API

```python
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import EngineArgs

# Create async engine (V1)
engine_args = EngineArgs(model="meta-llama/Llama-3.1-8B")
engine = AsyncLLMEngine.from_engine_args(engine_args)

# Use async API
async for output in engine.generate("Hello", sampling_params, request_id):
    print(output)
```

### With LoRA

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3.1-8B", enable_lora=True)

lora_request = LoRARequest("my_adapter", 1, "/path/to/adapter")
outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
```
