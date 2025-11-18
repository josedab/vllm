# RFC-0002: Error Message Enhancement

**Status:** Draft
**Author:** Analysis Team
**Created:** November 18, 2025
**Commit Reference:** `67745d189fd981ee824bde35666a3737a962c031`

---

## Summary

Improve vLLM's error messages to include actionable remediation steps, relevant configuration hints, and links to documentation, reducing user confusion and support burden.

---

## Motivation

### Current Problems

1. **Cryptic Errors:** Many errors expose internal details without context
2. **No Remediation:** Users don't know how to fix issues
3. **Missing Context:** Errors don't show relevant configuration
4. **Support Burden:** Common issues generate repeated questions

### Examples of Poor Error Messages

```python
# Current
RuntimeError: CUDA out of memory

# Better
vLLMMemoryError: Insufficient GPU memory for model configuration.
  Model: meta-llama/Llama-2-70b-hf
  Required: 142GB, Available: 80GB

  Solutions:
  1. Enable quantization: quantization="fp8"
  2. Reduce context: max_model_len=2048
  3. Use tensor parallelism: tensor_parallel_size=2

  See: https://docs.vllm.ai/en/latest/troubleshooting/memory.html
```

---

## Detailed Design

### 1. Error Exception Hierarchy

```python
# vllm/exceptions.py

class VLLMError(Exception):
    """Base class with enhanced formatting."""

    def __init__(self, message, *, context=None, solutions=None, docs_url=None):
        self.message = message
        self.context = context or {}
        self.solutions = solutions or []
        self.docs_url = docs_url
        super().__init__(self.format_message())

    def format_message(self):
        parts = [self.message]

        if self.context:
            parts.append("\nContext:")
            for key, value in self.context.items():
                parts.append(f"  {key}: {value}")

        if self.solutions:
            parts.append("\nSolutions:")
            for i, solution in enumerate(self.solutions, 1):
                parts.append(f"  {i}. {solution}")

        if self.docs_url:
            parts.append(f"\nDocumentation: {self.docs_url}")

        return "\n".join(parts)


class MemoryError(VLLMError):
    """GPU memory related errors."""
    pass


class ConfigurationError(VLLMError):
    """Configuration validation errors."""
    pass


class ModelLoadError(VLLMError):
    """Model loading errors."""
    pass
```

### 2. Common Error Improvements

#### OOM Errors

```python
# vllm/v1/worker/gpu_model_runner.py

def allocate_kv_cache(self, num_blocks):
    try:
        cache = torch.empty(...)
    except torch.cuda.OutOfMemoryError:
        raise MemoryError(
            "Insufficient GPU memory for KV cache allocation",
            context={
                "Model": self.model_config.model,
                "Requested blocks": num_blocks,
                "Block size": self.block_size,
                "GPU memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
                "Current utilization": f"{self.gpu_memory_utilization:.0%}",
            },
            solutions=[
                f"Reduce gpu_memory_utilization (current: {self.gpu_memory_utilization})",
                f"Reduce max_num_seqs (current: {self.max_num_seqs})",
                f"Reduce max_model_len (current: {self.max_model_len})",
                "Enable KV cache quantization: kv_cache_dtype='fp8_e4m3'",
            ],
            docs_url="https://docs.vllm.ai/en/latest/troubleshooting/memory.html"
        )
```

#### Model Loading Errors

```python
# vllm/model_executor/model_loader/loader.py

def load_model(self, model_config):
    try:
        model = AutoModelForCausalLM.from_pretrained(...)
    except OSError as e:
        if "not found" in str(e).lower():
            raise ModelLoadError(
                f"Model '{model_config.model}' not found",
                context={
                    "Model": model_config.model,
                    "Tokenizer": model_config.tokenizer,
                },
                solutions=[
                    "Check the model name/path is correct",
                    "Ensure you're logged in: huggingface-cli login",
                    "For private models, set HF_TOKEN environment variable",
                    "Check network connectivity to HuggingFace",
                ],
                docs_url="https://docs.vllm.ai/en/latest/models/loading.html"
            )
        raise
```

#### Configuration Errors

```python
# vllm/config/validation.py

def validate_parallel_config(config):
    if config.tensor_parallel_size > torch.cuda.device_count():
        raise ConfigurationError(
            "Tensor parallel size exceeds available GPUs",
            context={
                "tensor_parallel_size": config.tensor_parallel_size,
                "Available GPUs": torch.cuda.device_count(),
                "GPU devices": [torch.cuda.get_device_name(i)
                               for i in range(torch.cuda.device_count())],
            },
            solutions=[
                f"Reduce tensor_parallel_size to {torch.cuda.device_count()} or less",
                "Add more GPUs to your system",
                "Use pipeline parallelism across nodes instead",
            ],
            docs_url="https://docs.vllm.ai/en/latest/serving/distributed.html"
        )
```

### 3. Error Context Capture

Automatically capture relevant context:

```python
# vllm/utils/error_context.py

class ErrorContext:
    """Captures context for error reporting."""

    @staticmethod
    def get_system_context():
        return {
            "vLLM version": vllm.__version__,
            "PyTorch version": torch.__version__,
            "CUDA version": torch.version.cuda,
            "GPU": torch.cuda.get_device_name(0),
            "GPU memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
        }

    @staticmethod
    def get_config_context(config):
        return {
            "Model": config.model,
            "TP size": config.tensor_parallel_size,
            "PP size": config.pipeline_parallel_size,
            "Max seqs": config.max_num_seqs,
            "Max len": config.max_model_len,
        }
```

### 4. Common Issues Database

```python
# vllm/utils/known_issues.py

KNOWN_ISSUES = {
    "CUDA_ERROR_OUT_OF_MEMORY": {
        "pattern": r"CUDA out of memory",
        "message": "GPU memory exhausted",
        "solutions": [...],
        "docs": "memory.html",
    },
    "NCCL_TIMEOUT": {
        "pattern": r"NCCL.*timeout",
        "message": "Distributed communication timeout",
        "solutions": [...],
        "docs": "distributed.html",
    },
}

def enhance_error(error):
    """Enhance error with known issue information."""
    error_str = str(error)
    for issue_id, issue in KNOWN_ISSUES.items():
        if re.search(issue["pattern"], error_str):
            return VLLMError(
                issue["message"],
                solutions=issue["solutions"],
                docs_url=f"https://docs.vllm.ai/en/latest/troubleshooting/{issue['docs']}"
            )
    return error
```

---

## Example Usage

### Before

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 79.35 GiB total capacity; 77.38 GiB already allocated; 1.31 GiB free; 77.49 GiB reserved in total by PyTorch)
```

### After

```
vLLMMemoryError: Insufficient GPU memory for model execution

Context:
  Model: meta-llama/Llama-2-70b-hf
  Operation: KV cache allocation
  Required: 2.00 GB
  Available: 1.31 GB
  GPU: NVIDIA A100 80GB
  Current utilization: 97%

Solutions:
  1. Reduce max_num_seqs from 256 to 128
  2. Reduce max_model_len from 4096 to 2048
  3. Enable quantization: quantization="fp8"
  4. Enable KV cache quantization: kv_cache_dtype="fp8_e4m3"
  5. Increase GPU memory utilization is not recommended (already at 97%)

Documentation: https://docs.vllm.ai/en/latest/troubleshooting/memory.html
```

---

## Implementation Plan

### Phase 1: Exception Hierarchy (Day 1)
- Create VLLMError base class
- Add context/solutions support
- Migrate critical errors

### Phase 2: OOM Errors (Day 2)
- Enhance all memory-related errors
- Add memory estimation
- Provide specific solutions

### Phase 3: Config/Model Errors (Day 3)
- Configuration validation errors
- Model loading errors
- Distributed setup errors

### Phase 4: Documentation (Day 4)
- Create troubleshooting guide
- Link from errors
- Add search by error code

---

## Backwards Compatibility

- Error types remain catchable
- String representation enhanced but compatible
- New attributes are optional

---

## Alternatives Considered

### 1. Error Codes Only

**Rejected:** Requires users to look up codes; less user-friendly

### 2. Logging Instead of Exceptions

**Rejected:** Exceptions are the expected pattern

---

## Open Questions

1. Should we include telemetry for common errors?
2. How verbose should context be by default?
3. Should solutions be prioritized by likelihood?

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Error-related issues | 50/month | 20/month |
| Avg time to resolution | 2 hours | 15 min |
| User satisfaction | - | 4/5 |

---

## Effort Estimate

- **Total:** 4 dev-days
- **Risk:** Low
- **Required Approvals:** Core team

---

*RFC based on analysis of commit `67745d189fd981ee824bde35666a3737a962c031`*
