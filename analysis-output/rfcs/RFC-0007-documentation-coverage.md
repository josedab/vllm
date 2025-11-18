# RFC-0007: Documentation Coverage Improvement

**Status:** Draft
**Author:** Analysis Team
**Created:** November 18, 2025
**Commit Reference:** `67745d189fd981ee824bde35666a3737a962c031`

---

## Summary

Systematically improve vLLM's documentation coverage from ~50% to 90%, focusing on API reference completeness, practical examples, and troubleshooting guides to reduce user friction and support burden.

---

## Motivation

### Current State

vLLM's documentation has significant gaps:

1. **API Reference:** ~50% of public APIs documented
2. **Docstrings:** ~50% coverage in codebase
3. **Examples:** Limited real-world use cases
4. **Troubleshooting:** Common issues not documented
5. **Architecture:** Limited internal documentation

### Evidence from Analysis

```python
# Many public methods lack docstrings
class LLM:
    def generate(self, prompts, sampling_params=None, ...):
        # No docstring explaining parameters, return types, exceptions
        ...

# Configuration options not fully documented
class SamplingParams:
    # 50+ parameters, many undocumented
    temperature: float  # What's the valid range? Default?
    top_p: float        # How does it interact with top_k?
```

### User Pain Points

- "What parameters does this method accept?"
- "What's the difference between these two options?"
- "How do I do X in production?"
- "Why is this error happening?"

### Support Burden

- 40% of GitHub issues are documentation-related
- Repeated questions on Slack
- Users struggle with configuration

---

## Detailed Design

### 1. Documentation Structure

Reorganize documentation into clear sections:

```
docs/
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── first-deployment.md
├── user-guide/
│   ├── offline-inference.md
│   ├── api-server.md
│   ├── distributed.md
│   ├── quantization.md
│   └── multi-modal.md
├── api-reference/
│   ├── llm.md
│   ├── sampling-params.md
│   ├── engine.md
│   └── config.md
├── tutorials/
│   ├── production-deployment.md
│   ├── performance-tuning.md
│   ├── custom-models.md
│   └── monitoring.md
├── troubleshooting/
│   ├── common-errors.md
│   ├── memory-issues.md
│   ├── performance-issues.md
│   └── distributed-issues.md
├── architecture/
│   ├── overview.md
│   ├── scheduler.md
│   ├── paged-attention.md
│   └── distributed.md
└── contributing/
    ├── development-setup.md
    ├── code-style.md
    └── testing.md
```

### 2. API Reference Generation

Auto-generate API documentation from docstrings:

```python
# vllm/entrypoints/llm.py

class LLM:
    """High-level interface for vLLM inference.

    This class provides a simple interface for running inference with
    large language models. It handles model loading, tokenization,
    scheduling, and generation.

    Args:
        model: HuggingFace model name or path to local model.
        tokenizer: Tokenizer name/path. Defaults to model name.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        dtype: Data type for model weights. Options: "auto", "float16",
            "bfloat16", "float32".
        quantization: Quantization method. Options: None, "awq", "gptq",
            "fp8", "squeezellm".
        max_model_len: Maximum sequence length. Defaults to model's max.
        gpu_memory_utilization: Fraction of GPU memory for model and cache.
            Default: 0.9.
        **kwargs: Additional arguments passed to engine configuration.

    Example:
        Basic usage::

            from vllm import LLM, SamplingParams

            llm = LLM(model="meta-llama/Llama-2-7b-hf")
            outputs = llm.generate(
                ["Hello, world!"],
                SamplingParams(max_tokens=100)
            )
            print(outputs[0].outputs[0].text)

        With tensor parallelism::

            llm = LLM(
                model="meta-llama/Llama-2-70b-hf",
                tensor_parallel_size=8
            )

        With quantization::

            llm = LLM(
                model="meta-llama/Llama-2-7b-hf",
                quantization="fp8"
            )

    Raises:
        ValueError: If model not found or invalid configuration.
        RuntimeError: If CUDA not available or insufficient memory.

    See Also:
        - :class:`SamplingParams`: Generation parameters
        - :class:`RequestOutput`: Output format
        - :doc:`/user-guide/offline-inference`: Complete guide
    """

    def generate(
        self,
        prompts: Union[str, List[str], List[Dict]],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
    ) -> List[RequestOutput]:
        """Generate completions for the given prompts.

        Args:
            prompts: Input prompts. Can be:
                - Single string: "Hello, world!"
                - List of strings: ["Hello", "Hi there"]
                - List of dicts for chat: [{"role": "user", "content": "Hi"}]
            sampling_params: Parameters controlling generation.
                If None, uses default parameters.
            use_tqdm: Show progress bar. Default: True.
            lora_request: LoRA adapter to use for generation.

        Returns:
            List of RequestOutput objects, one per prompt.
            Each contains:
                - prompt: Original prompt
                - outputs: List of generated sequences
                - finished: Whether generation completed

        Example:
            Generate with custom parameters::

                params = SamplingParams(
                    temperature=0.8,
                    top_p=0.95,
                    max_tokens=200,
                    stop=["\\n\\n"]
                )
                outputs = llm.generate(["Write a poem:"], params)

            Chat completion::

                messages = [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello!"}
                ]
                outputs = llm.chat([messages])

        Raises:
            ValueError: If prompts is empty or invalid format.
            RuntimeError: If engine is not initialized.

        Note:
            For streaming generation, use the API server or
            AsyncLLMEngine directly.
        """
        ...
```

### 3. Comprehensive Examples

Add practical, runnable examples:

```python
# examples/production_deployment.py
"""
Production Deployment Example

This example shows how to deploy vLLM for production use with:
- Optimized configuration for throughput
- Health checks and monitoring
- Graceful shutdown handling
"""

import signal
import sys
from vllm import LLM, SamplingParams

def create_production_llm():
    """Create LLM configured for production."""
    return LLM(
        model="meta-llama/Llama-2-7b-hf",

        # Memory optimization
        gpu_memory_utilization=0.95,
        max_model_len=4096,

        # Throughput optimization
        max_num_seqs=256,
        enable_prefix_caching=True,

        # Reliability
        disable_log_requests=False,  # Keep for debugging
    )

def health_check(llm):
    """Verify LLM is responding correctly."""
    try:
        output = llm.generate(
            ["Health check"],
            SamplingParams(max_tokens=10)
        )
        return len(output) > 0
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def main():
    llm = create_production_llm()

    # Verify health
    if not health_check(llm):
        sys.exit(1)

    print("LLM ready for production")

    # Your application logic here
    ...

if __name__ == "__main__":
    main()
```

### 4. Troubleshooting Guides

Create comprehensive troubleshooting documentation:

```markdown
# Memory Issues Troubleshooting

## Out of Memory (OOM) Errors

### Symptoms
- `RuntimeError: CUDA out of memory`
- `torch.cuda.OutOfMemoryError`
- Process killed by OOM killer

### Diagnosis

1. **Check memory usage**
   ```python
   import torch
   print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
   ```

2. **Monitor via metrics**
   ```bash
   curl http://localhost:8000/metrics | grep kv_cache
   ```

### Solutions

#### Solution 1: Reduce GPU memory utilization
```python
llm = LLM(
    model="...",
    gpu_memory_utilization=0.85  # Default is 0.9
)
```

#### Solution 2: Reduce maximum sequence length
```python
llm = LLM(
    model="...",
    max_model_len=2048  # Reduce from model's max
)
```

#### Solution 3: Reduce concurrent requests
```python
llm = LLM(
    model="...",
    max_num_seqs=128  # Default is 256
)
```

#### Solution 4: Enable quantization
```python
# FP8 quantization (recommended for most cases)
llm = LLM(
    model="...",
    quantization="fp8"
)

# Or use pre-quantized model
llm = LLM(model="TheBloke/Llama-2-7B-AWQ")
```

#### Solution 5: Enable KV cache quantization
```python
llm = LLM(
    model="...",
    kv_cache_dtype="fp8_e4m3"  # Reduces KV cache by 50%
)
```

#### Solution 6: Use tensor parallelism
```python
# Distribute across multiple GPUs
llm = LLM(
    model="...",
    tensor_parallel_size=2  # or 4, 8
)
```

### Prevention

1. Use auto-tuning to find optimal configuration
2. Monitor KV cache usage in production
3. Set appropriate max_model_len for your use case
4. Test with production-like workloads before deployment

### Related
- [Performance Tuning Guide](/tutorials/performance-tuning)
- [Memory Estimation](/architecture/memory)
- [Quantization Guide](/user-guide/quantization)
```

### 5. Architecture Documentation

Document internal architecture for contributors:

```markdown
# Scheduler Architecture

## Overview

The scheduler is responsible for deciding which requests to execute
in each iteration. It manages request queues, allocates KV cache blocks,
and handles preemption.

## Key Files

- `vllm/v1/core/sched/scheduler.py` - Main scheduler logic
- `vllm/v1/core/kv_cache_manager.py` - KV cache allocation
- `vllm/v1/request.py` - Request data structure

## Request States

```
WAITING → RUNNING → FINISHED
            ↓
        PREEMPTED
            ↓
         WAITING
```

## Scheduling Algorithm

### Phase 1: Schedule Running Requests

For each running request:
1. Calculate tokens to generate this iteration
2. Try to allocate KV cache blocks
3. If allocation fails, preempt lower-priority requests
4. If still fails, preempt this request

### Phase 2: Schedule Waiting Requests

For each waiting request (by priority):
1. Check for prefix cache hits
2. Try to allocate KV cache blocks
3. If success, move to running
4. If fail, stop scheduling (no more capacity)

## Key Design Decisions

### Why token-budget based?

The scheduler uses a token budget rather than request count because:
- Different prompts have different lengths
- Chunked prefill needs partial scheduling
- Enables fine-grained memory control

### Why FCFS with priorities?

- Simple and predictable
- Avoids starvation
- Priority override for important requests

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_num_seqs | 256 | Max concurrent sequences |
| max_num_batched_tokens | 8192 | Max tokens per iteration |
| enable_chunked_prefill | False | Break up long prompts |

## Metrics

- `vllm:num_requests_running` - Current running requests
- `vllm:num_requests_waiting` - Queue depth
- `vllm:num_preemptions` - Preemption count

## See Also

- [PagedAttention Architecture](/architecture/paged-attention)
- [KV Cache Manager](/architecture/kv-cache)
```

### 6. Documentation Tooling

Set up automated documentation checks:

```python
# tools/check_docstrings.py

import ast
import sys
from pathlib import Path

def check_module(path):
    """Check docstring coverage for a module."""
    with open(path) as f:
        tree = ast.parse(f.read())

    total = 0
    documented = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name.startswith('_'):
                continue  # Skip private
            total += 1
            if ast.get_docstring(node):
                documented += 1

    return documented, total

def main():
    vllm_path = Path("vllm")
    total_documented = 0
    total_items = 0

    for py_file in vllm_path.rglob("*.py"):
        if "test" in str(py_file):
            continue
        documented, items = check_module(py_file)
        total_documented += documented
        total_items += items

    coverage = total_documented / total_items * 100 if total_items else 0
    print(f"Docstring coverage: {coverage:.1f}%")
    print(f"Documented: {total_documented}/{total_items}")

    if coverage < 90:
        print("Coverage below 90% threshold")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 7. Documentation CI

Add documentation checks to CI:

```yaml
# .github/workflows/docs.yml

name: Documentation

on: [push, pull_request]

jobs:
  check-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Check docstring coverage
        run: python tools/check_docstrings.py

      - name: Build docs
        run: |
          pip install -r requirements/docs.txt
          mkdocs build --strict

      - name: Check for broken links
        run: |
          pip install linkchecker
          linkchecker site/
```

---

## Implementation Plan

### Phase 1: Infrastructure (Week 1-2)
- Set up docstring checker
- Add CI integration
- Create documentation templates

### Phase 2: API Reference (Week 3-6)
- Document all public classes
- Document all public methods
- Add examples to each

### Phase 3: Guides and Tutorials (Week 7-10)
- User guide sections
- Tutorials with examples
- Troubleshooting guides

### Phase 4: Architecture Docs (Week 11-14)
- Internal architecture
- Design decisions
- Contributor guides

### Phase 5: Review and Polish (Week 15-16)
- Community review
- Fix gaps
- Final polish

---

## Backwards Compatibility

- No code changes required
- Documentation is additive
- Existing docs preserved

---

## Alternatives Considered

### 1. Wiki Instead of Docs

**Rejected:** Less integrated, harder to version

### 2. External Documentation Service

**Rejected:** Should be in repo for PRs

### 3. Video Tutorials Only

**Rejected:** Text is searchable and faster to update

---

## Open Questions

1. Should we require docstrings in PR reviews?
2. How to handle documentation for experimental features?
3. Should we translate documentation?

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Docstring coverage | ~50% | 90% |
| API reference completeness | ~40% | 95% |
| Troubleshooting guides | 5 | 20+ |
| Examples | 10 | 50+ |
| Documentation issues | 40/month | 10/month |

---

## Effort Estimate

- **Total:** 20 dev-days (spread over 4 months)
- **Risk:** Low
- **Required Approvals:** Core team

---

## Maintenance Plan

1. **PR Requirements:** All public API changes need docstrings
2. **CI Enforcement:** Block PRs that reduce coverage
3. **Quarterly Review:** Check for outdated content
4. **Community Contributions:** Welcome documentation PRs

---

*RFC based on analysis of commit `67745d189fd981ee824bde35666a3737a962c031`*
