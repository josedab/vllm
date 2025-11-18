# RFC-0001: Configuration Simplification

**Status:** Draft
**Author:** Analysis Team
**Created:** November 18, 2025
**Commit Reference:** `67745d189fd981ee824bde35666a3737a962c031`

---

## Summary

Reduce vLLM's configuration complexity by consolidating 230+ environment variables into logical groups, providing better defaults, and creating a configuration profile system for common use cases.

---

## Motivation

### Current Problems

1. **Overwhelming Options:** 230+ environment variables with unclear interactions
2. **Poor Defaults:** Users must tune many settings for reasonable performance
3. **No Guidance:** Users don't know which options matter for their use case
4. **Hidden Dependencies:** Some options interact in non-obvious ways

### User Feedback

- "I don't know where to start with configuration"
- "Setting X breaks setting Y but there's no error message"
- "What should I set for a production chat bot?"

### Evidence

```python
# Current: User must figure this out themselves
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.9,
    max_num_seqs=256,
    max_model_len=4096,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    compilation_config={"level": "PIECEWISE"},
    # ... 20 more options
)
```

---

## Detailed Design

### 1. Configuration Profiles

Introduce predefined profiles for common use cases:

```python
from vllm import LLM, ConfigProfile

# Simple: Use a profile
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    profile=ConfigProfile.HIGH_THROUGHPUT  # or LATENCY_OPTIMIZED, MEMORY_CONSTRAINED
)

# Advanced: Override specific settings
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    profile=ConfigProfile.HIGH_THROUGHPUT,
    max_model_len=8192  # Override just this
)
```

**Profile Definitions:**

```python
# vllm/config/profiles.py

class ConfigProfile(Enum):
    HIGH_THROUGHPUT = "high_throughput"
    LOW_LATENCY = "low_latency"
    MEMORY_CONSTRAINED = "memory_constrained"
    DEVELOPMENT = "development"

PROFILES = {
    ConfigProfile.HIGH_THROUGHPUT: {
        "max_num_seqs": 256,
        "gpu_memory_utilization": 0.95,
        "enable_prefix_caching": True,
        "compilation_level": "PIECEWISE",
    },
    ConfigProfile.LOW_LATENCY: {
        "max_num_seqs": 32,
        "enable_chunked_prefill": True,
        "compilation_level": "FULL",
        "speculative_method": "ngram",
    },
    # ...
}
```

### 2. Environment Variable Consolidation

Reduce from 230+ to ~50 documented variables:

**Tier 1 (Essential, ~10):**
- `VLLM_MODEL`
- `VLLM_TENSOR_PARALLEL_SIZE`
- `VLLM_GPU_MEMORY_UTILIZATION`
- `VLLM_MAX_MODEL_LEN`
- `VLLM_QUANTIZATION`
- `VLLM_PROFILE`
- `VLLM_LOGGING_LEVEL`
- `VLLM_PORT`
- `VLLM_HOST`

**Tier 2 (Common, ~20):**
- Performance tuning
- Caching
- Distributed settings

**Tier 3 (Advanced, ~20):**
- Compilation details
- Hardware-specific
- Debugging

**Deprecated (~160):**
- Move to internal-only
- Add deprecation warnings

### 3. Improved Defaults

Update defaults based on production experience:

```python
# Current defaults vs Proposed
CURRENT = {
    "enable_prefix_caching": False,  # Should be True
    "gpu_memory_utilization": 0.9,   # Keep
    "max_num_seqs": 256,             # Keep
}

PROPOSED = {
    "enable_prefix_caching": True,   # Free performance
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 256,
    "compilation_level": "PIECEWISE",  # Reasonable default
}
```

### 4. Configuration Validation

Add comprehensive validation with actionable messages:

```python
def validate_config(config):
    errors = []

    # Check interactions
    if config.tensor_parallel_size > 1 and config.compilation_level == "FULL":
        errors.append(
            "FULL compilation is not supported with tensor_parallel_size > 1. "
            "Use PIECEWISE instead or reduce tensor_parallel_size to 1."
        )

    # Check resource requirements
    estimated_memory = estimate_memory(config)
    if estimated_memory > available_memory:
        errors.append(
            f"Estimated memory {estimated_memory}GB exceeds available {available_memory}GB. "
            f"Reduce max_model_len or max_num_seqs, or enable quantization."
        )

    if errors:
        raise ConfigurationError("\n".join(errors))
```

### 5. Configuration Documentation

Auto-generate documentation from config classes:

```python
@config
class ModelConfig:
    """Configuration for the model.

    Attributes:
        model: HuggingFace model name or path
        max_model_len: Maximum sequence length (default: model's max)
    """
    model: str = Field(description="HuggingFace model name or path")
    max_model_len: Optional[int] = Field(
        default=None,
        description="Maximum sequence length. Defaults to model's configured max.",
        tier="essential"
    )
```

---

## Example Usage

### Before

```python
# User must research each option
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_USE_V1"] = "1"
# ... 10 more env vars

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.9,
    max_num_seqs=256,
    # ... 15 more options
)
```

### After

```python
# Simple and clear
from vllm import LLM, ConfigProfile

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    profile=ConfigProfile.HIGH_THROUGHPUT
)
```

---

## Implementation Plan

### Phase 1: Profiles (Week 1)
- Define profile system
- Implement HIGH_THROUGHPUT, LOW_LATENCY profiles
- Add profile CLI argument

### Phase 2: Consolidation (Week 1-2)
- Categorize all env vars into tiers
- Add deprecation warnings to Tier 3+
- Update documentation

### Phase 3: Validation (Week 2-3)
- Implement comprehensive validation
- Add interaction checking
- Improve error messages

### Phase 4: Documentation (Week 3)
- Auto-generate config docs
- Write profile selection guide
- Update examples

---

## Backwards Compatibility

### Deprecation Strategy

1. **Phase 1 (Immediate):** Log warnings for deprecated env vars
2. **Phase 2 (3 months):** Loud warnings with alternatives
3. **Phase 3 (6 months):** Errors with migration guide

### Example Migration

```python
# Old way (deprecated)
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

# New way
llm = LLM(model="...", attention_backend="FLASH_ATTN")
# Or let profile choose automatically
llm = LLM(model="...", profile=ConfigProfile.HIGH_THROUGHPUT)
```

---

## Alternatives Considered

### 1. Configuration File Only

**Rejected:** Env vars are important for containers/K8s

### 2. Keep All Options

**Rejected:** User experience too poor, support burden too high

### 3. Automatic Detection Only

**Rejected:** Users need control, detection not always accurate

---

## Open Questions

1. Should profiles be user-customizable?
2. How to handle hardware-specific defaults (A100 vs H100)?
3. Should we provide a configuration wizard?

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Documented env vars | 230+ | 50 |
| Lines to basic usage | 20+ | 3 |
| Config-related issues | 30/month | 10/month |
| Time to first working config | 30 min | 5 min |

---

## Effort Estimate

- **Total:** 3 dev-days
- **Risk:** Low
- **Required Approvals:** Core team

---

*RFC based on analysis of commit `67745d189fd981ee824bde35666a3737a962c031`*
