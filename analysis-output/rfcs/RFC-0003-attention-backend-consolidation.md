# RFC-0003: Attention Backend Consolidation

**Status:** Draft
**Author:** Analysis Team
**Created:** November 18, 2025
**Commit Reference:** `67745d189fd981ee824bde35666a3737a962c031`

---

## Summary

Consolidate vLLM's 33+ attention backend implementations into a core set of 8-10 well-maintained backends, reducing code duplication, maintenance burden, and user confusion.

---

## Motivation

### Current State

vLLM has accumulated 33+ attention backend implementations:

```
vllm/attention/backends/
├── abstract.py
├── blocksparse_attn.py
├── flash_attn.py
├── flashinfer.py
├── ipex_attn.py
├── pallas.py
├── rocm_flash_attn.py
├── torch_sdpa.py
├── triton.py
├── xformers.py
└── ... (23 more)
```

### Problems

1. **Maintenance Burden:** Each backend needs updates when core changes
2. **Code Duplication:** Similar implementations across backends
3. **User Confusion:** Which backend to choose?
4. **Testing Overhead:** Testing all combinations
5. **Inconsistent Features:** Some backends lack features (prefix caching, etc.)

### Evidence

- 20% of PRs touch attention backends
- Regular bugs from inconsistent implementations
- User questions: "Which attention backend should I use?"

---

## Detailed Design

### 1. Core Backend Set

Retain only the essential backends:

| Backend | Platform | Use Case | Priority |
|---------|----------|----------|----------|
| FlashAttention | CUDA | Default NVIDIA | P0 |
| FlashInfer | CUDA | Advanced features | P0 |
| Triton | CUDA/ROCm | Custom, portable | P1 |
| xFormers | CUDA | Legacy support | P1 |
| ROCm Flash | ROCm | Default AMD | P0 |
| Pallas | TPU | Default TPU | P0 |
| IPEX | XPU | Default Intel | P1 |
| torch.sdpa | All | Fallback | P1 |

### 2. Deprecate and Remove

| Backend | Action | Reason |
|---------|--------|--------|
| BlockSparse | Remove | Rarely used, unmaintained |
| Custom variants | Merge | Consolidate into main |
| Experimental | Remove or Mature | Clear status |

### 3. Unified Interface

Ensure all core backends implement full interface:

```python
# vllm/attention/backends/base.py

class AttentionBackend(ABC):
    """Base class for attention backends."""

    @abstractmethod
    def get_name(self) -> str:
        """Backend identifier."""
        pass

    @abstractmethod
    def supports_paged_attention(self) -> bool:
        pass

    @abstractmethod
    def supports_prefix_caching(self) -> bool:
        pass

    @abstractmethod
    def supports_sliding_window(self) -> bool:
        pass

    @abstractmethod
    def get_supported_head_sizes(self) -> List[int]:
        pass

    @abstractmethod
    def forward_prefill(self, query, key, value, attn_metadata):
        pass

    @abstractmethod
    def forward_decode(self, query, kv_cache, attn_metadata):
        pass

    # Feature matrix
    CAPABILITIES = {
        "paged_attention": True,
        "prefix_caching": True,
        "sliding_window": True,
        "speculative_decoding": True,
    }
```

### 4. Automatic Selection

Improve automatic backend selection:

```python
# vllm/attention/selector.py

def get_optimal_backend(
    model_config: ModelConfig,
    platform: Platform,
    features_required: Set[str],
) -> Type[AttentionBackend]:
    """Select best backend for configuration."""

    candidates = []

    for backend in BACKENDS:
        # Check platform
        if not backend.supports_platform(platform):
            continue

        # Check features
        if not features_required.issubset(backend.CAPABILITIES):
            continue

        # Check head size
        if model_config.head_size not in backend.get_supported_head_sizes():
            continue

        candidates.append(backend)

    # Rank by performance
    return rank_by_performance(candidates, model_config)[0]
```

### 5. Migration Path

For deprecated backends:

```python
# vllm/attention/backends/deprecated.py

class BlockSparseAttention(AttentionBackend):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "BlockSparseAttention is deprecated and will be removed in v0.7.0. "
            "Use FlashAttention with enable_sparse=True instead.",
            DeprecationWarning
        )
        # Delegate to replacement
        self._impl = FlashAttention(*args, enable_sparse=True, **kwargs)
```

---

## Implementation Plan

### Phase 1: Audit (Week 1)
- Catalog all backends
- Document features and gaps
- Identify usage patterns

### Phase 2: Interface Unification (Week 1-2)
- Define complete interface
- Add missing methods to core backends
- Add capability declarations

### Phase 3: Consolidation (Week 2-3)
- Merge similar implementations
- Add deprecation warnings
- Update tests

### Phase 4: Documentation (Week 3)
- Backend selection guide
- Feature comparison table
- Migration guides

---

## Backwards Compatibility

### Deprecation Timeline

1. **v0.6.0:** Add warnings to deprecated backends
2. **v0.7.0:** Remove deprecated backends
3. **v0.6.x:** Provide automatic migration where possible

### Migration Support

```python
# Automatic fallback
if backend_name in DEPRECATED_BACKENDS:
    logger.warning(f"{backend_name} is deprecated, using {replacement}")
    backend_name = DEPRECATED_BACKENDS[backend_name]
```

---

## Alternatives Considered

### 1. Keep All Backends

**Rejected:** Unsustainable maintenance burden

### 2. Single Backend

**Rejected:** Different platforms need different implementations

### 3. Plugin System

**Considered:** Could move non-core backends to plugins
- Pro: Reduces core maintenance
- Con: Complicates dependency management

---

## Open Questions

1. How to handle community-contributed backends?
2. Should experimental backends be in a separate package?
3. How to measure performance regressions?

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Backends in core | 33+ | 8-10 |
| LoC in attention/ | ~20,000 | ~12,000 |
| Test combinations | 100+ | 30 |
| Backend-related bugs | 10/month | 3/month |

---

## Effort Estimate

- **Total:** 10 dev-days
- **Risk:** Medium (potential regressions)
- **Required Approvals:** Core team, major users

---

## Rollback Strategy

- Keep deprecated code for one major version
- Feature flag to force old implementations
- Performance regression tests as gate

---

*RFC based on analysis of commit `67745d189fd981ee824bde35666a3737a962c031`*
