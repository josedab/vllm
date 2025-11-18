# RFC-0004: V0 Engine Deprecation Plan

**Status:** Draft
**Author:** Analysis Team
**Created:** November 18, 2025
**Commit Reference:** `67745d189fd981ee824bde35666a3737a962c031`

---

## Summary

Establish a clear deprecation timeline for the V0 engine, migrating all users to V1, and eventually removing V0 code to reduce maintenance burden and code complexity.

---

## Motivation

### Current State

vLLM maintains two parallel engine implementations:
- **V0 (Legacy):** Original engine architecture
- **V1 (Current):** Redesigned with 1.7x speedup, cleaner code

### Problems

1. **Dual Maintenance:** Every feature must be implemented twice
2. **Code Bloat:** ~100K LoC in V0 that duplicates V1
3. **Testing Overhead:** All tests run on both engines
4. **User Confusion:** Which engine to use?
5. **Bug Divergence:** Fixes may not be backported

### Evidence

From the codebase analysis:
- V0 code: ~100,000 LoC
- V1 code: ~60,000 LoC (cleaner)
- ~30% of PRs need changes in both

---

## Detailed Design

### 1. Deprecation Timeline

| Phase | Version | Date | Action |
|-------|---------|------|--------|
| Announce | v0.6.0 | Current | Warning when V0 is used |
| Default V1 | v0.7.0 | +2 months | V1 becomes default |
| V0 Deprecated | v0.8.0 | +4 months | Loud deprecation warnings |
| V0 Removed | v1.0.0 | +6 months | V0 code deleted |

### 2. Migration Assistance

#### Feature Parity Check

Ensure V1 has all V0 features:

```python
# scripts/v0_v1_parity.py

V0_FEATURES = [
    "continuous_batching",
    "prefix_caching",
    "speculative_decoding",
    "lora",
    "multimodal",
    # ...
]

def check_parity():
    missing = []
    for feature in V0_FEATURES:
        if not v1_has_feature(feature):
            missing.append(feature)
    return missing
```

Current gaps:
- Some edge cases in speculative decoding
- Certain attention backends

#### Migration Guide

```markdown
# Migrating from V0 to V1

## Quick Migration

```python
# Before (V0)
from vllm import LLM
llm = LLM(model="...")

# After (V1) - same API!
from vllm import LLM
llm = LLM(model="...", use_v1=True)  # or set VLLM_USE_V1=1
```

## Breaking Changes

1. **Async API changes:** See section 3
2. **Engine internals:** If you accessed private APIs
3. **Config options:** Some renamed

## Performance Improvements

V1 provides:
- 1.7x throughput improvement
- Lower latency
- Better memory efficiency
```

### 3. Deprecation Warnings

```python
# vllm/engine/__init__.py

def create_engine(use_v1=None, **kwargs):
    if use_v1 is None:
        use_v1 = os.getenv("VLLM_USE_V1", "1") == "1"

    if not use_v1:
        warnings.warn(
            "V0 engine is deprecated and will be removed in v1.0.0. "
            "Please migrate to V1 by setting use_v1=True or VLLM_USE_V1=1. "
            "See https://docs.vllm.ai/en/latest/migration/v0-to-v1.html",
            DeprecationWarning,
            stacklevel=2
        )
        return LLMEngineV0(**kwargs)

    return LLMEngineV1(**kwargs)
```

### 4. Compatibility Layer

For users who depend on V0 internals:

```python
# vllm/compat/v0.py

def get_scheduler_output_v0_format(v1_output):
    """Convert V1 scheduler output to V0 format."""
    return SchedulerOutputV0(
        scheduled_seq_groups=...,
        blocks_to_swap_in=...,
        blocks_to_swap_out=...,
    )
```

### 5. V0 Removal

After removal deadline:

```python
# vllm/engine/__init__.py

def create_engine(**kwargs):
    if kwargs.get("use_v1") is False:
        raise RuntimeError(
            "V0 engine has been removed. Please use V1 engine. "
            "See https://docs.vllm.ai/en/latest/migration/v0-removal.html"
        )
    return LLMEngineV1(**kwargs)
```

---

## Implementation Plan

### Phase 1: Announce (Week 1)
- Add deprecation warnings
- Create migration guide
- Blog post announcement

### Phase 2: Feature Parity (Week 2-4)
- Fill V1 feature gaps
- Fix V1-specific bugs
- Performance regression testing

### Phase 3: Default Switch (Week 5-6)
- Change default to V1
- Update documentation
- Update examples

### Phase 4: Loud Deprecation (Week 7-8)
- More prominent warnings
- Community outreach
- Track migration metrics

### Phase 5: Removal (Week 12+)
- Delete V0 code
- Clean up compatibility layers
- Update all documentation

---

## Backwards Compatibility

### Breaking Changes

1. **Internal APIs:** V0-specific internals gone
2. **Engine access:** `engine.scheduler` differs
3. **Config options:** Some V0-only options removed

### Non-breaking

1. **Public API:** `LLM`, `SamplingParams` unchanged
2. **API server:** Same endpoints
3. **Configuration:** Most options work

### Migration Support

- Compatibility shims for common patterns
- Detailed migration guide
- Community support during transition

---

## Alternatives Considered

### 1. Keep Both Forever

**Rejected:** Unsustainable maintenance burden

### 2. Faster Deprecation (3 months)

**Rejected:** Not enough time for enterprise users

### 3. Feature Branch V0

**Rejected:** Would still need maintenance

---

## Open Questions

1. How to handle users with V0 plugins/extensions?
2. Should we keep V0 in a separate package for legacy?
3. How to communicate to users not following updates?

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| V0 usage | 20% | 0% |
| Codebase reduction | 0 | 100K LoC |
| Test time reduction | 0 | 40% |
| Maintenance PRs | 30% | 0% (V0) |

---

## Effort Estimate

- **Total:** 15 dev-days (spread over 3 months)
- **Risk:** High (user breakage)
- **Required Approvals:** Core team, steering committee

---

## Rollback Strategy

- If critical issues found, extend deprecation period
- Keep V0 code in branch for emergency backport
- Provide docker image with V0 for legacy users

---

*RFC based on analysis of commit `67745d189fd981ee824bde35666a3737a962c031`*
