# V0 Engine Removal Notice

As of vLLM v1.0.0, the V0 engine has been completely removed. This document provides guidance for users who encounter errors related to V0 removal.

## What Was Removed

- V0 engine implementation (~100K lines of code)
- V0-specific scheduler
- V0-specific block manager
- `best_of` sampling parameter support
- `use_v1=False` option
- `VLLM_USE_V1=0` environment variable support

## Error Messages

### "V0 engine has been removed"

If you see this error:

```
RuntimeError: V0 engine has been removed as of v1.0.0.
Please use V1 engine. See https://docs.vllm.ai/en/latest/migration/v0-removal.html
```

This means your code is trying to use the V0 engine which no longer exists.

## Required Actions

### 1. Remove V0 Engine Selection

```python
# Remove this
llm = LLM(model="...", use_v1=False)  # ERROR

# Use this instead
llm = LLM(model="...")  # V1 is the only option
```

### 2. Remove Environment Variable

```bash
# Remove from your environment
unset VLLM_USE_V1
# or
export VLLM_USE_V1=1
```

### 3. Remove `best_of` Parameter

```python
# Remove this
params = SamplingParams(n=1, best_of=5)  # ERROR

# Use n parameter instead
params = SamplingParams(n=5)
# Then select the best output yourself
```

## Why V0 Was Removed

1. **Maintenance burden**: Every feature needed dual implementation
2. **Code complexity**: ~100K lines of duplicate code
3. **Testing overhead**: Tests ran on both engines
4. **User confusion**: Which engine to use?
5. **Performance**: V1 is 1.7x faster

## Benefits of V1

- **1.7x throughput improvement**
- **Lower latency**
- **Better memory efficiency**
- **Cleaner, more maintainable code**
- **Faster feature development**

## Legacy Support

For users who absolutely need V0:

### Docker Image

A legacy Docker image with V0 support is available:

```bash
docker pull vllm/vllm-openai:v0-legacy
```

**Warning**: This image will not receive updates or security patches.

### Archived Branch

V0 code is preserved in an archived branch:

```bash
git checkout v0-archive
```

**Warning**: This branch is unmaintained.

## Migration Resources

- [V0 to V1 Migration Guide](v0-to-v1.md)
- [Feature Parity Script](../../scripts/v0_v1_parity.py)
- [Compatibility Layer](../../vllm/compat/v0.py)

## Compatibility Layer

For gradual migration, a compatibility layer is available:

```python
from vllm.compat.v0 import (
    get_scheduler_output_v0_format,
    wrap_v1_engine_with_v0_interface,
)
```

**Note**: This compatibility layer is deprecated and will be removed.

## Common Migration Issues

### Issue: Code Depends on Scheduler Access

**Before (V0)**:
```python
scheduler = engine.scheduler
running = scheduler.running
```

**After (V1)**:
```python
# Direct scheduler access is not available
# Use public APIs instead
num_running = engine.get_num_unfinished_requests()
```

### Issue: Block Manager Access

**Before (V0)**:
```python
block_manager = engine.scheduler.block_manager
free_blocks = block_manager.get_num_free_gpu_blocks()
```

**After (V1)**:
```python
# Block manager is internal to engine_core
# No direct access provided
# If you need this information, file a feature request
```

### Issue: Custom Scheduler Policy

If you implemented a custom scheduler policy for V0, you'll need to:

1. Review V1's scheduler architecture
2. Implement your policy for V1's scheduler
3. Consider contributing upstream if broadly useful

## Getting Help

- **GitHub Issues**: [vllm-project/vllm](https://github.com/vllm-project/vllm/issues)
- **Discord**: Join the vLLM community
- **Documentation**: [docs.vllm.ai](https://docs.vllm.ai)

## FAQ

### Q: Why was V0 removed without a longer deprecation period?

A: V0 was deprecated for 6 months with multiple warnings. Maintaining two engines was unsustainable for the project.

### Q: Can I get V0 back?

A: No. V0 is not maintained. Use the legacy Docker image only if absolutely necessary, but plan to migrate.

### Q: My code worked before, why doesn't it now?

A: Check for:
- `use_v1=False` in your code
- `VLLM_USE_V1=0` in your environment
- `best_of` parameter in SamplingParams
- Direct access to internal engine attributes

### Q: I need a feature that was in V0 but not V1

A: File a GitHub issue. Most V0 features are in V1, and missing ones can be considered for implementation.

### Q: How do I check if I'm using any V0-specific features?

A: Run the parity check script:

```bash
python scripts/v0_v1_parity.py --verbose
```

And check your code for deprecation warnings.

## Rollback (Emergency Only)

If you absolutely cannot migrate immediately:

1. Pin to the last version with V0: `pip install vllm==0.8.x`
2. Use the legacy Docker image
3. Plan your migration - unmaintained code is a security risk

**This is not recommended** as you'll miss security updates and new features.
