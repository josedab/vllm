# Scheduler Architecture

This document describes the internal architecture of vLLM's scheduler component.

## Overview

The scheduler is responsible for deciding which requests to execute in each iteration. It manages request queues, allocates KV cache blocks, and handles preemption when memory is constrained.

## Key Files

- `vllm/v1/core/sched/scheduler.py` - Main V1 scheduler logic
- `vllm/core/scheduler.py` - V0 scheduler (legacy)
- `vllm/v1/core/kv_cache_manager.py` - KV cache allocation
- `vllm/v1/request.py` - Request data structure

## Request States

Requests transition through the following states:

```
WAITING → RUNNING → FINISHED
            ↓
        PREEMPTED
            ↓
         WAITING
```

- **WAITING**: Request is queued, waiting for resources
- **RUNNING**: Request is actively being processed
- **PREEMPTED**: Request was running but preempted due to memory pressure
- **FINISHED**: Request has completed (success or error)

## Scheduling Algorithm

The scheduler runs in each iteration to determine which requests to process.

### Phase 1: Schedule Running Requests

For each currently running request:

1. Calculate tokens to generate this iteration
2. Try to allocate additional KV cache blocks if needed
3. If allocation fails due to memory pressure:
   - Preempt lower-priority requests first
   - If still fails, preempt this request
4. Add to the scheduled batch if resources available

### Phase 2: Schedule Waiting Requests

For each waiting request (ordered by priority/FCFS):

1. Check for prefix cache hits (reuse existing KV cache)
2. Calculate required KV cache blocks
3. Try to allocate the blocks
4. If successful, move request to running state
5. If allocation fails, stop scheduling (no more capacity)

### Phase 3: Prepare Outputs

After scheduling decisions are made:

1. Prepare the `SchedulerOutputs` structure
2. Include metadata for model execution
3. Return blocks to free list for preempted requests

## Key Design Decisions

### Token-Budget Based Scheduling

The scheduler uses a token budget rather than request count because:

- **Variable prompt lengths**: Different prompts have vastly different sizes
- **Chunked prefill support**: Long prompts need to be processed in chunks
- **Fine-grained control**: Enables precise memory management
- **Better batching**: Can fit more short requests or fewer long ones optimally

### FCFS with Priority Override

The default scheduling policy is First-Come-First-Served (FCFS) with optional priority:

**Advantages:**
- Simple and predictable behavior
- Avoids starvation of old requests
- Easy to reason about performance

**Priority override allows:**
- Important requests to jump the queue
- Business logic integration (e.g., paying users first)
- Latency-sensitive vs throughput-optimized paths

### Preemption Strategy

When memory is insufficient, the scheduler preempts requests:

1. **Recompute preemption**: Discard KV cache, recompute when resumed
   - Simple implementation
   - Works well with prefix caching

2. **Swap preemption**: Move KV cache to CPU memory
   - Preserves computation
   - Higher memory overhead
   - More complex implementation

The default is recompute preemption for simplicity.

## Configuration

Key scheduler configuration parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_num_seqs` | 256 | Maximum concurrent sequences |
| `max_num_batched_tokens` | None | Maximum tokens per iteration (auto-calculated if None) |
| `enable_chunked_prefill` | False | Break up long prompts into chunks |
| `delay_factor` | 0.0 | Delay scheduling to improve batching |
| `enable_prefix_caching` | False | Reuse KV cache for common prefixes |

### Example Configuration

```python
from vllm import LLM

# Throughput-optimized
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_num_seqs=256,
    max_num_batched_tokens=8192,
    enable_prefix_caching=True
)

# Latency-optimized
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_num_seqs=64,
    enable_chunked_prefill=True
)
```

## Metrics

The scheduler exposes several Prometheus metrics:

| Metric | Description |
|--------|-------------|
| `vllm:num_requests_running` | Current running requests |
| `vllm:num_requests_waiting` | Queue depth |
| `vllm:num_preemptions_total` | Total preemption count |
| `vllm:num_requests_swapped` | Requests currently swapped to CPU |

### Monitoring Example

```bash
# Get current scheduler state
curl http://localhost:8000/metrics | grep -E "requests_(running|waiting)"
```

## Integration Points

### With KV Cache Manager

The scheduler requests block allocations from the KV cache manager:

```python
# Simplified flow
blocks = kv_cache_manager.allocate(num_blocks=required_blocks)
if blocks is None:
    # Need to preempt
    ...
```

### With Model Runner

The scheduler produces `SchedulerOutputs` consumed by the model runner:

- `scheduled_seq_groups`: Sequences to process
- `blocks_to_swap_in`: CPU→GPU transfers
- `blocks_to_swap_out`: GPU→CPU transfers
- `blocks_to_copy`: Block copies for beam search

## Advanced Topics

### Chunked Prefill

When enabled, long prompts are split into chunks:

1. First chunk scheduled with `is_first_prefill=True`
2. Subsequent chunks scheduled as continuation
3. Decode phase starts after all chunks processed

Benefits:
- Better batching (mix prefill and decode)
- Lower TTFT for short prompts
- More predictable latency

### Prefix Caching

The scheduler checks for prefix cache hits:

1. Hash the prompt tokens
2. Check if KV cache blocks exist
3. Skip recomputation for cached prefix
4. Only compute new tokens

This significantly improves throughput for:
- Shared system prompts
- Few-shot examples
- RAG applications

### Multi-Step Scheduling

In multi-step mode, the scheduler runs less frequently:

1. Schedule once for N steps
2. Model runner executes N decode steps
3. Reduces scheduling overhead
4. Requires careful memory management

## See Also

- [Architecture Overview](arch_overview.md)
- [PagedAttention Design](paged_attention.md)
- [Prefix Caching](prefix_caching.md)
- [Metrics Documentation](metrics.md)
