# Deep Dive: PagedAttention and Memory Management

**Part 2 of the vLLM Technical Blog Series**
**Commit SHA:** `67745d189fd981ee824bde35666a3737a962c031`

---

## What You'll Learn

- Why traditional KV cache allocation wastes 60-80% of GPU memory
- How PagedAttention solves this with OS-inspired paging
- Block management: allocation, reference counting, eviction
- Prefix caching: sharing computation across requests

---

## Introduction

In Part 1, we explored vLLM's architecture and saw how continuous batching maximizes GPU utilization. But there's another dimension to optimization: memory. If your GPU memory fragments into unusable chunks, no amount of clever scheduling will help.

This is where PagedAttention shines. vLLM's core innovation treats KV cache memory like operating system virtual memory, with fixed-size blocks that can be allocated anywhere. The result? Near-zero memory waste and 7x more concurrent requests.

Let's dive into how it works.

---

## The Memory Problem

In autoregressive LLM generation, we cache the key and value projections (KV cache) to avoid recomputation. For each token, we need to store:

```
KV per token = 2 × num_layers × num_heads × head_dim × precision
```

For Llama-2-7B: `2 × 32 × 32 × 128 × 2 bytes = 512 KB per token`

For a 2048-token sequence: `512 KB × 2048 = 1 GB`

### The Traditional Approach

Traditional systems pre-allocate contiguous memory for the maximum possible sequence length:

```
┌─────────────────────────────────────┐
│  Request A: max_tokens=2048         │
│  [Used: 100 tokens][Wasted: 1948]   │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│  Request B: max_tokens=2048         │
│  [Used: 500 tokens][Wasted: 1548]   │
└─────────────────────────────────────┘
```

Problems:
1. **Internal fragmentation**: Reserved but unused memory within allocations
2. **External fragmentation**: Free memory scattered between allocations
3. **Over-reservation**: Must reserve for max length even if requests are shorter

Research shows this wastes **60-80% of GPU memory** in practice.

---

## The PagedAttention Solution

PagedAttention borrows the brilliant idea of **virtual memory paging** from operating systems. Instead of contiguous allocation, memory is divided into fixed-size **blocks**.

### Key Insight

Just as OS pages let processes see contiguous virtual addresses while physical memory is fragmented, KV cache blocks let sequences see logical positions while physical GPU memory is scattered.

```
Logical View (Sequence A):      Physical GPU Memory:
┌───┬───┬───┬───┐               ┌───┐ Block 0: Free
│ 0 │ 1 │ 2 │ 3 │               ├───┤
└─┬─┴─┬─┴─┬─┴─┬─┘               │ 1 │ Block 1: Free
  │   │   │   │                 ├───┤
  │   │   │   │     Block       │ 2 │ Block 2: Seq A
  │   │   │   │     Table       ├───┤
  │   │   │   └────────────────►│ 3 │ Block 3: Seq A
  │   │   └────────────────────►│ 4 │ Block 4: Free
  │   └────────────────────────►│ 5 │ Block 5: Seq B
  └────────────────────────────►│ 6 │ Block 6: Free
                                ├───┤
                                │ 7 │ Block 7: Seq A
                                ├───┤
                                │ 9 │ Block 9: Seq A
                                └───┘
```

---

## Implementation Details

### Block Structure

Each block holds a fixed number of tokens (default: 16):

```python
# From vllm/v1/core/block_pool.py
# https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/v1/core/block_pool.py#L28

@dataclass
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0  # Reference counting for sharing
    block_hash: Optional[int] = None  # For prefix caching
    prev_free_block: Optional['KVCacheBlock'] = None  # LRU list
    next_free_block: Optional['KVCacheBlock'] = None
```

### Block Pool

The BlockPool manages free and allocated blocks:

```python
# From vllm/v1/core/block_pool.py
# https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/v1/core/block_pool.py#L85

class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching):
        # Pre-allocate all blocks
        self.blocks = [KVCacheBlock(i) for i in range(num_gpu_blocks)]
        # Free list for allocation
        self.free_list_head = self.blocks[0]
        # Hash table for prefix caching
        self.cached_block_hash_to_block = {}

    def allocate(self, num_blocks):
        """Allocate blocks from free list."""
        allocated = []
        for _ in range(num_blocks):
            block = self._pop_free_block()
            block.ref_cnt = 1
            allocated.append(block)
        return allocated

    def free(self, block):
        """Return block to free list."""
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self._push_free_block(block)
```

### Reference Counting

Multiple sequences can share blocks (for prefix caching). Reference counting tracks this:

```python
# When a new sequence shares a cached prefix
def touch(self, block):
    block.ref_cnt += 1

# When a sequence frees its reference
def free(self, block):
    block.ref_cnt -= 1
    if block.ref_cnt == 0:
        # Actually free the block
        self._return_to_free_list(block)
```

---

## KV Cache Manager

The `KVCacheManager` coordinates allocation across requests:

```python
# From vllm/v1/core/kv_cache_manager.py
# https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/v1/core/kv_cache_manager.py#L89

class KVCacheManager:
    def allocate_slots(self, request, num_new_tokens):
        """Allocate KV cache for new tokens."""
        num_blocks_needed = math.ceil(
            (request.num_computed_tokens + num_new_tokens) / self.block_size
        ) - len(request.block_table)

        if num_blocks_needed > 0:
            new_blocks = self.block_pool.allocate(num_blocks_needed)
            request.block_table.extend(new_blocks)

        return SlotMapping(request.block_table, num_new_tokens)

    def get_num_free_blocks(self):
        """How many blocks are available for new requests."""
        return self.block_pool.num_free_blocks
```

---

## Prefix Caching

One of PagedAttention's most powerful features is **prefix caching**. If multiple requests share the same system prompt, why compute it multiple times?

### How It Works

```python
# From vllm/v1/core/kv_cache_utils.py
# https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/v1/core/kv_cache_utils.py#L45

def hash_block_tokens(parent_block_hash, curr_block_token_ids):
    """Hash a block's content for cache lookup."""
    return hash((parent_block_hash, *curr_block_token_ids))
```

The process:
1. Hash each block's tokens (including parent block's hash for chain)
2. On new request, check if block hash exists in cache
3. If hit, share the existing block (increment ref_cnt)
4. If miss, compute and add to cache

### Example

```
Request A: "You are a helpful assistant. What is Python?"
Request B: "You are a helpful assistant. What is JavaScript?"
                ↓
     Shared System Prompt (cached)
       [Block 0] [Block 1]
           ↓           ↓
         ref_cnt=2  ref_cnt=2
```

Request B saves computation for the entire system prompt!

### Cache Eviction

When memory is tight, the BlockPool uses **LRU eviction** with a twist:

```python
def evict_lru_block(self):
    """Evict least recently used cached block."""
    # Find LRU block that's cached (ref_cnt might still be 0)
    block = self.lru_list_tail
    while block and block.block_hash:
        if block.ref_cnt == 0:
            # Safe to evict
            del self.cached_block_hash_to_block[block.block_hash]
            block.block_hash = None
            return block
        block = block.prev_free_block
    return None
```

---

## GPU Memory Layout

On the GPU, memory is pre-allocated as a large tensor:

```python
# From vllm/v1/worker/gpu_model_runner.py (simplified)

def _init_kv_caches(self):
    # Allocate KV cache as contiguous tensor
    # Shape: [num_layers, 2, num_blocks, block_size, num_heads, head_dim]
    kv_cache = torch.zeros(
        self.num_layers,
        2,  # K and V
        self.num_gpu_blocks,
        self.block_size,
        self.num_kv_heads,
        self.head_dim,
        dtype=self.dtype,
        device="cuda"
    )
    return kv_cache
```

The BlockPool's block IDs directly index into this tensor.

---

## Attention with Paged KV Cache

The attention kernel must handle non-contiguous blocks. Here's the key operation:

```python
# From vllm/attention/ops/paged_attn.py
# https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/attention/ops/paged_attn.py#L67

class PagedAttention:
    @staticmethod
    def forward_decode(
        query,           # [num_seqs, num_heads, head_dim]
        key_cache,       # [num_blocks, block_size, num_heads, head_dim]
        value_cache,     # [num_blocks, block_size, num_heads, head_dim]
        block_tables,    # [num_seqs, max_num_blocks]
        seq_lens,        # [num_seqs]
    ):
        """Attention with paged KV cache."""
        # Custom CUDA kernel handles block table indirection
        return paged_attention_v1(
            query, key_cache, value_cache,
            block_tables, seq_lens
        )
```

The CUDA kernel iterates over block tables to gather the correct K/V values for each sequence.

---

## Performance Benefits

### Memory Efficiency

| Approach | Allocation | Waste | Concurrent Requests |
|----------|------------|-------|---------------------|
| Traditional | Contiguous, max length | 60-80% | N |
| PagedAttention | Block-based, as needed | <4% | 7N |

### Why 7x?

With 60% waste eliminated:
- Before: 100 GB usable → 40 GB actual
- After: 100 GB usable → 96 GB actual (only block overhead)

More memory = more concurrent requests = higher throughput.

### Prefix Caching Impact

For applications with common prompts (chatbots, RAG):
- System prompt: 500 tokens × 1000 requests
- Without caching: 500,000 token computations
- With caching: 500 token computations + 999 × 0

---

## Configuration Options

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",

    # Block configuration
    block_size=16,  # Tokens per block (default: 16)

    # Prefix caching
    enable_prefix_caching=True,  # Enable automatic prefix sharing

    # Memory budget
    gpu_memory_utilization=0.9,  # Fraction of GPU memory for KV cache

    # Swap space (for preemption)
    swap_space=4,  # GB of CPU memory for swapped blocks
)
```

---

## Trade-offs

### Block Size

| Size | Pros | Cons |
|------|------|------|
| Small (8) | Less waste per block | More block table overhead |
| Large (32) | Less overhead | More internal fragmentation |
| Default (16) | Balanced | Good for most cases |

### Prefix Caching

| Setting | Benefit | Cost |
|---------|---------|------|
| Enabled | Saves compute for shared prefixes | Hash computation overhead |
| Disabled | Simpler, less memory for hash table | Redundant computation |

For most production workloads, prefix caching is a clear win.

---

## Debugging Memory Issues

### Check Block Utilization

```python
# Monitor via Prometheus metrics
# GET /metrics
# vllm:kv_cache_usage_perc - Current utilization
# vllm:num_free_gpu_blocks - Available blocks
```

### Common Issues

1. **OOM during allocation**: Reduce `max_model_len` or `max_num_seqs`
2. **Low cache hit rate**: Check if prompts actually share prefixes
3. **High fragmentation**: Shouldn't happen with PagedAttention, check for bugs

---

## Key Takeaways

1. **PagedAttention uses OS-inspired paging** to eliminate memory fragmentation

2. **Fixed-size blocks** can be allocated anywhere in GPU memory

3. **Block tables** map logical sequence positions to physical blocks

4. **Reference counting** enables safe sharing for prefix caching

5. **LRU eviction** manages cache pressure intelligently

6. **Result:** Near-zero waste, 7x more concurrent requests

---

## Next Steps

In [Part 3: Patterns and Practices](03-patterns-practices.md), we'll explore the design patterns vLLM uses throughout the codebase—from plugin architectures for attention backends to state machines for request lifecycle management.

---

## Code References

- **BlockPool:** [vllm/v1/core/block_pool.py](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/v1/core/block_pool.py)
- **KVCacheManager:** [vllm/v1/core/kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/v1/core/kv_cache_manager.py)
- **PagedAttention Ops:** [vllm/attention/ops/paged_attn.py](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/attention/ops/paged_attn.py)
- **Block Hashing:** [vllm/v1/core/kv_cache_utils.py](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/v1/core/kv_cache_utils.py)

---

*This is Part 2 of the vLLM Technical Blog Series based on commit `67745d189fd981ee824bde35666a3737a962c031`.*
