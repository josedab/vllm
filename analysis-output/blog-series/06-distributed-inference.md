# Distributed Inference and Scaling

**Part 6 of the vLLM Technical Blog Series**
**Commit SHA:** `67745d189fd981ee824bde35666a3737a962c031`

---

## What You'll Learn

- Tensor, Pipeline, and Expert Parallelism strategies
- Worker coordination and communication patterns
- Multi-node deployment with Ray
- Scaling considerations and trade-offs

---

## Introduction

When a model is too large for one GPU, or when you need more throughput than a single GPU can provide, distributed inference becomes necessary. vLLM supports multiple parallelism strategies to scale across GPUs and nodes.

In this final post of our series, we'll explore how vLLM distributes models across hardware and coordinates inference at scale.

---

## Parallelism Strategies

### Overview

vLLM supports three main parallelism strategies:

| Strategy | Splits | Communication | Best For |
|----------|--------|---------------|----------|
| **Tensor (TP)** | Each layer across GPUs | All-reduce | Single node, NVLink |
| **Pipeline (PP)** | Layers sequentially | Point-to-point | Multi-node |
| **Expert (EP)** | Experts across GPUs | All-to-all | MoE models |

### Tensor Parallelism

Tensor parallelism splits each layer horizontally:

```
GPU 0: [Layer 0: weights 0-50%] [Layer 1: weights 0-50%] ...
GPU 1: [Layer 0: weights 50-100%] [Layer 1: weights 50-100%] ...

Communication: All-reduce after each layer
```

**Usage:**

```python
from vllm import LLM

# 2-way tensor parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2
)
```

**How it works:**

```python
# From vllm/model_executor/layers/linear.py
# https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/model_executor/layers/linear.py#L230

class ColumnParallelLinear(nn.Module):
    """Linear layer split column-wise across GPUs."""

    def __init__(self, input_size, output_size, ...):
        # Each GPU gets output_size / tp_size columns
        self.output_size_per_partition = output_size // tp_size
        self.weight = Parameter(torch.empty(
            self.output_size_per_partition,
            input_size
        ))

    def forward(self, x):
        # Each GPU computes partial output
        output = F.linear(x, self.weight)
        # No communication needed here
        return output

class RowParallelLinear(nn.Module):
    """Linear layer split row-wise across GPUs."""

    def forward(self, x):
        # Each GPU computes partial output
        output = F.linear(x, self.weight)
        # All-reduce to sum partial results
        output = tensor_model_parallel_all_reduce(output)
        return output
```

### Pipeline Parallelism

Pipeline parallelism assigns different layers to different GPUs:

```
GPU 0: [Layers 0-5]
GPU 1: [Layers 6-11]
GPU 2: [Layers 12-17]
GPU 3: [Layers 18-23]

Communication: Point-to-point between adjacent stages
```

**Usage:**

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    pipeline_parallel_size=4
)
```

**Advantage:** Minimal communication (just activations between stages)
**Disadvantage:** GPU bubbles (some GPUs idle while waiting)

### Expert Parallelism

For Mixture-of-Experts (MoE) models, experts are distributed:

```
Model: 8 experts
GPU 0: Experts 0, 1
GPU 1: Experts 2, 3
GPU 2: Experts 4, 5
GPU 3: Experts 6, 7

Communication: All-to-all for routing tokens to experts
```

**Usage:**

```python
llm = LLM(
    model="mistralai/Mixtral-8x7B-v0.1",
    tensor_parallel_size=4  # Automatically enables EP for MoE
)
```

### Combined Parallelism

For very large models, combine strategies:

```python
# 8 GPUs: 4-way TP, 2-way PP
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    pipeline_parallel_size=2
)
```

---

## Worker Architecture

### Process Groups

vLLM creates process groups for different communication patterns:

```python
# From vllm/distributed/parallel_state.py
# https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/distributed/parallel_state.py

def initialize_model_parallel(
    tensor_model_parallel_size,
    pipeline_model_parallel_size
):
    """Initialize distributed process groups."""

    # Tensor parallel group: GPUs that share layers
    # Example with TP=2, PP=2:
    # TP group 0: [GPU 0, GPU 1]
    # TP group 1: [GPU 2, GPU 3]
    tensor_model_parallel_group = create_group(...)

    # Pipeline parallel group: GPUs in same pipeline
    # PP group 0: [GPU 0, GPU 2]
    # PP group 1: [GPU 1, GPU 3]
    pipeline_model_parallel_group = create_group(...)
```

### Worker Lifecycle

```python
# From vllm/v1/worker/gpu_worker.py (simplified)

class GPUWorker:
    def __init__(self, rank, config):
        self.rank = rank
        self.config = config

    def init_distributed(self):
        """Initialize NCCL and process groups."""
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size
        )
        initialize_model_parallel(
            self.config.tensor_parallel_size,
            self.config.pipeline_parallel_size
        )

    def load_model(self):
        """Load and shard model weights."""
        model = get_model(self.config)
        # Weights are automatically sharded based on TP/PP
        return model

    def execute_model(self, scheduler_output):
        """Run forward pass."""
        with torch.cuda.stream(self.stream):
            output = self.model_runner.execute(scheduler_output)

        # Synchronize across workers
        if self.config.pipeline_parallel_size > 1:
            self._pipeline_sync(output)

        return output
```

---

## Communication Patterns

### Tensor Parallel All-Reduce

After row-parallel layers, results must be summed:

```python
# From vllm/distributed/communication_op.py

def tensor_model_parallel_all_reduce(tensor):
    """Sum tensor across tensor parallel group."""
    if get_tensor_model_parallel_world_size() == 1:
        return tensor

    # Use custom all-reduce for better performance
    torch.distributed.all_reduce(
        tensor,
        group=get_tensor_model_parallel_group()
    )
    return tensor
```

### Custom All-Reduce

vLLM implements optimized all-reduce for specific topologies:

```python
# From vllm/distributed/device_communicators/custom_all_reduce.py

class CustomAllReduce:
    """Optimized all-reduce for NVLink topologies."""

    def __init__(self, group):
        # Detect topology and choose algorithm
        if is_nvlink_full_mesh():
            self.impl = OneShot(group)
        elif is_nvlink_ring():
            self.impl = TwoShot(group)
        else:
            self.impl = NCCLImpl(group)

    def all_reduce(self, tensor):
        return self.impl.all_reduce(tensor)
```

### Pipeline P2P Communication

```python
# Send activations to next stage
def send_forward(tensor, next_rank):
    torch.distributed.send(tensor, dst=next_rank)

# Receive activations from previous stage
def recv_forward(shape, dtype, prev_rank):
    tensor = torch.empty(shape, dtype=dtype)
    torch.distributed.recv(tensor, src=prev_rank)
    return tensor
```

---

## Multi-Node Deployment

### Ray Integration

For multi-node clusters, vLLM uses Ray:

```python
import ray
from vllm import LLM

# Start Ray cluster first
ray.init(address="auto")

# vLLM will automatically use Ray for distribution
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=8,  # Across multiple nodes
)
```

### Executor Types

```python
# From vllm/v1/executor/ray_executor.py (simplified)

class RayDistributedExecutor:
    """Manage workers across Ray cluster."""

    def __init__(self, config):
        self.config = config
        self.workers = []

    def _init_workers(self):
        """Create Ray actors for each GPU."""
        for rank in range(self.world_size):
            # Create worker as Ray actor
            worker = ray.remote(
                num_gpus=1,
            )(GPUWorker).remote(rank, self.config)
            self.workers.append(worker)

    def execute_model(self, scheduler_output):
        """Execute on all workers."""
        # Broadcast input to all workers
        futures = [
            w.execute_model.remote(scheduler_output)
            for w in self.workers
        ]
        # Gather results
        return ray.get(futures)
```

### Deployment Configuration

```bash
# Head node
ray start --head --port=6379

# Worker nodes
ray start --address=<head-node-ip>:6379 --num-gpus=8

# Start vLLM
vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 16 \
    --pipeline-parallel-size 2
```

---

## Scaling Considerations

### When to Use Each Strategy

| Scenario | Recommendation |
|----------|----------------|
| Single node, 2-8 GPUs | TP only |
| 2 nodes, 8+ GPUs | TP within node, PP across |
| Very large model | PP for memory, TP for compute |
| MoE model | EP + TP |
| Latency-critical | TP (lower latency than PP) |
| Throughput-critical | Larger TP for more parallelism |

### Performance Trade-offs

**Tensor Parallelism:**
- Pros: Lower latency, simple scheduling
- Cons: High communication (all-reduce), needs fast interconnect

**Pipeline Parallelism:**
- Pros: Minimal communication, works across nodes
- Cons: GPU bubbles, scheduling complexity

**Expert Parallelism:**
- Pros: Sparse computation
- Cons: All-to-all communication overhead

### Network Requirements

| Strategy | Bandwidth Need | Latency Need |
|----------|---------------|--------------|
| TP | Very High | Low |
| PP | Medium | Medium |
| EP | High | Medium |

Recommendation: NVLink for TP, InfiniBand for PP across nodes.

---

## Configuration Examples

### 8-GPU Single Node

```python
# Llama-70B on 8x A100
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=8
)
```

### 2-Node 16-GPU

```python
# Very large model across nodes
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=8,
    pipeline_parallel_size=2
)
```

### MoE Model

```python
# Mixtral with expert parallelism
llm = LLM(
    model="mistralai/Mixtral-8x7B-v0.1",
    tensor_parallel_size=8,
    # EP is automatic for MoE
)
```

---

## Debugging Distributed Issues

### Common Problems

**1. Hang on initialization:**
```bash
# Check NCCL environment
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

**2. OOM on specific GPU:**
```python
# Check memory distribution
torch.cuda.memory_summary()
```

**3. Slow communication:**
```bash
# Profile NCCL
export NCCL_DEBUG=INFO
# or use nsys for detailed profiling
nsys profile --trace=cuda,nvtx python your_script.py
```

### Verification

```python
# Test distributed setup
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

print(f"TP rank: {get_tensor_model_parallel_rank()}")
print(f"TP world size: {get_tensor_model_parallel_world_size()}")
```

---

## Key Takeaways

1. **Tensor parallelism** splits layers horizontally, needs fast interconnect

2. **Pipeline parallelism** splits layers sequentially, works across nodes

3. **Expert parallelism** distributes MoE experts, uses all-to-all

4. **Combine strategies** for very large models

5. **Ray integration** enables multi-node deployment

6. **Network topology matters** - use NVLink for TP, InfiniBand for PP

7. **Profile communication** to identify bottlenecks

---

## Series Conclusion

Over these six posts, we've explored vLLM from architecture to optimization:

1. **Architecture**: Producer-consumer pipeline with scheduling-centric design
2. **PagedAttention**: OS-inspired memory management for 7x more requests
3. **Patterns**: Plugin, state machine, and registry patterns
4. **Extension**: Adding models, backends, and integrations
5. **Performance**: Quantization, compilation, and profiling
6. **Distribution**: Scaling across GPUs and nodes

vLLM represents the state-of-the-art in open-source LLM serving. Understanding its internals enables you to optimize for your specific workloads and contribute to its development.

---

## Code References

- **Parallel State:** [vllm/distributed/parallel_state.py](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/distributed/parallel_state.py)
- **Parallel Layers:** [vllm/model_executor/layers/linear.py](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/model_executor/layers/linear.py)
- **Ray Executor:** [vllm/v1/executor/ray_executor.py](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/v1/executor/)
- **Communication:** [vllm/distributed/communication_op.py](https://github.com/vllm-project/vllm/blob/67745d189fd981ee824bde35666a3737a962c031/vllm/distributed/communication_op.py)

---

*This is Part 6 (Final) of the vLLM Technical Blog Series based on commit `67745d189fd981ee824bde35666a3737a962c031`.*
