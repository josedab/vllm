# vLLM Terminology Glossary

**Commit SHA:** `67745d189fd981ee824bde35666a3737a962c031`

## Core Concepts

### PagedAttention
The core innovation of vLLM. A memory management technique that stores attention keys and values in non-contiguous blocks (like OS virtual memory pages), enabling:
- Near-zero memory waste
- Dynamic memory sharing across sequences
- Efficient prefix caching

### KV Cache
**Key-Value Cache.** Storage for the attention mechanism's key and value projections. In autoregressive generation, previously computed KV pairs are cached to avoid recomputation. vLLM manages this via PagedAttention.

### Continuous Batching
A scheduling technique where requests can join or leave a batch at each generation step (token-level granularity), rather than waiting for the entire batch to complete. This maximizes GPU utilization.

### Prefix Caching
Automatic detection and reuse of common prompt prefixes across requests. If multiple requests share the same system prompt, the KV cache blocks for that prefix are computed once and shared.

## Architecture Terms

### EngineCore
The central component in vLLM V1 that runs the main scheduling loop. Responsible for:
- Scheduling requests
- Coordinating workers
- Managing the execution pipeline

### Scheduler
The component that decides which requests to execute in each iteration. Manages:
- Request queues (waiting, running, preempted)
- KV cache block allocation
- Preemption decisions

### Worker
A process that executes model inference on a specific device (GPU). In distributed settings, multiple workers coordinate via collective operations.

### Executor
Manages the lifecycle of workers. Types include:
- **UniprocExecutor:** Single-process execution
- **MultiprocExecutor:** Multi-process on single node
- **RayDistributedExecutor:** Cluster distribution via Ray

### Model Runner
The component that actually executes model forward passes on GPU. Handles:
- Input preparation
- Model execution
- Output sampling

## Parallelism Terms

### Tensor Parallelism (TP)
Splitting model layers across GPUs within a single node. Each GPU holds a portion of each layer's weights. Requires high-bandwidth interconnect (NVLink).

### Pipeline Parallelism (PP)
Splitting model layers sequentially across GPUs/nodes. GPU 0 has layers 0-11, GPU 1 has layers 12-23, etc. Lower communication overhead than TP.

### Expert Parallelism (EP)
For Mixture-of-Experts models. Distributing experts across GPUs so each GPU handles a subset of experts.

### Data Parallelism (DP)
Running the same model on multiple GPUs with different data. Used for aggregating results or handling high request volumes.

### Sequence Parallelism (SP)
Distributing sequence dimensions across devices for very long sequences.

## Memory Terms

### Block
A fixed-size unit of KV cache memory (default: 16 tokens). The atomic unit of allocation in PagedAttention.

### Block Table
A mapping from logical block indices (per sequence) to physical GPU memory blocks. Like a page table in OS.

### Block Pool
The memory manager that tracks free and allocated blocks. Implements LRU eviction for prefix caching.

### Swap Space
CPU memory used to temporarily store preempted sequences' KV cache. Allows preemption without losing computation.

### GPU Memory Utilization
The fraction of GPU memory vLLM is allowed to use. Default: 0.9 (90%).

## Scheduling Terms

### Request States
- **WAITING:** In queue, not yet scheduled
- **RUNNING:** Currently being processed
- **PREEMPTED:** Temporarily paused due to memory pressure
- **FINISHED_STOPPED:** Completed with stop token
- **FINISHED_LENGTH:** Reached max tokens
- **FINISHED_ABORTED:** Cancelled or errored

### Preemption
When memory pressure is high, the scheduler pauses lower-priority requests to free memory for higher-priority ones. Preempted requests are either swapped to CPU or recomputed later.

### Chunked Prefill
Breaking a long prompt into chunks that are processed across multiple iterations. Prevents a single long prompt from monopolizing the GPU.

### Token Budget
The maximum number of tokens (prefill + decode) the scheduler can include in a single batch/iteration.

## Performance Terms

### Time to First Token (TTFT)
Latency from request arrival to the first generated token. Critical for interactive applications.

### Inter-Token Latency (ITL)
Time between consecutive generated tokens. Determines perceived "typing speed."

### Throughput
Total tokens generated per second across all requests. Primary metric for batch inference.

### Prefill Phase
Processing the input prompt to generate initial KV cache. Compute-intensive.

### Decode Phase
Generating output tokens one at a time. Memory-bandwidth intensive.

## Quantization Terms

### FP8
8-bit floating point format. Two variants:
- **E4M3:** 4 exponent, 3 mantissa bits (for weights)
- **E5M2:** 5 exponent, 2 mantissa bits (for gradients)

### AWQ (Activation-aware Weight Quantization)
Quantization method that protects salient weights based on activation patterns.

### GPTQ
Post-training quantization using approximate second-order information.

### Marlin
Highly optimized quantization kernel for specific bit-widths.

### Weight-only Quantization
Quantizing only model weights, keeping activations in higher precision.

## Attention Terms

### FlashAttention
Memory-efficient attention algorithm that avoids materializing the full attention matrix by tiling and recomputation.

### FlashInfer
Framework-optimized attention kernels with features like cascade attention and fused operations.

### Multi-Head Attention (MHA)
Standard attention with multiple parallel attention heads.

### Grouped Query Attention (GQA)
Attention where multiple query heads share the same key-value heads. Reduces KV cache size.

### Multi-Query Attention (MQA)
Extreme case of GQA where all query heads share one key-value head.

### Multi-head Latent Attention (MLA)
Alternative attention mechanism used in some models (e.g., DeepSeek).

## Speculative Decoding Terms

### Draft Model
A smaller, faster model that proposes multiple tokens, which are then verified by the main model in parallel.

### Speculative Tokens
Tokens proposed by the draft model before verification.

### N-gram Proposer
Uses prompt patterns to propose likely continuations without a separate model.

### Medusa/EAGLE
Speculative decoding methods using additional prediction heads on the main model.

## API Terms

### OpenAI-Compatible API
vLLM's HTTP API that matches OpenAI's endpoints and schemas, allowing drop-in replacement.

### Sampling Parameters
Settings that control generation:
- `temperature` - Randomness
- `top_p` - Nucleus sampling
- `top_k` - Top-k sampling
- `max_tokens` - Output length limit

### Streaming
Sending generated tokens as they're produced, rather than waiting for completion.

### Structured Output
Constraining generation to follow a specific format (JSON schema, regex, grammar).

## Compilation Terms

### torch.compile
PyTorch's compilation system using Dynamo for tracing and Inductor for code generation.

### CUDA Graph
A recorded sequence of GPU operations that can be replayed with minimal CPU overhead. Major performance optimization.

### Inductor
PyTorch's backend compiler that generates optimized CUDA/Triton kernels.

### Batch Invariance
Property where operations produce the same results regardless of batch size, enabling CUDA graph reuse.

## Distributed Terms

### NCCL
NVIDIA Collective Communications Library. High-performance multi-GPU communication.

### All-Reduce
Collective operation that aggregates values from all processes and distributes the result to all.

### All-to-All
Collective operation where each process sends distinct data to all other processes. Used in MoE routing.

### Process Group
A subset of processes that can communicate with each other for collective operations.

## Observability Terms

### Prometheus Metrics
Time-series metrics exposed at `/metrics` endpoint. Includes latency histograms, throughput counters, and resource gauges.

### OpenTelemetry (OTEL)
Standard for distributed tracing. vLLM can export traces to OTEL collectors.

### Tracing
Recording the execution path and timing of requests through the system.

## Model Terms

### LoRA (Low-Rank Adaptation)
Parameter-efficient fine-tuning where small adapter matrices are added to model layers. vLLM supports serving multiple LoRA adapters.

### Mixture of Experts (MoE)
Architecture where each token is routed to a subset of "expert" FFN layers, enabling sparse computation.

### Multi-modal
Models that process multiple input types (text, images, audio, video).

### Embedding Model
Model that produces vector representations of text for retrieval/similarity tasks.

## File Format Terms

### safetensors
Safe, fast format for storing tensors. Default for HuggingFace models.

### GGUF
GGML Unified Format. Used by llama.cpp and compatible with vLLM.

### Checkpoint
Saved model weights and configuration.

## Configuration Terms

### VllmConfig
Master configuration object aggregating all sub-configs.

### Engine Arguments
CLI/programmatic arguments that configure the engine (model, parallelism, memory, etc.).

### Environment Variables
`VLLM_*` prefixed variables for runtime configuration without code changes.

---

*Terminology based on vLLM commit `67745d189fd981ee824bde35666a3737a962c031`*
