# vLLM Repository Structure

**Commit SHA:** `67745d189fd981ee824bde35666a3737a962c031`

## Root Directory Overview

```
vllm/
├── vllm/                    # Main Python package (402,982 LoC)
├── csrc/                    # CUDA/C++ source (74 kernels)
├── tests/                   # Test suite (800+ files, 166,425 LoC)
├── benchmarks/              # Performance benchmarks
├── docs/                    # Documentation (MkDocs)
├── examples/                # Usage examples
├── tools/                   # Development tools
├── docker/                  # Container definitions
├── requirements/            # Dependency files
├── .buildkite/              # CI/CD pipeline
└── .github/                 # GitHub workflows
```

## Core Package Structure (`vllm/`)

### Entry Points and API Layer

```
vllm/entrypoints/
├── llm.py                   # Offline LLM class (74KB)
├── cli/                     # CLI interface
├── openai/                  # OpenAI-compatible API
│   ├── api_server.py        # FastAPI server (30+ endpoints)
│   ├── protocol.py          # Request/response models
│   └── serving_*.py         # Endpoint handlers
└── anthropic/               # Anthropic API compatibility
```

### Engine Core (V1 Architecture)

```
vllm/v1/
├── engine/
│   ├── core.py              # Main scheduling loop (56KB)
│   ├── async_llm.py         # Async API wrapper (797 lines)
│   ├── processor.py         # Input/output processing (26KB)
│   └── core_client.py       # Communication abstraction
├── core/
│   ├── sched/
│   │   └── scheduler.py     # Scheduling logic (73KB)
│   ├── kv_cache_manager.py  # KV cache management
│   └── block_pool.py        # Memory block management
├── worker/
│   ├── gpu_worker.py        # GPU worker implementation
│   └── gpu_model_runner.py  # Model execution (224KB)
└── executor/
    ├── multiproc_executor.py # Multi-process executor
    └── ray_executor.py       # Ray distributed executor
```

### Model Execution

```
vllm/model_executor/
├── models/                  # Model implementations (150+ architectures)
│   ├── llama.py
│   ├── mistral.py
│   ├── qwen2_vl.py
│   └── ...
├── layers/
│   ├── linear.py            # Tensor parallel layers
│   ├── attention/           # Attention implementations
│   ├── quantization/        # Quantization methods (25+)
│   └── fused_moe/           # MoE implementations
└── model_loader/            # Weight loading
```

### Attention System

```
vllm/attention/
├── backends/                # Backend implementations
│   ├── flash_attn.py
│   ├── flashinfer.py
│   └── pallas.py (TPU)
├── ops/
│   └── paged_attn.py        # PagedAttention operations
└── layer.py                 # Unified attention layer

vllm/v1/attention/
└── backends/                # V1-specific backends (33+)
```

### Distributed Computing

```
vllm/distributed/
├── parallel_state.py        # Process group management
├── communication_op.py      # Collective operations
└── device_communicators/    # Backend implementations
    ├── pynccl.py
    ├── custom_all_reduce.py
    └── ...
```

### Configuration System

```
vllm/config/
├── __init__.py              # VllmConfig aggregator
├── model.py                 # ModelConfig
├── cache.py                 # CacheConfig
├── parallel.py              # ParallelConfig
├── scheduler.py             # SchedulerConfig
├── compilation.py           # CompilationConfig
└── observability.py         # ObservabilityConfig

vllm/envs.py                 # Environment variables (230+)
```

### Supporting Systems

```
vllm/
├── compilation/             # torch.compile integration
│   ├── backends.py          # Compilation backends
│   ├── cuda_graph.py        # CUDA graph capture
│   └── collective_fusion.py # Inductor passes
├── transformers_utils/      # HuggingFace integration
├── triton_utils/            # Triton kernel utilities
├── lora/                    # LoRA adapter support
├── multimodal/              # Multi-modal processing
├── profiler/                # GPU profiling
├── usage/                   # Usage statistics
└── logging_utils/           # Custom logging
```

## CUDA/C++ Source (`csrc/`)

```
csrc/
├── attention/               # Attention kernels
│   ├── paged_attention_v1.cu
│   ├── paged_attention_v2.cu
│   └── attention_kernels.cu
├── quantization/            # Quantization kernels
│   ├── fp8/
│   ├── gptq/
│   ├── awq/
│   └── marlin/
├── cache_kernels.cu         # KV cache operations
├── moe/                     # MoE kernels
├── layernorm_kernels.cu
├── pos_encoding_kernels.cu
└── activation_kernels.cu
```

## Test Suite (`tests/`)

```
tests/
├── conftest.py              # Core fixtures (1,399 lines)
├── utils.py                 # Test utilities (1,305 lines)
├── v1/                      # V1 API tests
├── entrypoints/             # API endpoint tests
├── distributed/             # Multi-GPU tests
├── models/                  # Model correctness tests
├── kernels/                 # Kernel unit tests
├── quantization/            # Quantization tests
├── lora/                    # LoRA tests
└── basic_correctness/       # Output verification
```

## Benchmarks (`benchmarks/`)

```
benchmarks/
├── throughput.py            # Offline throughput
├── latency.py               # Latency measurement
├── serving.py               # Online serving
├── prefix_caching.py        # Cache efficiency
├── kernels/                 # Kernel benchmarks (40+)
│   ├── paged_attention.py
│   ├── fp8_gemm.py
│   └── ...
└── structured_output.py     # Constrained generation
```

## Documentation (`docs/`)

```
docs/
├── getting_started/         # Installation, quickstart
├── models/                  # Supported models
├── features/                # Feature guides
├── design/                  # Architecture documentation
├── contributing/            # Contributor guide
└── api/                     # API reference
```

## CI/CD (`.buildkite/`, `.github/`)

```
.buildkite/
├── test-pipeline.yaml       # Main CI pipeline (1,315 lines)
├── release-pipeline.yaml    # Release workflow
└── test-amd.yaml            # AMD GPU tests

.github/
├── workflows/               # GitHub Actions
│   ├── pre-commit.yml
│   └── issue_autolabel.yml
└── dependabot.yml           # Dependency updates
```

## Key File Sizes

| File | Size | Purpose |
|------|------|---------|
| `vllm/v1/worker/gpu_model_runner.py` | 224KB | GPU execution |
| `vllm/_custom_ops.py` | 85KB | CUDA op bindings |
| `vllm/envs.py` | 76KB | Environment config |
| `vllm/entrypoints/llm.py` | 74KB | Offline LLM class |
| `vllm/v1/core/sched/scheduler.py` | 73KB | Scheduler |
| `vllm/v1/engine/core.py` | 56KB | Engine core |

## Module Dependencies

```
Entry Points → Engine → Scheduler → Executor → Workers → Model Runner
                 ↓
            KV Cache Manager → Block Pool
                 ↓
            Attention Backends → CUDA Kernels
```

## Notes for Contributors

1. **V1 vs Legacy:** New development should target `vllm/v1/`
2. **Testing:** Add tests to corresponding directory in `tests/`
3. **Kernels:** CUDA code requires matching Python bindings in `_custom_ops.py`
4. **Models:** Follow registry pattern in `model_executor/models/`
5. **Config:** Add new configs to `vllm/config/` with validation

---

*Structure as of commit `67745d189fd981ee824bde35666a3737a962c031`*
