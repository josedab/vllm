# vLLM Codebase Metrics Summary

**Commit SHA:** `67745d189fd981ee824bde35666a3737a962c031`
**Analysis Date:** November 2025

## Code Volume Metrics

### Lines of Code

| Component | Files | Lines of Code |
|-----------|-------|---------------|
| vLLM Package (`vllm/`) | ~1,200 | 402,982 |
| Tests (`tests/`) | 800+ | 166,425 |
| CUDA Kernels (`csrc/`) | 74 .cu + 67 .cuh | ~50,000 |
| Total Python | 2,041 | ~570,000 |

### Package Breakdown

| Subpackage | Estimated LoC | Key Responsibilities |
|------------|---------------|---------------------|
| `model_executor/` | ~120,000 | Model implementations, layers |
| `v1/` | ~60,000 | V1 engine, scheduler, workers |
| `entrypoints/` | ~40,000 | API server, CLI, offline LLM |
| `distributed/` | ~25,000 | Communication, parallel state |
| `attention/` | ~20,000 | Attention backends, ops |
| `config/` | ~15,000 | Configuration classes |
| `compilation/` | ~10,000 | torch.compile integration |
| Other | ~113,000 | Utilities, logging, etc. |

## Dependency Metrics

### Core Dependencies

| Category | Count | Key Packages |
|----------|-------|--------------|
| Deep Learning | 5 | torch (2.9.0), transformers (4.56+), xformers, flashinfer |
| Data Processing | 8 | numpy, pillow, opencv, einops |
| Web/API | 6 | fastapi, aiohttp, pydantic, uvicorn |
| Serialization | 5 | msgspec, protobuf, gguf, safetensors |
| Observability | 4 | prometheus_client, opentelemetry |
| Total Direct | 52+ | See requirements/common.txt |

### Python Version Support

- Python 3.10, 3.11, 3.12, 3.13
- Primary target: 3.11

### Hardware Support

| Platform | Status | Key Dependencies |
|----------|--------|------------------|
| NVIDIA CUDA | Primary | torch, xformers, flashinfer |
| AMD ROCm | Supported | torch-rocm, triton |
| Intel XPU | Supported | intel-extension-for-pytorch |
| TPU | Supported | torch-xla, pallas |
| CPU | Supported | intel-extension-for-pytorch |

## Test Coverage Metrics

### Test Organization

| Category | Files | Estimated Tests |
|----------|-------|-----------------|
| Unit Tests | ~200 | ~1,000 |
| Integration Tests | ~300 | ~1,500 |
| E2E Tests | ~150 | ~750 |
| Distributed Tests | ~100 | ~500 |
| Model Tests | ~50 | ~2,000+ (parametrized) |

### Test Infrastructure

- **Fixtures:** 27 major fixtures in root conftest.py
- **Test Runners:** HfRunner (baseline), VllmRunner (testing)
- **CI Time:** ~100 minutes for fast-check pipeline
- **Test Markers:** 10 custom markers for categorization

### Code Quality Tools

| Tool | Purpose | Configuration |
|------|---------|---------------|
| Ruff | Linting + formatting | pyproject.toml |
| MyPy | Type checking | pyproject.toml |
| clang-format | C++ formatting | .clang-format |
| typos | Spell checking | pyproject.toml |
| SPDX | License headers | .pre-commit-config.yaml |

## Architecture Metrics

### Core Abstractions

| Abstraction | Files | Complexity |
|-------------|-------|------------|
| Engine | 10+ | High (scheduling loop) |
| Scheduler | 5+ | Very High (state machine) |
| Worker | 8+ | High (device management) |
| Model Runner | 3+ | Very High (execution) |
| Attention | 33+ backends | Medium-High |
| KV Cache | 6+ | High (memory management) |

### Configuration Complexity

| Config Class | Fields | Validation Complexity |
|--------------|--------|----------------------|
| ModelConfig | 50+ | High |
| CacheConfig | 20+ | Medium |
| ParallelConfig | 30+ | High (cross-validation) |
| SchedulerConfig | 15+ | Medium |
| CompilationConfig | 25+ | High |
| Total Env Vars | 230+ | - |

## Performance Metrics

### Optimization Components

| Category | Count | Examples |
|----------|-------|----------|
| CUDA Kernels | 74 | Attention, quantization, activation |
| Quantization Methods | 25+ | FP8, AWQ, GPTQ, Marlin |
| Attention Backends | 33+ | FlashAttention, FlashInfer, Triton |
| Speculative Methods | 7 | ngram, medusa, eagle, draft model |
| Compilation Modes | 5 | None, torch.compile, CUDA graphs |

### Typical Performance Numbers

| Metric | Range | Notes |
|--------|-------|-------|
| Throughput | 2,000-10,000 tok/s | Depends on model/hardware |
| TTFT | 10-100ms | First token latency |
| ITL | 5-30ms | Inter-token latency |
| GPU Memory | 90-95% | Utilization target |
| KV Cache Hit | 0-90% | With prefix caching |

## Model Support Metrics

### Supported Architectures

| Category | Count | Examples |
|----------|-------|----------|
| Decoder-only LLMs | 100+ | LLaMA, Mistral, GPT-NeoX |
| Encoder-Decoder | 10+ | T5, BART |
| Vision-Language | 20+ | LLaVA, Qwen-VL, Phi-4 |
| Embedding | 10+ | E5, BGE |
| MoE | 10+ | Mixtral, DeepSeek |

### Model Sizes Tested

- 1B - 405B parameters
- Context lengths: 2K - 128K tokens
- Quantization: BF16, FP16, FP8, INT8, INT4

## CI/CD Metrics

### Pipeline Complexity

| Pipeline | Stages | YAML Lines | Duration |
|----------|--------|------------|----------|
| Test Pipeline | 30+ | 1,315 | ~100 min |
| Release Pipeline | 10+ | 500+ | ~60 min |
| AMD Pipeline | 15+ | 400+ | ~120 min |

### Test Hardware

- NVIDIA: A100, H100, H200, B200
- AMD: MI250, MI300X
- Intel: Xeon, Gaudi
- TPU: v4, v5

## Documentation Metrics

### Documentation Coverage

| Section | Pages | Quality |
|---------|-------|---------|
| Getting Started | 5+ | High |
| Features | 15+ | High |
| Models | 10+ | Medium |
| API Reference | 20+ | Medium |
| Design Docs | 10+ | High |
| Contributing | 5+ | High |

### Code Documentation

- Type hints: ~70% coverage
- Docstrings: ~50% coverage
- Inline comments: Variable

## Security Metrics

### Security Practices

- No credentials in code
- HTTPS by default for API server
- Input validation via Pydantic
- Rate limiting support
- Security advisories via GitHub

### License

- Apache 2.0
- All dependencies compatible

## Complexity Hotspots

### High Complexity Files

| File | Lines | Cyclomatic Complexity |
|------|-------|-----------------------|
| `gpu_model_runner.py` | 7,000+ | Very High |
| `scheduler.py` | 2,000+ | Very High |
| `core.py` | 1,800+ | High |
| `llm.py` | 2,000+ | High |
| `api_server.py` | 1,500+ | Medium-High |

### Technical Debt Areas

1. **Legacy Engine:** V0 engine still maintained alongside V1
2. **Backend Proliferation:** 33+ attention backends with overlap
3. **Configuration Sprawl:** 230+ environment variables
4. **Test Isolation:** Some tests require specific GPU types

## Recommendations

### For Understanding

1. Start with `vllm/v1/engine/core.py` for main loop
2. Read `vllm/v1/core/sched/scheduler.py` for scheduling
3. Study `vllm/attention/ops/paged_attn.py` for PagedAttention

### For Contributing

1. Focus on V1 architecture
2. Add comprehensive tests
3. Follow existing patterns
4. Update documentation

### For Optimization

1. Profile with built-in tools
2. Check attention backend selection
3. Enable compilation/CUDA graphs
4. Use appropriate quantization

---

*Metrics collected from commit `67745d189fd981ee824bde35666a3737a962c031`*
