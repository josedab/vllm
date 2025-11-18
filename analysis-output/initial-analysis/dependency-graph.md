# vLLM Dependency Graph

**Commit SHA:** `67745d189fd981ee824bde35666a3737a962c031`

## Core Dependencies Overview

```
                    ┌─────────────────────────────────────────┐
                    │              vLLM                        │
                    └─────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐          ┌───────────────┐            ┌───────────────┐
│  Deep Learning │          │   Web/API     │            │   Utilities   │
└───────────────┘          └───────────────┘            └───────────────┘
        │                             │                             │
        ▼                             ▼                             ▼
    torch 2.9.0               fastapi >=0.115           numpy, pillow
    transformers >=4.56       pydantic >=2.12           psutil, tqdm
    xformers 0.0.33           aiohttp                   msgspec
    flashinfer 0.5.2          uvicorn                   cloudpickle
```

## Dependency Categories

### 1. Deep Learning Framework

| Package | Version | Purpose | Update Frequency |
|---------|---------|---------|------------------|
| torch | ==2.9.0 | Core tensor operations | Quarterly |
| torchaudio | ==2.9.0 | Audio processing | With torch |
| torchvision | ==0.24.0 | Vision transforms | With torch |
| transformers | >=4.56.0, <5 | Model architectures | Monthly |
| tokenizers | >=0.21.1 | Fast tokenization | Monthly |
| xformers | ==0.0.33.post1 | Memory-efficient ops | With torch |
| flashinfer-python | ==0.5.2 | Attention kernels | Monthly |

### 2. API and Web

| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| fastapi[standard] | >=0.115.0 | HTTP server | Production-grade |
| pydantic | >=2.12.0 | Data validation | Type-safe |
| aiohttp | latest | Async HTTP client | |
| uvicorn | (fastapi dep) | ASGI server | |
| openai | >=1.99.1 | Client compatibility | SDK compat |

### 3. Serialization

| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| msgspec | latest | Fast serialization | IPC |
| protobuf | latest | LlamaTokenizer | HF models |
| gguf | >=0.17.0 | GGML format | llama.cpp compat |
| safetensors | (transformers) | Safe weights | Default format |
| cbor2 | latest | Cross-language | Hashing |

### 4. Observability

| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| prometheus_client | >=0.18.0 | Metrics | 40+ metrics |
| prometheus-fastapi-instrumentator | >=7.0.0 | HTTP metrics | Auto-instrument |
| opentelemetry-* | (optional) | Tracing | OTEL support |
| python-json-logger | latest | Structured logs | |

### 5. Quantization and Optimization

| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| compressed-tensors | ==0.12.2 | Weight compression | nm-vllm |
| lm-format-enforcer | ==0.11.3 | Constrained gen | Deprecated |
| xgrammar | ==0.1.25 | Grammar enforcement | New |
| outlines_core | ==0.2.11 | Structured output | |

### 6. Utilities

| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| numpy | latest | Numerical ops | Core |
| pillow | latest | Image processing | Multi-modal |
| opencv-python-headless | >=4.11.0 | Video IO | Vision |
| psutil | latest | System info | Resource monitor |
| tqdm | latest | Progress bars | |
| regex | latest | Fast regex | Performance |
| blake3 | latest | Fast hashing | Cache keys |
| filelock | >=3.16.1 | File locking | Download locks |

### 7. Platform-Specific

| Platform | Packages | Notes |
|----------|----------|-------|
| CUDA | numba ==0.61.2, triton | Speculative decoding |
| ROCm | torch-rocm, triton-rocm | AMD support |
| TPU | torch-xla, jax | Google TPU |
| XPU | intel-extension-for-pytorch | Intel GPU |
| CPU | intel-extension-for-pytorch | Intel CPU |

### 8. Distributed Computing

| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| ray[cgraph] | >=2.48.0 | Distributed exec | PP support |
| pyzmq | >=25.0.0 | Message passing | IPC |
| nccl | (torch) | GPU collective | NVIDIA |

## Build Dependencies

```toml
[build-system]
requires = [
    "cmake>=3.26.1",
    "ninja",
    "packaging>=24.2",
    "setuptools>=77.0.3,<81.0.0",
    "setuptools-scm>=8.0",
    "torch == 2.9.0",
    "wheel",
    "jinja2",
]
```

## Dependency Flow

```
User Code
    │
    ▼
┌─────────────────┐
│ vllm.LLM        │ ──────────────────────────────┐
│ vllm.serve      │                               │
└─────────────────┘                               │
    │                                             │
    ▼                                             ▼
┌─────────────────┐                    ┌─────────────────┐
│ transformers    │                    │ fastapi         │
│ tokenizers      │                    │ pydantic        │
└─────────────────┘                    └─────────────────┘
    │                                             │
    ▼                                             ▼
┌─────────────────┐                    ┌─────────────────┐
│ torch           │                    │ uvicorn         │
│ xformers        │                    │ aiohttp         │
│ flashinfer      │                    └─────────────────┘
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ CUDA Toolkit    │
│ cuDNN           │
│ NCCL            │
└─────────────────┘
```

## Version Pinning Strategy

### Exact Pins (Stability Critical)
- `torch == 2.9.0` - CUDA kernel compatibility
- `torchvision == 0.24.0` - torch compatibility
- `xformers == 0.0.33.post1` - torch compatibility
- `numba == 0.61.2` - Speculative decoding
- `compressed-tensors == 0.12.2` - Quantization format

### Range Pins (Feature Requirements)
- `transformers >= 4.56.0, < 5` - Model support
- `fastapi >= 0.115.0` - Form models
- `pydantic >= 2.12.0` - Feature requirements

### Unpinned (Stable APIs)
- `numpy`, `pillow`, `tqdm` - Stable, backward compatible

## Potential Issues

### 1. Heavy Dependencies

| Package | Size | Alternative | Notes |
|---------|------|-------------|-------|
| torch | ~2GB | None | Required |
| transformers | ~500MB | Partial load | Load specific models |
| opencv | ~100MB | pillow | For basic image ops |

### 2. Version Conflicts

- `torch` and `transformers` versions must be compatible
- `xformers` tied to specific `torch` versions
- `flashinfer` requires compatible CUDA version

### 3. Platform Considerations

- `xformers` only on Linux x86_64
- `xgrammar` limited platforms
- Some packages need compilation

## Security Notes

### Known Audit Status
- No critical CVEs in direct dependencies
- Regular updates via Dependabot
- License: All Apache 2.0 / MIT compatible

### Recommendations
1. Use virtual environments
2. Pin versions in production
3. Audit transitive dependencies
4. Update quarterly for security

## Updating Dependencies

### Safe Updates
```bash
# Update within version ranges
pip install --upgrade vllm
```

### Compatibility Testing
```bash
# Test with new versions
pip install vllm[dev]
pytest tests/basic_correctness/
```

### Version Constraints
- torch updates require kernel recompilation
- transformers updates may add new models
- Always test after updates

---

*Dependencies as of commit `67745d189fd981ee824bde35666a3737a962c031`*
