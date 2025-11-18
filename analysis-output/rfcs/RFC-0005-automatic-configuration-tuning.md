# RFC-0005: Automatic Configuration Tuning

**Status:** Draft
**Author:** Analysis Team
**Created:** November 18, 2025
**Commit Reference:** `67745d189fd981ee824bde35666a3737a962c031`

---

## Summary

Implement an automatic configuration tuning system that analyzes the model, hardware, and workload to recommend optimal settings, reducing time-to-production and improving out-of-box performance.

---

## Motivation

### Current Problems

1. **Manual Tuning:** Users must experiment with many parameters
2. **Hardware Variability:** Optimal settings differ by GPU type
3. **Model Variability:** Different models need different configs
4. **Workload Variability:** Throughput vs latency trade-offs
5. **Expert Knowledge:** Requires deep vLLM understanding

### User Impact

- Hours spent on configuration
- Suboptimal performance in production
- Frustration and support burden

### Opportunity

With access to model config, GPU specs, and workload hints, vLLM can automatically determine good configurations.

---

## Detailed Design

### 1. Auto-Tune Interface

```python
from vllm import LLM, AutoTune

# Basic: Auto-tune for balanced performance
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    auto_tune=True
)

# With hints
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    auto_tune=AutoTune(
        optimize_for="throughput",  # or "latency", "memory"
        expected_batch_size=100,
        expected_input_length=500,
        expected_output_length=200,
    )
)

# Show recommendations
config = AutoTune.analyze(
    model="meta-llama/Llama-2-7b-hf",
    optimize_for="throughput"
)
print(config.recommendations)
print(config.estimated_throughput)
```

### 2. Analysis Components

```python
# vllm/autotune/analyzer.py

class ConfigurationAnalyzer:
    """Analyzes system and recommends configuration."""

    def analyze(self, model_name, optimize_for, hints):
        # 1. Gather information
        model_config = self.get_model_config(model_name)
        hardware_info = self.get_hardware_info()

        # 2. Estimate memory requirements
        memory_estimate = self.estimate_memory(model_config)

        # 3. Determine parallelism
        parallelism = self.recommend_parallelism(
            model_config, hardware_info, memory_estimate
        )

        # 4. Tune batching parameters
        batching = self.recommend_batching(
            model_config, hardware_info, hints, optimize_for
        )

        # 5. Select attention backend
        attention = self.recommend_attention(
            model_config, hardware_info
        )

        # 6. Quantization recommendation
        quantization = self.recommend_quantization(
            model_config, hardware_info, memory_estimate, hints
        )

        return Configuration(
            parallelism=parallelism,
            batching=batching,
            attention=attention,
            quantization=quantization,
            estimated_performance=self.estimate_performance(...)
        )
```

### 3. Memory Estimation

```python
# vllm/autotune/memory.py

def estimate_memory(model_config, batch_size, seq_len):
    """Estimate GPU memory requirements."""

    # Model weights
    num_params = model_config.num_parameters
    bytes_per_param = 2 if model_config.dtype == "bf16" else 4
    weights_memory = num_params * bytes_per_param

    # KV cache
    kv_per_token = (
        2 * model_config.num_layers *
        model_config.num_kv_heads *
        model_config.head_dim *
        bytes_per_param
    )
    kv_memory = batch_size * seq_len * kv_per_token

    # Activations
    activation_memory = estimate_activations(
        model_config, batch_size, seq_len
    )

    # Overhead
    overhead = 0.1 * (weights_memory + kv_memory)

    return MemoryEstimate(
        weights=weights_memory,
        kv_cache=kv_memory,
        activations=activation_memory,
        overhead=overhead,
        total=weights_memory + kv_memory + activation_memory + overhead
    )
```

### 4. Performance Estimation

```python
# vllm/autotune/performance.py

def estimate_throughput(
    model_config,
    hardware_info,
    config,
):
    """Estimate throughput for configuration."""

    # Prefill: compute-bound
    prefill_time = estimate_prefill_time(
        model_config, hardware_info, config.batch_size, config.input_len
    )

    # Decode: memory-bound
    decode_time = estimate_decode_time(
        model_config, hardware_info, config.batch_size
    )

    # Total time
    total_time = prefill_time + config.output_len * decode_time

    # Throughput
    tokens = config.batch_size * (config.input_len + config.output_len)
    throughput = tokens / total_time

    return PerformanceEstimate(
        throughput=throughput,
        ttft=prefill_time,
        itl=decode_time,
        confidence=0.8  # Estimation confidence
    )
```

### 5. Hardware-Specific Rules

```python
# vllm/autotune/hardware.py

HARDWARE_PROFILES = {
    "A100-80GB": {
        "memory": 80 * 1e9,
        "bandwidth": 2039 * 1e9,  # GB/s
        "compute": 312 * 1e12,    # TFLOPS BF16
        "nvlink": True,
        "recommended_attention": "FLASH_ATTN",
        "recommended_quantization": "fp8",
    },
    "H100-80GB": {
        "memory": 80 * 1e9,
        "bandwidth": 3350 * 1e9,
        "compute": 989 * 1e12,
        "nvlink": True,
        "recommended_attention": "FLASHINFER",
        "recommended_quantization": "fp8",
    },
    # ...
}

def get_hardware_profile():
    """Detect GPU and return profile."""
    gpu_name = torch.cuda.get_device_name()
    for profile_name, profile in HARDWARE_PROFILES.items():
        if profile_name in gpu_name:
            return profile
    return HARDWARE_PROFILES["generic"]
```

### 6. Recommendation Output

```python
@dataclass
class TuningRecommendation:
    """Auto-tune recommendation."""

    # Recommended config
    tensor_parallel_size: int
    max_num_seqs: int
    max_model_len: int
    gpu_memory_utilization: float
    quantization: Optional[str]
    attention_backend: str
    enable_prefix_caching: bool

    # Explanation
    explanations: Dict[str, str]

    # Estimates
    estimated_throughput: float
    estimated_ttft: float
    estimated_memory: float

    # Confidence
    confidence: float

    def explain(self):
        """Print human-readable explanation."""
        print("Auto-Tune Recommendations:")
        for param, explanation in self.explanations.items():
            print(f"  {param}: {explanation}")
        print(f"\nEstimated throughput: {self.estimated_throughput:.0f} tok/s")
        print(f"Estimated TTFT: {self.estimated_ttft*1000:.1f}ms")
        print(f"Confidence: {self.confidence:.0%}")
```

---

## Example Usage

### Command Line

```bash
# Auto-tune and show recommendations
vllm tune meta-llama/Llama-2-70b-hf --optimize-for throughput

# Output:
# Model: meta-llama/Llama-2-70b-hf
# GPU: 8x NVIDIA A100 80GB
#
# Recommendations:
#   tensor_parallel_size: 8 (model too large for single GPU)
#   max_num_seqs: 256 (maximize batching for throughput)
#   quantization: fp8 (2x memory savings, <0.5% quality loss)
#   attention_backend: FLASH_ATTN (fastest for A100)
#   enable_prefix_caching: True (free performance)
#
# Estimated Performance:
#   Throughput: 4,200 tok/s
#   TTFT: 35ms
#   Memory: 72GB / 80GB (90%)
#
# To use:
#   vllm serve meta-llama/Llama-2-70b-hf --config tuned.yaml
```

### Programmatic

```python
from vllm.autotune import AutoTuner

tuner = AutoTuner()
result = tuner.analyze(
    model="meta-llama/Llama-2-70b-hf",
    optimize_for="throughput",
    hints={
        "expected_batch_size": 100,
        "expected_input_length": 500,
    }
)

# Use recommendations
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    **result.config
)

# Or apply directly
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    auto_tune=True
)
```

---

## Implementation Plan

### Phase 1: Memory Estimation (Week 1-2)
- Accurate memory model
- Hardware detection
- Basic recommendations

### Phase 2: Performance Model (Week 2-4)
- Throughput estimation
- Latency estimation
- Calibration data collection

### Phase 3: Recommendation Engine (Week 4-6)
- Rule-based recommendations
- Optimization targets
- Explanation generation

### Phase 4: Integration (Week 6-8)
- CLI integration
- API integration
- Documentation

---

## Backwards Compatibility

- Purely additive feature
- Existing configurations work unchanged
- Auto-tune disabled by default

---

## Alternatives Considered

### 1. Online Tuning (A/B testing)

**Considered for Phase 2:** Run experiments to refine estimates

### 2. ML-based Tuning

**Considered for Phase 2:** Learn from production data

### 3. External Tool

**Rejected:** Should be integrated for best UX

---

## Open Questions

1. How to handle multi-model deployments?
2. Should auto-tune run benchmarks to calibrate?
3. How to update hardware profiles?

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Time to optimal config | Hours | Minutes |
| Config performance vs optimal | Variable | >90% |
| User satisfaction | - | 4/5 |

---

## Effort Estimate

- **Total:** 25 dev-days
- **Risk:** Medium (estimation accuracy)
- **Required Approvals:** Core team

---

*RFC based on analysis of commit `67745d189fd981ee824bde35666a3737a962c031`*
