# SPDX-License-Identifier: Apache-2.0
"""Automatic configuration tuning for vLLM.

This module provides tools for automatically analyzing model and hardware
characteristics to recommend optimal vLLM configurations.

Example:
    ```python
    from vllm import LLM
    from vllm.autotune import AutoTune

    # Basic: Auto-tune for balanced performance
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        auto_tune=True
    )

    # With optimization hints
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        auto_tune=AutoTune(
            optimize_for="throughput",
            expected_batch_size=100,
            expected_input_length=500,
            expected_output_length=200,
        )
    )

    # Analyze without creating LLM
    from vllm.autotune import AutoTuner

    tuner = AutoTuner()
    result = tuner.analyze(
        model="meta-llama/Llama-2-70b-hf",
        optimize_for="throughput"
    )
    print(result.explain())
    ```
"""

from vllm.autotune.analyzer import (
    ConfigurationAnalyzer,
    OptimizeFor,
    TuningHints,
    TuningRecommendation,
)
from vllm.autotune.hardware import (
    HARDWARE_PROFILES,
    HardwareInfo,
    HardwareProfile,
    get_hardware_info,
    get_hardware_profile,
)
from vllm.autotune.memory import (
    MemoryEstimate,
    ModelConfig,
    calculate_max_batch_size,
    calculate_max_model_len,
    estimate_memory,
)
from vllm.autotune.performance import (
    PerformanceEstimate,
    estimate_latency,
    estimate_throughput,
)
from vllm.autotune.tuner import AutoTune, AutoTuner, apply_auto_tune

__all__ = [
    # Main API
    "AutoTune",
    "AutoTuner",
    "apply_auto_tune",
    # Analyzer
    "ConfigurationAnalyzer",
    "TuningRecommendation",
    "TuningHints",
    "OptimizeFor",
    # Hardware
    "HardwareProfile",
    "HardwareInfo",
    "HARDWARE_PROFILES",
    "get_hardware_info",
    "get_hardware_profile",
    # Memory
    "MemoryEstimate",
    "ModelConfig",
    "estimate_memory",
    "calculate_max_batch_size",
    "calculate_max_model_len",
    # Performance
    "PerformanceEstimate",
    "estimate_throughput",
    "estimate_latency",
]
