# SPDX-License-Identifier: Apache-2.0
"""Configuration analyzer for auto-tuning."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from vllm.autotune.hardware import HardwareInfo, get_hardware_info
from vllm.autotune.memory import (
    ModelConfig,
    calculate_max_batch_size,
    calculate_max_model_len,
    estimate_memory,
)
from vllm.autotune.performance import (
    estimate_throughput,
    optimize_for_latency,
    optimize_for_throughput,
)


OptimizeFor = Literal["throughput", "latency", "memory"]


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
    explanations: dict[str, str] = field(default_factory=dict)

    # Estimates
    estimated_throughput: float = 0.0
    estimated_ttft: float = 0.0
    estimated_memory: float = 0.0

    # Confidence
    confidence: float = 0.8

    def explain(self) -> str:
        """Get human-readable explanation."""
        lines = ["Auto-Tune Recommendations:"]
        for param, explanation in self.explanations.items():
            lines.append(f"  {param}: {explanation}")
        lines.append("")
        lines.append(f"Estimated throughput: {self.estimated_throughput:.0f} tok/s")
        lines.append(f"Estimated TTFT: {self.estimated_ttft * 1000:.1f}ms")
        lines.append(f"Estimated memory: {self.estimated_memory:.1f}GB")
        lines.append(f"Confidence: {self.confidence:.0%}")
        return "\n".join(lines)

    def to_engine_args(self) -> dict[str, Any]:
        """Convert to engine args dict."""
        args = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": self.enable_prefix_caching,
        }
        if self.quantization:
            args["quantization"] = self.quantization
        return args


@dataclass
class TuningHints:
    """Hints for auto-tuning."""

    expected_batch_size: Optional[int] = None
    expected_input_length: int = 500
    expected_output_length: int = 200
    target_throughput: Optional[float] = None
    target_latency: Optional[float] = None


class ConfigurationAnalyzer:
    """Analyzes system and recommends configuration."""

    def __init__(self):
        """Initialize analyzer."""
        self.hardware_info: Optional[HardwareInfo] = None
        self.model_config: Optional[ModelConfig] = None

    def get_hardware_info(self) -> HardwareInfo:
        """Get hardware info, caching result."""
        if self.hardware_info is None:
            self.hardware_info = get_hardware_info()
        return self.hardware_info

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get model configuration.

        Args:
            model_name: Model name or path.

        Returns:
            ModelConfig for the model.
        """
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model_config = ModelConfig.from_hf_config(hf_config)
        except Exception:
            # Fallback to default config for unknown models
            self.model_config = ModelConfig(
                num_parameters=7_000_000_000,
                num_layers=32,
                hidden_size=4096,
                num_attention_heads=32,
                num_kv_heads=32,
                head_dim=128,
                intermediate_size=11008,
                vocab_size=32000,
            )
        return self.model_config

    def recommend_parallelism(
        self,
        model_config: ModelConfig,
        hardware_info: HardwareInfo,
        memory_estimate: int,
    ) -> dict[str, Any]:
        """Recommend parallelism settings.

        Args:
            model_config: Model configuration.
            hardware_info: Hardware information.
            memory_estimate: Total memory estimate.

        Returns:
            Dict with parallelism recommendations.
        """
        explanations = {}

        # Check if model fits on single GPU
        single_gpu_memory = hardware_info.profile.memory * 0.9
        if memory_estimate <= single_gpu_memory:
            tp_size = 1
            explanations["tensor_parallel_size"] = (
                "1 (model fits on single GPU)"
            )
        else:
            # Need tensor parallelism
            tp_size = 1
            while tp_size < hardware_info.num_gpus:
                if memory_estimate / tp_size <= single_gpu_memory:
                    break
                tp_size *= 2

            tp_size = min(tp_size, hardware_info.num_gpus)
            explanations["tensor_parallel_size"] = (
                f"{tp_size} (model requires {tp_size} GPUs for memory)"
            )

        return {
            "tensor_parallel_size": tp_size,
            "explanations": explanations,
        }

    def recommend_batching(
        self,
        model_config: ModelConfig,
        hardware_info: HardwareInfo,
        hints: TuningHints,
        optimize_for: OptimizeFor,
    ) -> dict[str, Any]:
        """Recommend batching settings.

        Args:
            model_config: Model configuration.
            hardware_info: Hardware information.
            hints: Tuning hints.
            optimize_for: Optimization target.

        Returns:
            Dict with batching recommendations.
        """
        explanations = {}

        seq_len = hints.expected_input_length + hints.expected_output_length
        available_memory = hardware_info.profile.memory

        if optimize_for == "throughput":
            # Maximize batch size for throughput
            max_batch = calculate_max_batch_size(
                model_config, available_memory, seq_len
            )
            # Cap at reasonable value
            max_num_seqs = min(max_batch, 256)
            explanations["max_num_seqs"] = (
                f"{max_num_seqs} (maximize batching for throughput)"
            )
        elif optimize_for == "latency":
            # Lower batch size for latency
            max_num_seqs = 32
            explanations["max_num_seqs"] = (
                f"{max_num_seqs} (lower batch for latency optimization)"
            )
        else:  # memory
            # Conservative batch size
            max_batch = calculate_max_batch_size(
                model_config, available_memory, seq_len
            )
            max_num_seqs = max(1, max_batch // 2)
            explanations["max_num_seqs"] = (
                f"{max_num_seqs} (conservative for memory optimization)"
            )

        # Calculate max model length
        max_model_len = calculate_max_model_len(
            model_config, available_memory, max_num_seqs
        )
        explanations["max_model_len"] = (
            f"{max_model_len} (maximum context that fits in memory)"
        )

        return {
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
            "explanations": explanations,
        }

    def recommend_attention(
        self,
        model_config: ModelConfig,
        hardware_info: HardwareInfo,
    ) -> dict[str, Any]:
        """Recommend attention backend.

        Args:
            model_config: Model configuration.
            hardware_info: Hardware information.

        Returns:
            Dict with attention recommendations.
        """
        backend = hardware_info.profile.recommended_attention
        explanation = f"{backend} (fastest for {hardware_info.profile.name})"

        return {
            "attention_backend": backend,
            "explanations": {"attention_backend": explanation},
        }

    def recommend_quantization(
        self,
        model_config: ModelConfig,
        hardware_info: HardwareInfo,
        memory_estimate: int,
        hints: TuningHints,
    ) -> dict[str, Any]:
        """Recommend quantization.

        Args:
            model_config: Model configuration.
            hardware_info: Hardware information.
            memory_estimate: Memory estimate.
            hints: Tuning hints.

        Returns:
            Dict with quantization recommendations.
        """
        # Check if quantization is needed
        available_memory = hardware_info.total_memory

        if memory_estimate > available_memory * 0.9:
            # Need quantization
            quant = hardware_info.profile.recommended_quantization
            explanation = (
                f"{quant} (needed to fit model in memory, <1% quality loss)"
            )
        elif memory_estimate > available_memory * 0.7:
            # Could benefit from quantization
            quant = hardware_info.profile.recommended_quantization
            explanation = f"{quant} (2x memory savings, <0.5% quality loss)"
        else:
            # No quantization needed
            quant = None
            explanation = "None (model fits comfortably in memory)"

        return {
            "quantization": quant,
            "explanations": {"quantization": explanation},
        }

    def estimate_performance(
        self,
        model_config: ModelConfig,
        hardware_info: HardwareInfo,
        config: dict[str, Any],
        hints: TuningHints,
    ) -> dict[str, float]:
        """Estimate performance for configuration.

        Args:
            model_config: Model configuration.
            hardware_info: Hardware information.
            config: Recommended configuration.
            hints: Tuning hints.

        Returns:
            Dict with performance estimates.
        """
        estimate = estimate_throughput(
            model_config,
            hardware_info,
            batch_size=config.get("max_num_seqs", 128),
            input_len=hints.expected_input_length,
            output_len=hints.expected_output_length,
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
        )

        memory = estimate_memory(
            model_config,
            config.get("max_num_seqs", 128),
            config.get("max_model_len", 4096),
            config.get("quantization"),
        )

        return {
            "throughput": estimate.throughput,
            "ttft": estimate.ttft,
            "memory_gb": memory.total_gb,
            "confidence": estimate.confidence,
        }

    def analyze(
        self,
        model_name: str,
        optimize_for: OptimizeFor = "throughput",
        hints: Optional[TuningHints] = None,
    ) -> TuningRecommendation:
        """Analyze and recommend configuration.

        Args:
            model_name: Model name or path.
            optimize_for: Optimization target.
            hints: Tuning hints.

        Returns:
            TuningRecommendation with configuration and explanations.
        """
        if hints is None:
            hints = TuningHints()

        # 1. Gather information
        model_config = self.get_model_config(model_name)
        hardware_info = self.get_hardware_info()

        # 2. Estimate memory requirements
        memory_estimate = estimate_memory(
            model_config,
            hints.expected_batch_size or 128,
            hints.expected_input_length + hints.expected_output_length,
        )

        # 3. Determine parallelism
        parallelism = self.recommend_parallelism(
            model_config, hardware_info, memory_estimate.total
        )

        # 4. Tune batching parameters
        batching = self.recommend_batching(
            model_config, hardware_info, hints, optimize_for
        )

        # 5. Select attention backend
        attention = self.recommend_attention(model_config, hardware_info)

        # 6. Quantization recommendation
        quantization = self.recommend_quantization(
            model_config, hardware_info, memory_estimate.total, hints
        )

        # Combine recommendations
        config = {
            **parallelism,
            **batching,
            **attention,
            **quantization,
        }

        # Merge explanations
        explanations = {}
        for key in ["parallelism", "batching", "attention", "quantization"]:
            result = locals()[key.split("_")[0]]
            if "explanations" in result:
                explanations.update(result["explanations"])

        # 7. Estimate performance
        performance = self.estimate_performance(
            model_config, hardware_info, config, hints
        )

        return TuningRecommendation(
            tensor_parallel_size=config["tensor_parallel_size"],
            max_num_seqs=config["max_num_seqs"],
            max_model_len=config["max_model_len"],
            gpu_memory_utilization=0.9,
            quantization=config["quantization"],
            attention_backend=config["attention_backend"],
            enable_prefix_caching=True,
            explanations=explanations,
            estimated_throughput=performance["throughput"],
            estimated_ttft=performance["ttft"],
            estimated_memory=performance["memory_gb"],
            confidence=performance["confidence"],
        )
