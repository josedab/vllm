# SPDX-License-Identifier: Apache-2.0
"""Auto-tuning system for vLLM configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from vllm.autotune.analyzer import (
    ConfigurationAnalyzer,
    OptimizeFor,
    TuningHints,
    TuningRecommendation,
)


@dataclass
class AutoTune:
    """Auto-tune configuration hints.

    Use this class to provide hints to the auto-tuning system.

    Example:
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
                optimize_for="throughput",
                expected_batch_size=100,
                expected_input_length=500,
                expected_output_length=200,
            )
        )
        ```
    """

    optimize_for: OptimizeFor = "throughput"
    expected_batch_size: Optional[int] = None
    expected_input_length: int = 500
    expected_output_length: int = 200
    target_throughput: Optional[float] = None
    target_latency: Optional[float] = None

    @staticmethod
    def analyze(
        model: str,
        optimize_for: OptimizeFor = "throughput",
        expected_batch_size: Optional[int] = None,
        expected_input_length: int = 500,
        expected_output_length: int = 200,
    ) -> TuningRecommendation:
        """Analyze model and return recommendations.

        Args:
            model: Model name or path.
            optimize_for: Optimization target.
            expected_batch_size: Expected batch size.
            expected_input_length: Expected input length.
            expected_output_length: Expected output length.

        Returns:
            TuningRecommendation with configuration and explanations.
        """
        tuner = AutoTuner()
        return tuner.analyze(
            model=model,
            optimize_for=optimize_for,
            hints={
                "expected_batch_size": expected_batch_size,
                "expected_input_length": expected_input_length,
                "expected_output_length": expected_output_length,
            },
        )


class AutoTuner:
    """Main auto-tuning orchestrator.

    Example:
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

        # Show recommendations
        print(result.explain())

        # Use recommendations
        from vllm import LLM
        llm = LLM(
            model="meta-llama/Llama-2-70b-hf",
            **result.to_engine_args()
        )
        ```
    """

    def __init__(self):
        """Initialize auto-tuner."""
        self.analyzer = ConfigurationAnalyzer()

    def analyze(
        self,
        model: str,
        optimize_for: OptimizeFor = "throughput",
        hints: Optional[dict[str, Any]] = None,
    ) -> TuningRecommendation:
        """Analyze model and return recommendations.

        Args:
            model: Model name or path.
            optimize_for: Optimization target ("throughput", "latency", "memory").
            hints: Optional hints dict.

        Returns:
            TuningRecommendation with configuration and explanations.
        """
        tuning_hints = TuningHints()
        if hints:
            if "expected_batch_size" in hints:
                tuning_hints.expected_batch_size = hints["expected_batch_size"]
            if "expected_input_length" in hints:
                tuning_hints.expected_input_length = hints["expected_input_length"]
            if "expected_output_length" in hints:
                tuning_hints.expected_output_length = hints["expected_output_length"]
            if "target_throughput" in hints:
                tuning_hints.target_throughput = hints["target_throughput"]
            if "target_latency" in hints:
                tuning_hints.target_latency = hints["target_latency"]

        return self.analyzer.analyze(model, optimize_for, tuning_hints)

    def get_engine_args(
        self,
        model: str,
        auto_tune: Union[bool, AutoTune],
    ) -> dict[str, Any]:
        """Get engine args based on auto-tune settings.

        Args:
            model: Model name or path.
            auto_tune: Auto-tune setting (True or AutoTune instance).

        Returns:
            Dict of engine args to apply.
        """
        if auto_tune is True:
            # Use default hints
            recommendation = self.analyze(model)
        elif isinstance(auto_tune, AutoTune):
            # Use provided hints
            recommendation = self.analyze(
                model=model,
                optimize_for=auto_tune.optimize_for,
                hints={
                    "expected_batch_size": auto_tune.expected_batch_size,
                    "expected_input_length": auto_tune.expected_input_length,
                    "expected_output_length": auto_tune.expected_output_length,
                    "target_throughput": auto_tune.target_throughput,
                    "target_latency": auto_tune.target_latency,
                },
            )
        else:
            return {}

        return recommendation.to_engine_args()


def apply_auto_tune(
    model: str,
    engine_args: dict[str, Any],
    auto_tune: Union[bool, AutoTune],
) -> dict[str, Any]:
    """Apply auto-tune recommendations to engine args.

    This function modifies engine_args in place and returns it.

    Args:
        model: Model name or path.
        engine_args: Engine args dict.
        auto_tune: Auto-tune setting.

    Returns:
        Modified engine args dict.
    """
    tuner = AutoTuner()
    tuned_args = tuner.get_engine_args(model, auto_tune)

    # Only apply tuned args if not explicitly set
    for key, value in tuned_args.items():
        if key not in engine_args or engine_args[key] is None:
            engine_args[key] = value

    return engine_args
