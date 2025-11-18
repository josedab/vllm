# SPDX-License-Identifier: Apache-2.0
"""Tests for the autotune module."""

import pytest

from vllm.autotune import (
    AutoTune,
    AutoTuner,
    ConfigurationAnalyzer,
    HardwareInfo,
    HardwareProfile,
    MemoryEstimate,
    ModelConfig,
    PerformanceEstimate,
    TuningHints,
    TuningRecommendation,
    calculate_max_batch_size,
    calculate_max_model_len,
    estimate_memory,
    estimate_throughput,
    get_hardware_info,
    get_hardware_profile,
)


class TestHardwareProfile:
    """Tests for hardware profile detection."""

    def test_get_profile_a100_80gb(self):
        """Test A100 80GB profile detection."""
        profile = get_hardware_profile("NVIDIA A100-SXM4-80GB")
        assert profile.name == "A100-80GB"
        assert profile.memory == 80 * (1024**3)
        assert profile.nvlink is True
        assert profile.recommended_attention == "FLASH_ATTN"

    def test_get_profile_h100(self):
        """Test H100 profile detection."""
        profile = get_hardware_profile("NVIDIA H100 80GB HBM3")
        assert profile.name == "H100-80GB"
        assert profile.recommended_attention == "FLASHINFER"
        assert profile.recommended_quantization == "fp8"

    def test_get_profile_rtx_4090(self):
        """Test RTX 4090 profile detection."""
        profile = get_hardware_profile("NVIDIA GeForce RTX 4090")
        assert profile.name == "RTX 4090"
        assert profile.memory == 24 * (1024**3)
        assert profile.nvlink is False

    def test_get_profile_unknown(self):
        """Test fallback to generic profile."""
        profile = get_hardware_profile("Unknown GPU XYZ")
        assert profile.name == "generic"

    def test_hardware_profile_properties(self):
        """Test HardwareProfile property calculations."""
        profile = get_hardware_profile("NVIDIA A100-SXM4-80GB")
        assert profile.memory_gb == 80.0
        assert profile.compute_tflops == 312.0


class TestMemoryEstimation:
    """Tests for memory estimation."""

    @pytest.fixture
    def model_config(self):
        """Create a test model config (7B-like)."""
        return ModelConfig(
            num_parameters=7_000_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
            head_dim=128,
            intermediate_size=11008,
            vocab_size=32000,
        )

    def test_estimate_memory(self, model_config):
        """Test memory estimation."""
        estimate = estimate_memory(
            model_config,
            batch_size=32,
            seq_len=2048,
        )

        assert isinstance(estimate, MemoryEstimate)
        assert estimate.weights > 0
        assert estimate.kv_cache > 0
        assert estimate.activations > 0
        assert estimate.total > 0
        assert estimate.total == (
            estimate.weights
            + estimate.kv_cache
            + estimate.activations
            + estimate.overhead
        )

    def test_estimate_memory_quantized(self, model_config):
        """Test memory estimation with quantization."""
        fp16_estimate = estimate_memory(model_config, 32, 2048)
        awq_estimate = estimate_memory(model_config, 32, 2048, "awq")

        # Quantized model should use less memory for weights
        assert awq_estimate.weights < fp16_estimate.weights

    def test_calculate_max_batch_size(self, model_config):
        """Test max batch size calculation."""
        # 80GB GPU
        available_memory = 80 * (1024**3)
        max_batch = calculate_max_batch_size(
            model_config,
            available_memory,
            seq_len=2048,
        )

        assert max_batch > 0
        assert max_batch <= 1024  # Reasonable upper bound

    def test_calculate_max_model_len(self, model_config):
        """Test max model length calculation."""
        available_memory = 80 * (1024**3)
        max_len = calculate_max_model_len(
            model_config,
            available_memory,
            batch_size=32,
        )

        assert max_len > 0
        assert max_len <= model_config.max_position_embeddings

    def test_memory_estimate_properties(self, model_config):
        """Test MemoryEstimate property calculations."""
        estimate = estimate_memory(model_config, 32, 2048)
        assert estimate.weights_gb > 0
        assert estimate.kv_cache_gb > 0
        assert estimate.total_gb > 0


class TestPerformanceEstimation:
    """Tests for performance estimation."""

    @pytest.fixture
    def model_config(self):
        """Create a test model config."""
        return ModelConfig(
            num_parameters=7_000_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
            head_dim=128,
            intermediate_size=11008,
            vocab_size=32000,
        )

    @pytest.fixture
    def hardware_info(self):
        """Create test hardware info."""
        profile = get_hardware_profile("NVIDIA A100-SXM4-80GB")
        return HardwareInfo(
            profile=profile,
            num_gpus=8,
            gpu_name="NVIDIA A100-SXM4-80GB",
            total_memory=8 * 80 * (1024**3),
        )

    def test_estimate_throughput(self, model_config, hardware_info):
        """Test throughput estimation."""
        estimate = estimate_throughput(
            model_config,
            hardware_info,
            batch_size=32,
            input_len=500,
            output_len=200,
        )

        assert isinstance(estimate, PerformanceEstimate)
        assert estimate.throughput > 0
        assert estimate.ttft > 0
        assert estimate.itl > 0
        assert estimate.confidence > 0

    def test_performance_estimate_properties(self, model_config, hardware_info):
        """Test PerformanceEstimate property calculations."""
        estimate = estimate_throughput(
            model_config,
            hardware_info,
            batch_size=32,
            input_len=500,
            output_len=200,
        )

        assert estimate.throughput_k == estimate.throughput / 1000
        assert estimate.ttft_ms == estimate.ttft * 1000
        assert estimate.itl_ms == estimate.itl * 1000


class TestConfigurationAnalyzer:
    """Tests for the configuration analyzer."""

    def test_analyze_basic(self):
        """Test basic analysis with default hints."""
        analyzer = ConfigurationAnalyzer()

        # Use a small model for testing
        result = analyzer.analyze(
            model_name="gpt2",
            optimize_for="throughput",
        )

        assert isinstance(result, TuningRecommendation)
        assert result.tensor_parallel_size >= 1
        assert result.max_num_seqs > 0
        assert result.max_model_len > 0
        assert result.gpu_memory_utilization > 0
        assert result.enable_prefix_caching is True
        assert result.confidence > 0

    def test_analyze_with_hints(self):
        """Test analysis with custom hints."""
        analyzer = ConfigurationAnalyzer()

        hints = TuningHints(
            expected_batch_size=100,
            expected_input_length=1000,
            expected_output_length=500,
        )

        result = analyzer.analyze(
            model_name="gpt2",
            optimize_for="latency",
            hints=hints,
        )

        assert isinstance(result, TuningRecommendation)
        # Latency optimization should have lower batch size
        assert result.max_num_seqs <= 64

    def test_analyze_optimize_for_memory(self):
        """Test analysis optimized for memory."""
        analyzer = ConfigurationAnalyzer()

        result = analyzer.analyze(
            model_name="gpt2",
            optimize_for="memory",
        )

        assert isinstance(result, TuningRecommendation)

    def test_to_engine_args(self):
        """Test conversion to engine args."""
        analyzer = ConfigurationAnalyzer()

        result = analyzer.analyze(
            model_name="gpt2",
            optimize_for="throughput",
        )

        engine_args = result.to_engine_args()

        assert "tensor_parallel_size" in engine_args
        assert "max_num_seqs" in engine_args
        assert "max_model_len" in engine_args
        assert "gpu_memory_utilization" in engine_args
        assert "enable_prefix_caching" in engine_args

    def test_explain(self):
        """Test recommendation explanation."""
        analyzer = ConfigurationAnalyzer()

        result = analyzer.analyze(
            model_name="gpt2",
            optimize_for="throughput",
        )

        explanation = result.explain()

        assert isinstance(explanation, str)
        assert "Auto-Tune Recommendations" in explanation
        assert "throughput" in explanation.lower()


class TestAutoTune:
    """Tests for the AutoTune class."""

    def test_autotune_dataclass(self):
        """Test AutoTune dataclass creation."""
        autotune = AutoTune(
            optimize_for="throughput",
            expected_batch_size=100,
            expected_input_length=500,
            expected_output_length=200,
        )

        assert autotune.optimize_for == "throughput"
        assert autotune.expected_batch_size == 100
        assert autotune.expected_input_length == 500
        assert autotune.expected_output_length == 200

    def test_autotune_analyze_static(self):
        """Test AutoTune.analyze static method."""
        result = AutoTune.analyze(
            model="gpt2",
            optimize_for="throughput",
        )

        assert isinstance(result, TuningRecommendation)


class TestAutoTuner:
    """Tests for the AutoTuner class."""

    def test_autotuner_analyze(self):
        """Test AutoTuner.analyze method."""
        tuner = AutoTuner()
        result = tuner.analyze(
            model="gpt2",
            optimize_for="throughput",
        )

        assert isinstance(result, TuningRecommendation)

    def test_autotuner_analyze_with_hints(self):
        """Test AutoTuner.analyze with hints dict."""
        tuner = AutoTuner()
        result = tuner.analyze(
            model="gpt2",
            optimize_for="latency",
            hints={
                "expected_batch_size": 50,
                "expected_input_length": 1000,
            },
        )

        assert isinstance(result, TuningRecommendation)

    def test_get_engine_args_bool(self):
        """Test get_engine_args with True."""
        tuner = AutoTuner()
        args = tuner.get_engine_args("gpt2", True)

        assert isinstance(args, dict)
        assert "tensor_parallel_size" in args

    def test_get_engine_args_autotune(self):
        """Test get_engine_args with AutoTune instance."""
        tuner = AutoTuner()
        autotune = AutoTune(
            optimize_for="throughput",
            expected_batch_size=100,
        )
        args = tuner.get_engine_args("gpt2", autotune)

        assert isinstance(args, dict)

    def test_get_engine_args_invalid(self):
        """Test get_engine_args with invalid input."""
        tuner = AutoTuner()
        args = tuner.get_engine_args("gpt2", False)

        assert args == {}


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_model_config_creation(self):
        """Test ModelConfig creation."""
        config = ModelConfig(
            num_parameters=7_000_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
            head_dim=128,
            intermediate_size=11008,
            vocab_size=32000,
        )

        assert config.num_parameters == 7_000_000_000
        assert config.num_layers == 32

    def test_model_config_from_hf(self):
        """Test ModelConfig.from_hf_config."""
        # This test uses a mock config
        class MockConfig:
            hidden_size = 4096
            num_attention_heads = 32
            num_key_value_heads = 8
            num_hidden_layers = 32
            intermediate_size = 11008
            vocab_size = 32000
            max_position_embeddings = 4096

        config = ModelConfig.from_hf_config(MockConfig())

        assert config.hidden_size == 4096
        assert config.num_kv_heads == 8
        assert config.num_layers == 32


class TestTuningRecommendation:
    """Tests for TuningRecommendation."""

    def test_recommendation_creation(self):
        """Test TuningRecommendation creation."""
        rec = TuningRecommendation(
            tensor_parallel_size=8,
            max_num_seqs=256,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            quantization="fp8",
            attention_backend="FLASH_ATTN",
            enable_prefix_caching=True,
            explanations={"tensor_parallel_size": "test"},
            estimated_throughput=1000,
            estimated_ttft=0.05,
            estimated_memory=70,
            confidence=0.8,
        )

        assert rec.tensor_parallel_size == 8
        assert rec.quantization == "fp8"

    def test_recommendation_to_engine_args(self):
        """Test conversion to engine args."""
        rec = TuningRecommendation(
            tensor_parallel_size=8,
            max_num_seqs=256,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            quantization="fp8",
            attention_backend="FLASH_ATTN",
            enable_prefix_caching=True,
        )

        args = rec.to_engine_args()

        assert args["tensor_parallel_size"] == 8
        assert args["quantization"] == "fp8"
        assert args["enable_prefix_caching"] is True

    def test_recommendation_no_quantization(self):
        """Test engine args without quantization."""
        rec = TuningRecommendation(
            tensor_parallel_size=1,
            max_num_seqs=128,
            max_model_len=2048,
            gpu_memory_utilization=0.9,
            quantization=None,
            attention_backend="FLASH_ATTN",
            enable_prefix_caching=True,
        )

        args = rec.to_engine_args()

        assert "quantization" not in args
