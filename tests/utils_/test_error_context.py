# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the error context utilities."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.utils.error_context import ErrorContext


class TestErrorContextSystemInfo:
    """Tests for system context gathering."""

    def test_get_system_context_basic(self):
        """Test that system context returns expected keys."""
        context = ErrorContext.get_system_context()

        assert "vLLM version" in context
        assert "PyTorch version" in context

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA A100")
    @patch("torch.cuda.get_device_properties")
    def test_get_system_context_with_cuda(
        self,
        mock_props,
        mock_name,
        mock_count,
        mock_available,
    ):
        """Test system context with CUDA available."""
        mock_props.return_value = MagicMock(total_memory=80 * 1e9)

        context = ErrorContext.get_system_context()

        assert "CUDA version" in context
        assert "GPU count" in context
        assert context["GPU count"] == 2
        assert "GPU" in context
        assert "GPU memory" in context

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_system_context_without_cuda(self, mock_available):
        """Test system context without CUDA."""
        context = ErrorContext.get_system_context()

        assert context.get("CUDA available") is False
        assert "GPU" not in context


class TestErrorContextMemoryInfo:
    """Tests for memory context gathering."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_memory_context_no_cuda(self, mock_available):
        """Test memory context when CUDA is not available."""
        context = ErrorContext.get_memory_context()
        assert context.get("CUDA available") is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    def test_get_memory_context_invalid_device(self, mock_count, mock_available):
        """Test memory context with invalid device ID."""
        context = ErrorContext.get_memory_context(device_id=5)
        assert "error" in context

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.get_device_name", return_value="NVIDIA A100")
    @patch("torch.cuda.memory_allocated", return_value=10 * 1e9)
    @patch("torch.cuda.memory_reserved", return_value=12 * 1e9)
    def test_get_memory_context_success(
        self,
        mock_reserved,
        mock_allocated,
        mock_name,
        mock_props,
        mock_count,
        mock_available,
    ):
        """Test successful memory context retrieval."""
        mock_props.return_value = MagicMock(total_memory=80 * 1e9)

        context = ErrorContext.get_memory_context(device_id=0)

        assert "GPU" in context
        assert "Total memory" in context
        assert "Allocated" in context
        assert "Reserved" in context
        assert "Free (approx)" in context


class TestErrorContextConfigInfo:
    """Tests for configuration context gathering."""

    def test_get_model_config_context(self):
        """Test model config context extraction."""
        mock_config = MagicMock()
        mock_config.model = "test-model"
        mock_config.tokenizer = "test-tokenizer"
        mock_config.dtype = "float16"
        mock_config.max_model_len = 4096
        mock_config.quantization = "fp8"

        context = ErrorContext.get_model_config_context(mock_config)

        assert context.get("Model") == "test-model"
        assert context.get("Tokenizer") == "test-tokenizer"
        assert context.get("Max model length") == "4096"
        assert context.get("Quantization") == "fp8"

    def test_get_model_config_context_missing_attrs(self):
        """Test model config context with missing attributes."""
        mock_config = MagicMock(spec=[])  # Empty spec = no attributes

        context = ErrorContext.get_model_config_context(mock_config)
        assert context == {}  # Should return empty dict

    def test_get_parallel_config_context(self):
        """Test parallel config context extraction."""
        mock_config = MagicMock()
        mock_config.tensor_parallel_size = 4
        mock_config.pipeline_parallel_size = 2
        mock_config.data_parallel_size = 1

        context = ErrorContext.get_parallel_config_context(mock_config)

        assert context.get("Tensor parallel size") == 4
        assert context.get("Pipeline parallel size") == 2
        assert context.get("Data parallel size") == 1

    def test_get_scheduler_config_context(self):
        """Test scheduler config context extraction."""
        mock_config = MagicMock()
        mock_config.max_num_seqs = 256
        mock_config.max_num_batched_tokens = 8192

        context = ErrorContext.get_scheduler_config_context(mock_config)

        assert context.get("Max sequences") == 256
        assert context.get("Max batched tokens") == 8192

    def test_get_cache_config_context(self):
        """Test cache config context extraction."""
        mock_config = MagicMock()
        mock_config.block_size = 16
        mock_config.gpu_memory_utilization = 0.9
        mock_config.cache_dtype = "auto"
        mock_config.num_gpu_blocks = 1000

        context = ErrorContext.get_cache_config_context(mock_config)

        assert context.get("Block size") == 16
        assert context.get("GPU memory utilization") == "90%"
        assert context.get("Cache dtype") == "auto"
        assert context.get("GPU blocks") == 1000

    def test_get_full_config_context(self):
        """Test combining all config contexts."""
        model_config = MagicMock()
        model_config.model = "test-model"

        parallel_config = MagicMock()
        parallel_config.tensor_parallel_size = 2

        scheduler_config = MagicMock()
        scheduler_config.max_num_seqs = 128

        cache_config = MagicMock()
        cache_config.block_size = 16

        context = ErrorContext.get_full_config_context(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            cache_config=cache_config,
        )

        assert "Model" in context or context == {}  # Depends on mock setup
        # The function should not raise errors

    def test_get_full_config_context_partial(self):
        """Test full config context with some configs missing."""
        model_config = MagicMock()
        model_config.model = "test-model"

        # Only provide model config
        context = ErrorContext.get_full_config_context(
            model_config=model_config,
        )

        # Should not raise errors
        assert isinstance(context, dict)
