# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the enhanced exception hierarchy."""

import pytest

from vllm.exceptions import (
    VLLMConfigurationError,
    VLLMDistributedError,
    VLLMError,
    VLLMMemoryError,
    VLLMModelLoadError,
    VLLMQuantizationError,
    VLLMTokenizerError,
)


class TestVLLMError:
    """Tests for the VLLMError base class."""

    def test_basic_error(self):
        """Test creating a basic error with just a message."""
        error = VLLMError("Test error message")
        assert "Test error message" in str(error)
        assert error.message == "Test error message"
        assert error.context == {}
        assert error.solutions == []
        assert error.docs_url is None

    def test_error_with_context(self):
        """Test creating an error with context."""
        context = {"Model": "test-model", "GPU": "NVIDIA A100"}
        error = VLLMError("Test error", context=context)

        error_str = str(error)
        assert "Test error" in error_str
        assert "Context:" in error_str
        assert "Model: test-model" in error_str
        assert "GPU: NVIDIA A100" in error_str

    def test_error_with_solutions(self):
        """Test creating an error with solutions."""
        solutions = [
            "Reduce memory usage",
            "Enable quantization",
            "Use tensor parallelism",
        ]
        error = VLLMError("Test error", solutions=solutions)

        error_str = str(error)
        assert "Test error" in error_str
        assert "Solutions:" in error_str
        assert "1. Reduce memory usage" in error_str
        assert "2. Enable quantization" in error_str
        assert "3. Use tensor parallelism" in error_str

    def test_error_with_docs_url(self):
        """Test creating an error with documentation URL."""
        docs_url = "https://docs.vllm.ai/en/latest/troubleshooting.html"
        error = VLLMError("Test error", docs_url=docs_url)

        error_str = str(error)
        assert "Test error" in error_str
        assert f"Documentation: {docs_url}" in error_str

    def test_error_with_all_fields(self):
        """Test creating an error with all fields populated."""
        error = VLLMError(
            "Comprehensive test error",
            context={
                "Model": "test-model",
                "TP size": 2,
            },
            solutions=[
                "Solution 1",
                "Solution 2",
            ],
            docs_url="https://docs.vllm.ai/test",
        )

        error_str = str(error)
        assert "Comprehensive test error" in error_str
        assert "Context:" in error_str
        assert "Model: test-model" in error_str
        assert "TP size: 2" in error_str
        assert "Solutions:" in error_str
        assert "1. Solution 1" in error_str
        assert "2. Solution 2" in error_str
        assert "Documentation: https://docs.vllm.ai/test" in error_str

    def test_error_inheritance(self):
        """Test that VLLMError can be caught as Exception."""
        with pytest.raises(Exception):
            raise VLLMError("Test error")

    def test_error_catchable(self):
        """Test that VLLMError can be specifically caught."""
        with pytest.raises(VLLMError):
            raise VLLMError("Test error")


class TestVLLMMemoryError:
    """Tests for the VLLMMemoryError class."""

    def test_memory_error_creation(self):
        """Test creating a memory error."""
        error = VLLMMemoryError(
            "Out of GPU memory",
            context={"Required": "10GB", "Available": "8GB"},
            solutions=["Reduce batch size", "Enable quantization"],
        )

        error_str = str(error)
        assert "Out of GPU memory" in error_str
        assert "Required: 10GB" in error_str
        assert "Available: 8GB" in error_str

    def test_memory_error_inheritance(self):
        """Test that VLLMMemoryError inherits from VLLMError."""
        error = VLLMMemoryError("Memory error")
        assert isinstance(error, VLLMError)
        assert isinstance(error, Exception)

    def test_memory_error_catchable(self):
        """Test that VLLMMemoryError can be caught specifically."""
        with pytest.raises(VLLMMemoryError):
            raise VLLMMemoryError("Memory error")

        # Also catchable as VLLMError
        with pytest.raises(VLLMError):
            raise VLLMMemoryError("Memory error")


class TestVLLMConfigurationError:
    """Tests for the VLLMConfigurationError class."""

    def test_configuration_error_creation(self):
        """Test creating a configuration error."""
        error = VLLMConfigurationError(
            "Invalid configuration",
            context={"tensor_parallel_size": 8, "available_gpus": 4},
            solutions=["Reduce TP size to 4"],
            docs_url="https://docs.vllm.ai/distributed",
        )

        error_str = str(error)
        assert "Invalid configuration" in error_str
        assert "tensor_parallel_size: 8" in error_str

    def test_configuration_error_inheritance(self):
        """Test that VLLMConfigurationError inherits correctly."""
        error = VLLMConfigurationError("Config error")
        assert isinstance(error, VLLMError)


class TestVLLMModelLoadError:
    """Tests for the VLLMModelLoadError class."""

    def test_model_load_error_creation(self):
        """Test creating a model load error."""
        error = VLLMModelLoadError(
            "Model not found",
            context={"Model": "nonexistent-model"},
            solutions=[
                "Check model name",
                "Verify HuggingFace login",
            ],
        )

        error_str = str(error)
        assert "Model not found" in error_str
        assert "nonexistent-model" in error_str

    def test_model_load_error_inheritance(self):
        """Test that VLLMModelLoadError inherits correctly."""
        error = VLLMModelLoadError("Load error")
        assert isinstance(error, VLLMError)


class TestVLLMDistributedError:
    """Tests for the VLLMDistributedError class."""

    def test_distributed_error_creation(self):
        """Test creating a distributed error."""
        error = VLLMDistributedError(
            "NCCL timeout",
            context={"Operation": "all_reduce"},
            solutions=["Increase NCCL timeout", "Check network"],
        )

        error_str = str(error)
        assert "NCCL timeout" in error_str

    def test_distributed_error_inheritance(self):
        """Test that VLLMDistributedError inherits correctly."""
        error = VLLMDistributedError("Distributed error")
        assert isinstance(error, VLLMError)


class TestVLLMTokenizerError:
    """Tests for the VLLMTokenizerError class."""

    def test_tokenizer_error_creation(self):
        """Test creating a tokenizer error."""
        error = VLLMTokenizerError(
            "Tokenizer not found",
            context={"Tokenizer": "custom-tokenizer"},
            solutions=["Check tokenizer path"],
        )

        error_str = str(error)
        assert "Tokenizer not found" in error_str


class TestVLLMQuantizationError:
    """Tests for the VLLMQuantizationError class."""

    def test_quantization_error_creation(self):
        """Test creating a quantization error."""
        error = VLLMQuantizationError(
            "Quantization failed",
            context={"Method": "fp8"},
            solutions=["Check quantization support"],
        )

        error_str = str(error)
        assert "Quantization failed" in error_str


class TestErrorFormatting:
    """Tests for error message formatting."""

    def test_empty_context_not_shown(self):
        """Test that empty context section is not shown."""
        error = VLLMError("Test error", context={})
        error_str = str(error)
        assert "Context:" not in error_str

    def test_empty_solutions_not_shown(self):
        """Test that empty solutions section is not shown."""
        error = VLLMError("Test error", solutions=[])
        error_str = str(error)
        assert "Solutions:" not in error_str

    def test_none_docs_not_shown(self):
        """Test that None docs_url is not shown."""
        error = VLLMError("Test error", docs_url=None)
        error_str = str(error)
        assert "Documentation:" not in error_str

    def test_multiline_formatting(self):
        """Test that error message is properly formatted with newlines."""
        error = VLLMError(
            "Main error message",
            context={"key1": "value1"},
            solutions=["solution1"],
            docs_url="https://example.com",
        )

        error_str = str(error)
        lines = error_str.split("\n")
        assert len(lines) > 1  # Should have multiple lines
        assert lines[0] == "Main error message"
