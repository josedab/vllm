# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for attention backend capabilities and feature-based selection."""

import warnings

import pytest

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.selector import (
    DEPRECATED_BACKENDS,
    DEPRECATION_SCHEDULE,
    _cached_get_attn_backend,
    check_deprecated_backend,
    get_backend_by_features,
    get_backend_capabilities_summary,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching."""
    _cached_get_attn_backend.cache_clear()


class TestBackendCapabilities:
    """Test cases for backend capabilities system."""

    def test_base_backend_has_capabilities(self):
        """Test that AttentionBackend has CAPABILITIES class variable."""
        assert hasattr(AttentionBackend, "CAPABILITIES")
        assert isinstance(AttentionBackend.CAPABILITIES, dict)
        assert "paged_attention" in AttentionBackend.CAPABILITIES
        assert "prefix_caching" in AttentionBackend.CAPABILITIES
        assert "sliding_window" in AttentionBackend.CAPABILITIES
        assert "speculative_decoding" in AttentionBackend.CAPABILITIES

    def test_base_backend_has_priority(self):
        """Test that AttentionBackend has selection_priority."""
        assert hasattr(AttentionBackend, "selection_priority")
        assert isinstance(AttentionBackend.selection_priority, int)

    def test_supports_features_method(self):
        """Test the supports_features method."""
        # Test with empty set - should always return True
        assert AttentionBackend.supports_features(set()) is True

        # Test with paged_attention - should return True (default)
        assert AttentionBackend.supports_features({"paged_attention"}) is True

    def test_get_capabilities_returns_copy(self):
        """Test that get_capabilities returns a copy."""
        caps1 = AttentionBackend.get_capabilities()
        caps2 = AttentionBackend.get_capabilities()

        # Modify caps1 and ensure caps2 is not affected
        caps1["test_key"] = True
        assert "test_key" not in caps2

    def test_capability_check_methods(self):
        """Test individual capability check methods."""
        # These test the base class defaults
        assert AttentionBackend.supports_paged_attention() is True
        assert AttentionBackend.supports_prefix_caching() is False
        assert AttentionBackend.supports_sliding_window() is False
        assert AttentionBackend.supports_speculative_decoding() is False
        assert AttentionBackend.supports_chunked_prefill() is False
        assert AttentionBackend.supports_multi_step_decoding() is False

    def test_get_selection_priority(self):
        """Test get_selection_priority method."""
        priority = AttentionBackend.get_selection_priority()
        assert isinstance(priority, int)
        assert priority == AttentionBackend.selection_priority


class TestCustomBackendCapabilities:
    """Test that custom backends can override capabilities."""

    def test_custom_backend_capabilities(self):
        """Test creating a custom backend with different capabilities."""

        class CustomBackend(AttentionBackend):
            CAPABILITIES = {
                "paged_attention": True,
                "prefix_caching": True,
                "sliding_window": True,
                "speculative_decoding": True,
                "chunked_prefill": True,
                "multi_step_decoding": True,
            }
            selection_priority = 100

            @staticmethod
            def get_name():
                return "CUSTOM"

            @staticmethod
            def get_impl_cls():
                return None

            @staticmethod
            def get_builder_cls():
                return None

            @staticmethod
            def get_kv_cache_shape(*args, **kwargs):
                return (1, 1, 1, 1, 1)

        # Test that custom capabilities are properly inherited
        assert CustomBackend.supports_paged_attention() is True
        assert CustomBackend.supports_prefix_caching() is True
        assert CustomBackend.supports_sliding_window() is True
        assert CustomBackend.supports_speculative_decoding() is True
        assert CustomBackend.get_selection_priority() == 100

    def test_feature_support_check(self):
        """Test supports_features with multiple features."""

        class FullFeaturedBackend(AttentionBackend):
            CAPABILITIES = {
                "paged_attention": True,
                "prefix_caching": True,
                "sliding_window": True,
                "speculative_decoding": True,
                "chunked_prefill": True,
                "multi_step_decoding": True,
            }

            @staticmethod
            def get_name():
                return "FULL"

            @staticmethod
            def get_impl_cls():
                return None

            @staticmethod
            def get_builder_cls():
                return None

            @staticmethod
            def get_kv_cache_shape(*args, **kwargs):
                return (1, 1, 1, 1, 1)

        class LimitedBackend(AttentionBackend):
            CAPABILITIES = {
                "paged_attention": True,
                "prefix_caching": False,
                "sliding_window": False,
                "speculative_decoding": False,
                "chunked_prefill": False,
                "multi_step_decoding": False,
            }

            @staticmethod
            def get_name():
                return "LIMITED"

            @staticmethod
            def get_impl_cls():
                return None

            @staticmethod
            def get_builder_cls():
                return None

            @staticmethod
            def get_kv_cache_shape(*args, **kwargs):
                return (1, 1, 1, 1, 1)

        # Full backend supports all features
        assert FullFeaturedBackend.supports_features(
            {"paged_attention", "prefix_caching", "sliding_window"}
        ) is True

        # Limited backend doesn't support prefix_caching
        assert LimitedBackend.supports_features(
            {"paged_attention", "prefix_caching"}
        ) is False

        # Limited backend supports just paged_attention
        assert LimitedBackend.supports_features({"paged_attention"}) is True


class TestFeatureBasedSelection:
    """Test feature-based backend selection."""

    def test_get_backend_by_features_empty(self):
        """Test selection with empty features returns highest priority."""
        # This may return None if no backends are importable
        # Just ensure it doesn't raise an exception
        try:
            result = get_backend_by_features(set())
            # If we get a result, it should be a backend class
            if result is not None:
                assert hasattr(result, "get_name")
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Required backends not available")

    def test_get_backend_by_features_with_custom_backends(self):
        """Test selection with custom backend list."""

        class HighPriorityBackend(AttentionBackend):
            CAPABILITIES = {
                "paged_attention": True,
                "prefix_caching": True,
                "sliding_window": False,
                "speculative_decoding": False,
                "chunked_prefill": False,
                "multi_step_decoding": False,
            }
            selection_priority = 100

            @staticmethod
            def get_name():
                return "HIGH"

            @staticmethod
            def get_impl_cls():
                return None

            @staticmethod
            def get_builder_cls():
                return None

            @staticmethod
            def get_kv_cache_shape(*args, **kwargs):
                return (1, 1, 1, 1, 1)

        class LowPriorityBackend(AttentionBackend):
            CAPABILITIES = {
                "paged_attention": True,
                "prefix_caching": True,
                "sliding_window": True,
                "speculative_decoding": False,
                "chunked_prefill": False,
                "multi_step_decoding": False,
            }
            selection_priority = 10

            @staticmethod
            def get_name():
                return "LOW"

            @staticmethod
            def get_impl_cls():
                return None

            @staticmethod
            def get_builder_cls():
                return None

            @staticmethod
            def get_kv_cache_shape(*args, **kwargs):
                return (1, 1, 1, 1, 1)

        backends = [HighPriorityBackend, LowPriorityBackend]

        # Both support prefix_caching, so high priority should be selected
        result = get_backend_by_features({"prefix_caching"}, backends)
        assert result is HighPriorityBackend

        # Only low priority supports sliding_window
        result = get_backend_by_features({"sliding_window"}, backends)
        assert result is LowPriorityBackend

        # Neither supports speculative_decoding
        result = get_backend_by_features({"speculative_decoding"}, backends)
        assert result is None

    def test_selection_respects_priority(self):
        """Test that selection respects priority ordering."""

        class P100Backend(AttentionBackend):
            CAPABILITIES = {"paged_attention": True, "prefix_caching": False,
                          "sliding_window": False, "speculative_decoding": False,
                          "chunked_prefill": False, "multi_step_decoding": False}
            selection_priority = 100

            @staticmethod
            def get_name():
                return "P100"

            @staticmethod
            def get_impl_cls():
                return None

            @staticmethod
            def get_builder_cls():
                return None

            @staticmethod
            def get_kv_cache_shape(*args, **kwargs):
                return (1, 1, 1, 1, 1)

        class P50Backend(AttentionBackend):
            CAPABILITIES = {"paged_attention": True, "prefix_caching": False,
                          "sliding_window": False, "speculative_decoding": False,
                          "chunked_prefill": False, "multi_step_decoding": False}
            selection_priority = 50

            @staticmethod
            def get_name():
                return "P50"

            @staticmethod
            def get_impl_cls():
                return None

            @staticmethod
            def get_builder_cls():
                return None

            @staticmethod
            def get_kv_cache_shape(*args, **kwargs):
                return (1, 1, 1, 1, 1)

        class P10Backend(AttentionBackend):
            CAPABILITIES = {"paged_attention": True, "prefix_caching": False,
                          "sliding_window": False, "speculative_decoding": False,
                          "chunked_prefill": False, "multi_step_decoding": False}
            selection_priority = 10

            @staticmethod
            def get_name():
                return "P10"

            @staticmethod
            def get_impl_cls():
                return None

            @staticmethod
            def get_builder_cls():
                return None

            @staticmethod
            def get_kv_cache_shape(*args, **kwargs):
                return (1, 1, 1, 1, 1)

        # Pass backends in random order
        backends = [P10Backend, P100Backend, P50Backend]
        result = get_backend_by_features({"paged_attention"}, backends)
        assert result is P100Backend


class TestDeprecationWarnings:
    """Test deprecation warning system."""

    def test_check_deprecated_backend_not_deprecated(self):
        """Test that non-deprecated backends return None."""
        result = check_deprecated_backend("FLASH_ATTN")
        assert result is None

    def test_check_deprecated_backend_with_deprecation(self, monkeypatch):
        """Test that deprecated backends return replacement and warn."""
        # Temporarily add a deprecated backend
        test_deprecated = {"TEST_DEPRECATED": "FLASH_ATTN"}
        test_schedule = {
            "TEST_DEPRECATED": {
                "deprecated_in": "v0.6.0",
                "removed_in": "v0.7.0"
            }
        }

        monkeypatch.setattr(
            "vllm.attention.selector.DEPRECATED_BACKENDS",
            test_deprecated
        )
        monkeypatch.setattr(
            "vllm.attention.selector.DEPRECATION_SCHEDULE",
            test_schedule
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_deprecated_backend("TEST_DEPRECATED")

            assert result == "FLASH_ATTN"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "TEST_DEPRECATED" in str(w[0].message)
            assert "v0.6.0" in str(w[0].message)
            assert "v0.7.0" in str(w[0].message)

    def test_deprecated_backends_dict_structure(self):
        """Test that DEPRECATED_BACKENDS has correct structure."""
        assert isinstance(DEPRECATED_BACKENDS, dict)
        for key, value in DEPRECATED_BACKENDS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_deprecation_schedule_dict_structure(self):
        """Test that DEPRECATION_SCHEDULE has correct structure."""
        assert isinstance(DEPRECATION_SCHEDULE, dict)
        for key, value in DEPRECATION_SCHEDULE.items():
            assert isinstance(key, str)
            assert isinstance(value, dict)


class TestCapabilitiesSummary:
    """Test the capabilities summary function."""

    def test_get_backend_capabilities_summary(self):
        """Test getting capabilities summary."""
        # This may raise import errors for missing backends
        # Just test that it returns a dict structure
        try:
            summary = get_backend_capabilities_summary()
            assert isinstance(summary, dict)

            for backend_name, info in summary.items():
                assert isinstance(backend_name, str)
                assert isinstance(info, dict)
                assert "capabilities" in info
                assert "priority" in info
                assert isinstance(info["capabilities"], dict)
                assert isinstance(info["priority"], int)
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Required backends not available")


class TestRealBackendCapabilities:
    """Test capabilities of real backend implementations."""

    def test_flash_attention_capabilities(self):
        """Test FlashAttention backend has correct capabilities."""
        try:
            from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

            # FlashAttention should have high priority and full features
            assert FlashAttentionBackend.selection_priority == 100
            assert FlashAttentionBackend.supports_paged_attention() is True
            assert FlashAttentionBackend.supports_prefix_caching() is True
            assert FlashAttentionBackend.supports_sliding_window() is True
        except ImportError:
            pytest.skip("FlashAttention backend not available")

    def test_flashinfer_capabilities(self):
        """Test FlashInfer backend has correct capabilities."""
        try:
            from vllm.v1.attention.backends.flashinfer import FlashInferBackend

            # FlashInfer should have high priority
            assert FlashInferBackend.selection_priority == 100
            assert FlashInferBackend.supports_paged_attention() is True
            assert FlashInferBackend.supports_prefix_caching() is True
        except ImportError:
            pytest.skip("FlashInfer backend not available")

    def test_triton_capabilities(self):
        """Test Triton backend has correct capabilities."""
        try:
            from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

            # Triton should have P1 priority
            assert TritonAttentionBackend.selection_priority == 50
            assert TritonAttentionBackend.supports_paged_attention() is True
        except ImportError:
            pytest.skip("Triton backend not available")

    def test_xformers_capabilities(self):
        """Test XFormers backend has correct capabilities."""
        try:
            from vllm.v1.attention.backends.xformers import XFormersAttentionBackend

            # XFormers should have P1 priority
            assert XFormersAttentionBackend.selection_priority == 50
            assert XFormersAttentionBackend.supports_paged_attention() is True
        except ImportError:
            pytest.skip("XFormers backend not available")

    def test_cpu_capabilities(self):
        """Test CPU backend has correct capabilities."""
        try:
            from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

            # CPU should have low priority
            assert CPUAttentionBackend.selection_priority == 10
            assert CPUAttentionBackend.supports_paged_attention() is True
        except ImportError:
            pytest.skip("CPU backend not available")

    def test_pallas_capabilities(self):
        """Test Pallas backend has correct capabilities."""
        try:
            from vllm.v1.attention.backends.pallas import PallasAttentionBackend

            # Pallas should have P0 priority for TPU
            assert PallasAttentionBackend.selection_priority == 100
            assert PallasAttentionBackend.supports_paged_attention() is True
        except ImportError:
            pytest.skip("Pallas backend not available")
