# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for configuration profiles and validation."""

import os
from unittest.mock import patch

import pytest

from vllm.config import (
    ConfigProfile,
    ConfigurationError,
    ConfigurationWarning,
    apply_profile_settings,
    check_deprecated_env_vars,
    get_profile_description,
    get_profile_settings,
    list_profiles,
    validate_config,
    validate_profile_overrides,
)
from vllm.config.profiles import PROFILE_SETTINGS
from vllm.engine.arg_utils import EngineArgs


class TestConfigProfile:
    """Tests for ConfigProfile enum."""

    def test_profile_enum_values(self):
        """Test that all expected profiles exist."""
        assert ConfigProfile.HIGH_THROUGHPUT.value == "high_throughput"
        assert ConfigProfile.LOW_LATENCY.value == "low_latency"
        assert ConfigProfile.MEMORY_CONSTRAINED.value == "memory_constrained"
        assert ConfigProfile.DEVELOPMENT.value == "development"

    def test_profile_from_string(self):
        """Test creating profile from string value."""
        profile = ConfigProfile("high_throughput")
        assert profile == ConfigProfile.HIGH_THROUGHPUT

    def test_invalid_profile_string(self):
        """Test that invalid profile strings raise ValueError."""
        with pytest.raises(ValueError):
            ConfigProfile("invalid_profile")


class TestGetProfileSettings:
    """Tests for get_profile_settings function."""

    def test_get_high_throughput_settings(self):
        """Test getting HIGH_THROUGHPUT profile settings."""
        settings = get_profile_settings(ConfigProfile.HIGH_THROUGHPUT)
        assert settings["max_num_seqs"] == 256
        assert settings["gpu_memory_utilization"] == 0.95
        assert settings["enable_prefix_caching"] is True
        assert settings["enable_chunked_prefill"] is True

    def test_get_low_latency_settings(self):
        """Test getting LOW_LATENCY profile settings."""
        settings = get_profile_settings(ConfigProfile.LOW_LATENCY)
        assert settings["max_num_seqs"] == 32
        assert settings["enable_prefix_caching"] is True
        assert settings["enable_chunked_prefill"] is True

    def test_get_memory_constrained_settings(self):
        """Test getting MEMORY_CONSTRAINED profile settings."""
        settings = get_profile_settings(ConfigProfile.MEMORY_CONSTRAINED)
        assert settings["max_num_seqs"] == 64
        assert settings["gpu_memory_utilization"] == 0.8
        assert settings["swap_space"] == 8

    def test_get_development_settings(self):
        """Test getting DEVELOPMENT profile settings."""
        settings = get_profile_settings(ConfigProfile.DEVELOPMENT)
        assert settings["enforce_eager"] is True
        assert settings["enable_chunked_prefill"] is False

    def test_get_settings_from_string(self):
        """Test getting settings using string profile name."""
        settings = get_profile_settings("high_throughput")
        assert settings["max_num_seqs"] == 256

    def test_get_settings_none_profile(self):
        """Test that None profile returns empty dict."""
        settings = get_profile_settings(None)
        assert settings == {}

    def test_invalid_profile_raises_error(self):
        """Test that invalid profile raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_profile_settings("invalid")
        assert "Unknown profile" in str(exc_info.value)


class TestApplyProfileSettings:
    """Tests for apply_profile_settings function."""

    def test_apply_without_overrides(self):
        """Test applying profile without user overrides."""
        settings = apply_profile_settings(
            ConfigProfile.HIGH_THROUGHPUT,
            {}
        )
        assert settings["max_num_seqs"] == 256
        assert settings["gpu_memory_utilization"] == 0.95

    def test_apply_with_overrides(self):
        """Test that user overrides take precedence."""
        settings = apply_profile_settings(
            ConfigProfile.HIGH_THROUGHPUT,
            {"max_num_seqs": 128, "max_model_len": 8192}
        )
        # Override should take effect
        assert settings["max_num_seqs"] == 128
        # New setting should be added
        assert settings["max_model_len"] == 8192
        # Profile setting should remain
        assert settings["gpu_memory_utilization"] == 0.95

    def test_apply_with_none_values(self):
        """Test that None values in overrides are ignored."""
        settings = apply_profile_settings(
            ConfigProfile.HIGH_THROUGHPUT,
            {"max_num_seqs": None}
        )
        # None values should not override
        assert settings["max_num_seqs"] == 256


class TestListProfiles:
    """Tests for list_profiles function."""

    def test_list_all_profiles(self):
        """Test listing all available profiles."""
        profiles = list_profiles()
        assert len(profiles) == 4

        names = [p["name"] for p in profiles]
        assert "high_throughput" in names
        assert "low_latency" in names
        assert "memory_constrained" in names
        assert "development" in names

    def test_profile_info_structure(self):
        """Test that profile info has expected structure."""
        profiles = list_profiles()
        for profile in profiles:
            assert "name" in profile
            assert "enum" in profile
            assert "description" in profile
            assert "settings" in profile
            assert isinstance(profile["settings"], dict)


class TestGetProfileDescription:
    """Tests for get_profile_description function."""

    def test_get_description(self):
        """Test getting profile descriptions."""
        desc = get_profile_description(ConfigProfile.HIGH_THROUGHPUT)
        assert "throughput" in desc.lower()
        assert len(desc) > 50  # Should be a meaningful description


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_high_memory_utilization_warning(self):
        """Test warning for high GPU memory utilization."""
        warnings = validate_config(gpu_memory_utilization=0.99)
        assert len(warnings) >= 1
        assert any("very high" in str(w) for w in warnings)

    def test_low_memory_utilization_warning(self):
        """Test warning for low GPU memory utilization."""
        warnings = validate_config(gpu_memory_utilization=0.3)
        assert len(warnings) >= 1
        assert any("low" in str(w).lower() for w in warnings)

    def test_normal_memory_utilization_no_warning(self):
        """Test no warning for normal GPU memory utilization."""
        warnings = validate_config(gpu_memory_utilization=0.9)
        memory_warnings = [w for w in warnings if "memory utilization" in str(w)]
        assert len(memory_warnings) == 0

    def test_prefix_caching_disabled_warning(self):
        """Test warning when prefix caching is disabled."""
        warnings = validate_config(enable_prefix_caching=False)
        assert any("prefix caching" in str(w).lower() for w in warnings)

    def test_small_batch_size_warning(self):
        """Test warning for very small max_num_seqs."""
        warnings = validate_config(max_num_seqs=4)
        assert any("small" in str(w).lower() for w in warnings)

    def test_large_batch_size_warning(self):
        """Test warning for very large max_num_seqs."""
        warnings = validate_config(max_num_seqs=1000)
        assert any("large" in str(w).lower() for w in warnings)

    def test_enforce_eager_warning(self):
        """Test warning when enforce_eager is used in non-development context."""
        warnings = validate_config(enforce_eager=True)
        assert any("eager" in str(w).lower() for w in warnings)

    def test_no_eager_warning_in_development(self):
        """Test no eager warning when using development profile."""
        warnings = validate_config(enforce_eager=True, profile="development")
        eager_warnings = [w for w in warnings if "eager" in str(w).lower()]
        assert len(eager_warnings) == 0


class TestValidateProfileOverrides:
    """Tests for validate_profile_overrides function."""

    def test_conflicting_throughput_override(self):
        """Test warning for low batch size with HIGH_THROUGHPUT profile."""
        warnings = validate_profile_overrides(
            "high_throughput",
            {"max_num_seqs": 8}
        )
        assert len(warnings) >= 1
        assert any("max_num_seqs" in str(w) for w in warnings)

    def test_conflicting_latency_override(self):
        """Test warning for high batch size with LOW_LATENCY profile."""
        warnings = validate_profile_overrides(
            "low_latency",
            {"max_num_seqs": 512}
        )
        assert len(warnings) >= 1

    def test_conflicting_memory_override(self):
        """Test warning for high memory with MEMORY_CONSTRAINED profile."""
        warnings = validate_profile_overrides(
            "memory_constrained",
            {"gpu_memory_utilization": 0.95}
        )
        assert len(warnings) >= 1

    def test_no_warnings_for_compatible_overrides(self):
        """Test no warnings for compatible overrides."""
        warnings = validate_profile_overrides(
            "high_throughput",
            {"max_num_seqs": 512}  # Compatible with throughput
        )
        assert len(warnings) == 0


class TestCheckDeprecatedEnvVars:
    """Tests for check_deprecated_env_vars function."""

    def test_no_deprecated_vars(self):
        """Test no warnings when no deprecated vars are set."""
        # Clear any deprecated vars
        deprecated_vars = ["VLLM_USE_V1", "VLLM_NCCL_SO_PATH"]
        with patch.dict(os.environ, {}, clear=False):
            for var in deprecated_vars:
                os.environ.pop(var, None)
            warnings = check_deprecated_env_vars()
            # Filter to only warnings about these specific vars
            relevant_warnings = [
                w for w in warnings
                if any(v in w for v in deprecated_vars)
            ]
            assert len(relevant_warnings) == 0

    def test_deprecated_v1_var(self):
        """Test warning for VLLM_USE_V1."""
        with patch.dict(os.environ, {"VLLM_USE_V1": "1"}):
            warnings = check_deprecated_env_vars()
            assert any("VLLM_USE_V1" in w for w in warnings)


class TestEngineArgsProfile:
    """Tests for EngineArgs profile integration."""

    def test_engine_args_with_profile(self):
        """Test that EngineArgs accepts profile parameter."""
        args = EngineArgs(
            model="test-model",
            profile=ConfigProfile.HIGH_THROUGHPUT
        )
        assert args.profile == ConfigProfile.HIGH_THROUGHPUT

    def test_engine_args_profile_applies_settings(self):
        """Test that profile settings are applied in EngineArgs."""
        args = EngineArgs(
            model="test-model",
            profile=ConfigProfile.HIGH_THROUGHPUT
        )
        # Profile settings should be applied
        assert args.max_num_seqs == 256
        assert args.gpu_memory_utilization == 0.95
        assert args.enable_prefix_caching is True

    def test_engine_args_profile_with_override(self):
        """Test that user overrides take precedence over profile."""
        args = EngineArgs(
            model="test-model",
            profile=ConfigProfile.HIGH_THROUGHPUT,
            max_num_seqs=128  # Override
        )
        # Override should take effect
        assert args.max_num_seqs == 128
        # Other profile settings should still apply
        assert args.gpu_memory_utilization == 0.95

    def test_engine_args_profile_from_string(self):
        """Test that profile can be specified as string."""
        args = EngineArgs(
            model="test-model",
            profile="low_latency"
        )
        # LOW_LATENCY profile settings
        assert args.max_num_seqs == 32

    def test_engine_args_no_profile(self):
        """Test that no profile uses default values."""
        args = EngineArgs(model="test-model")
        # Should use default values, not profile values
        assert args.max_num_seqs is None  # Default is None
        assert args.gpu_memory_utilization == 0.9  # Default


class TestConfigurationWarning:
    """Tests for ConfigurationWarning class."""

    def test_warning_with_suggestion(self):
        """Test warning string includes suggestion."""
        warning = ConfigurationWarning(
            "This is a warning",
            "This is a suggestion"
        )
        warning_str = str(warning)
        assert "This is a warning" in warning_str
        assert "Suggestion:" in warning_str
        assert "This is a suggestion" in warning_str

    def test_warning_without_suggestion(self):
        """Test warning string without suggestion."""
        warning = ConfigurationWarning("This is a warning")
        warning_str = str(warning)
        assert "This is a warning" in warning_str
        assert "Suggestion:" not in warning_str


class TestProfileSettingsCompleteness:
    """Tests to ensure profile settings are complete and valid."""

    def test_all_profiles_have_settings(self):
        """Test that all profiles have settings defined."""
        for profile in ConfigProfile:
            assert profile in PROFILE_SETTINGS
            assert isinstance(PROFILE_SETTINGS[profile], dict)
            assert len(PROFILE_SETTINGS[profile]) > 0

    def test_profile_settings_are_valid_types(self):
        """Test that profile settings have valid types."""
        valid_keys = {
            "max_num_seqs",
            "gpu_memory_utilization",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "swap_space",
            "cpu_offload_gb",
            "enforce_eager",
            "_compilation_level",
        }
        for profile, settings in PROFILE_SETTINGS.items():
            for key in settings:
                # Allow internal keys starting with _
                if not key.startswith("_"):
                    assert key in valid_keys, (
                        f"Unknown key '{key}' in profile {profile}"
                    )
