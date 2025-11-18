# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for V0 engine deprecation system."""

import os
import warnings
from unittest import mock

import pytest


class TestV0DeprecationWarnings:
    """Test V0 deprecation warning functions."""

    def test_warn_v0_feature_usage(self):
        """Test that V0 feature usage warnings are emitted."""
        from vllm.engine import warn_v0_feature_usage

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_v0_feature_usage("test_feature")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "test_feature" in str(w[0].message)
            assert "V0-only" in str(w[0].message)

    def test_warn_v0_parameter_usage(self):
        """Test that V0 parameter usage warnings are emitted."""
        from vllm.engine import warn_v0_parameter_usage

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_v0_parameter_usage("test_param", alternative="new_param")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "test_param" in str(w[0].message)
            assert "new_param" in str(w[0].message)

    def test_warn_v0_parameter_usage_without_alternative(self):
        """Test parameter warning without alternative suggestion."""
        from vllm.engine import warn_v0_parameter_usage

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_v0_parameter_usage("old_param")

            assert len(w) == 1
            assert "old_param" in str(w[0].message)
            assert "instead" not in str(w[0].message)


class TestCreateEngine:
    """Test create_engine function."""

    def test_create_engine_v0_raises_error(self):
        """Test that requesting V0 engine raises RuntimeError."""
        from vllm.engine import create_engine
        from vllm.engine.arg_utils import EngineArgs

        # Mock EngineArgs since we don't need real engine creation
        with pytest.raises(RuntimeError) as exc_info:
            create_engine(
                engine_args=mock.MagicMock(spec=EngineArgs),
                use_v1=False
            )

        assert "V0 engine has been removed" in str(exc_info.value)

    def test_create_engine_env_v0_raises_error(self):
        """Test that VLLM_USE_V1=0 raises RuntimeError."""
        from vllm.engine import create_engine
        from vllm.engine.arg_utils import EngineArgs

        with mock.patch.dict(os.environ, {"VLLM_USE_V1": "0"}):
            with pytest.raises(RuntimeError) as exc_info:
                create_engine(
                    engine_args=mock.MagicMock(spec=EngineArgs),
                )

            assert "V0 engine has been removed" in str(exc_info.value)


class TestFeatureParityCheck:
    """Test feature parity check functions."""

    def test_check_v1_feature_support_known_features(self):
        """Test checking known V1 feature support."""
        from vllm.engine import check_v1_feature_support

        # These should be supported
        assert check_v1_feature_support("continuous_batching") is True
        assert check_v1_feature_support("prefix_caching") is True
        assert check_v1_feature_support("lora") is True

        # This should not be supported
        assert check_v1_feature_support("best_of") is False

    def test_check_v1_feature_support_unknown_feature(self):
        """Test checking unknown feature returns False."""
        from vllm.engine import check_v1_feature_support

        assert check_v1_feature_support("unknown_feature") is False

    def test_get_unsupported_v1_features(self):
        """Test getting list of unsupported features."""
        from vllm.engine import get_unsupported_v1_features

        unsupported = get_unsupported_v1_features()
        assert isinstance(unsupported, list)
        assert "best_of" in unsupported


class TestConstants:
    """Test deprecation constants."""

    def test_deprecation_version(self):
        """Test deprecation version is set."""
        from vllm.engine import V0_DEPRECATION_VERSION

        assert V0_DEPRECATION_VERSION == "v1.0.0"

    def test_migration_guide_url(self):
        """Test migration guide URL is set."""
        from vllm.engine import V0_MIGRATION_GUIDE_URL

        assert "v0-to-v1" in V0_MIGRATION_GUIDE_URL

    def test_removal_url(self):
        """Test removal URL is set."""
        from vllm.engine import V0_REMOVAL_URL

        assert "v0-removal" in V0_REMOVAL_URL


class TestCompatibilityLayer:
    """Test V0 compatibility layer."""

    def test_scheduler_output_v0_format(self):
        """Test converting V1 output to V0 format."""
        from vllm.compat.v0 import (
            SchedulerOutputV0,
            get_scheduler_output_v0_format,
        )

        # Mock V1 output
        v1_output = mock.MagicMock()
        v1_output.outputs = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v0_output = get_scheduler_output_v0_format(v1_output)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        assert isinstance(v0_output, SchedulerOutputV0)

    def test_sequence_group_v0_format(self):
        """Test creating V0 format sequence group."""
        from vllm.compat.v0 import SequenceGroupV0, get_sequence_group_v0_format

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            seq_group = get_sequence_group_v0_format(
                request_id="test-123",
                sampling_params=mock.MagicMock(),
                arrival_time=1.0,
            )

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        assert isinstance(seq_group, SequenceGroupV0)
        assert seq_group.request_id == "test-123"
        assert seq_group.arrival_time == 1.0

    def test_v0_engine_wrapper_init(self):
        """Test V0 engine wrapper initialization."""
        from vllm.compat.v0 import V0EngineWrapper

        mock_engine = mock.MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapper = V0EngineWrapper(mock_engine)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        assert wrapper._v1_engine is mock_engine

    def test_v0_engine_wrapper_getattr(self):
        """Test V0 engine wrapper forwards attribute access."""
        from vllm.compat.v0 import V0EngineWrapper

        mock_engine = mock.MagicMock()
        mock_engine.test_attr = "test_value"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            wrapper = V0EngineWrapper(mock_engine)

        # Attribute access should be forwarded
        assert wrapper.test_attr == "test_value"

    def test_wrap_v1_engine_with_v0_interface(self):
        """Test wrapping V1 engine with V0 interface."""
        from vllm.compat.v0 import V0EngineWrapper, wrap_v1_engine_with_v0_interface

        mock_engine = mock.MagicMock()

        wrapped = wrap_v1_engine_with_v0_interface(mock_engine)

        assert isinstance(wrapped, V0EngineWrapper)

    def test_v0_compatibility_error(self):
        """Test V0CompatibilityError can be raised."""
        from vllm.compat.v0 import V0CompatibilityError

        with pytest.raises(V0CompatibilityError) as exc_info:
            raise V0CompatibilityError("Test error")

        assert "Test error" in str(exc_info.value)


class TestParityScript:
    """Test the v0_v1_parity.py script functions."""

    def test_check_parity_returns_report(self):
        """Test that check_parity returns a valid report."""
        import sys
        sys.path.insert(0, "/home/user/vllm/scripts")

        from v0_v1_parity import check_parity

        report = check_parity()

        assert report.total_features > 0
        assert report.full_support >= 0
        assert report.partial_support >= 0
        assert report.unsupported >= 0
        assert report.deprecated >= 0
        assert 0 <= report.parity_percentage <= 100

    def test_get_missing_features(self):
        """Test getting missing features list."""
        import sys
        sys.path.insert(0, "/home/user/vllm/scripts")

        from v0_v1_parity import get_missing_features

        missing = get_missing_features()

        assert isinstance(missing, list)
        # best_of should be in the missing list
        assert "best_of" in missing

    def test_get_migration_blockers(self):
        """Test getting migration blockers."""
        import sys
        sys.path.insert(0, "/home/user/vllm/scripts")

        from v0_v1_parity import Feature, get_migration_blockers

        blockers = get_migration_blockers()

        assert isinstance(blockers, list)
        # All blockers should be Feature objects
        for blocker in blockers:
            assert isinstance(blocker, Feature)


class TestSchedulerOutputV0:
    """Test SchedulerOutputV0 dataclass."""

    def test_scheduler_output_defaults(self):
        """Test SchedulerOutputV0 has correct defaults."""
        from vllm.compat.v0 import SchedulerOutputV0

        output = SchedulerOutputV0()

        assert output.scheduled_seq_groups == []
        assert output.blocks_to_swap_in == {}
        assert output.blocks_to_swap_out == {}
        assert output.blocks_to_copy == []
        assert output.num_lookahead_slots == 0
        assert output.running_queue_size == 0
        assert output.preempted == 0

    def test_scheduler_output_with_values(self):
        """Test SchedulerOutputV0 with custom values."""
        from vllm.compat.v0 import SchedulerOutputV0

        output = SchedulerOutputV0(
            scheduled_seq_groups=["group1"],
            blocks_to_swap_in={1: 2},
            num_lookahead_slots=5,
        )

        assert output.scheduled_seq_groups == ["group1"]
        assert output.blocks_to_swap_in == {1: 2}
        assert output.num_lookahead_slots == 5


class TestSequenceGroupV0:
    """Test SequenceGroupV0 dataclass."""

    def test_sequence_group_defaults(self):
        """Test SequenceGroupV0 has correct defaults."""
        from vllm.compat.v0 import SequenceGroupV0

        seq_group = SequenceGroupV0(request_id="test")

        assert seq_group.request_id == "test"
        assert seq_group.seqs == []
        assert seq_group.sampling_params is None
        assert seq_group.arrival_time == 0.0
        assert seq_group.lora_request is None
        assert seq_group.prefix_indices is None

    def test_sequence_group_with_values(self):
        """Test SequenceGroupV0 with custom values."""
        from vllm.compat.v0 import SequenceGroupV0

        seq_group = SequenceGroupV0(
            request_id="test-123",
            arrival_time=10.5,
            prefix_indices=[1, 2, 3],
        )

        assert seq_group.request_id == "test-123"
        assert seq_group.arrival_time == 10.5
        assert seq_group.prefix_indices == [1, 2, 3]
