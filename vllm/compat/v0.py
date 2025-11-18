# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility layer for V0 engine internals.

This module provides compatibility shims for users who depend on V0 engine
internals. These shims allow gradual migration to V1 by converting V1
outputs to V0 formats.

Warning:
    This module is provided for backwards compatibility only. Users should
    migrate to V1 APIs directly. This compatibility layer will be removed
    in a future version.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

from vllm.engine import V0_DEPRECATION_VERSION, V0_MIGRATION_GUIDE_URL
from vllm.logger import init_logger

logger = init_logger(__name__)


class V0CompatibilityError(Exception):
    """Error raised when V0 compatibility cannot be maintained."""
    pass


@dataclass
class SchedulerOutputV0:
    """V0-format scheduler output for compatibility.

    This class mimics the V0 scheduler output format for users who
    depend on the V0 internal API structure.
    """

    scheduled_seq_groups: list[Any] = field(default_factory=list)
    """Sequence groups scheduled for this iteration."""

    blocks_to_swap_in: dict[int, int] = field(default_factory=dict)
    """Mapping of GPU block numbers to CPU block numbers to swap in."""

    blocks_to_swap_out: dict[int, int] = field(default_factory=dict)
    """Mapping of GPU block numbers to CPU block numbers to swap out."""

    blocks_to_copy: list[tuple[int, int]] = field(default_factory=list)
    """List of (src, dst) block numbers to copy."""

    num_lookahead_slots: int = 0
    """Number of lookahead slots for speculative decoding."""

    running_queue_size: int = 0
    """Number of requests in the running queue."""

    preempted: int = 0
    """Number of preempted requests."""


@dataclass
class SequenceGroupV0:
    """V0-format sequence group for compatibility.

    This class mimics the V0 sequence group format.
    """

    request_id: str
    """Unique request identifier."""

    seqs: list[Any] = field(default_factory=list)
    """Sequences in this group."""

    sampling_params: Any = None
    """Sampling parameters for this group."""

    arrival_time: float = 0.0
    """Request arrival timestamp."""

    lora_request: Any = None
    """LoRA request if applicable."""

    prefix_indices: list[int] | None = None
    """Prefix cache indices."""


def get_scheduler_output_v0_format(v1_output: Any) -> SchedulerOutputV0:
    """Convert V1 scheduler output to V0 format.

    This compatibility shim allows users who depend on the V0 scheduler
    output format to continue using their code while migrating to V1.

    Args:
        v1_output: V1 scheduler output (EngineCoreOutput or similar).

    Returns:
        SchedulerOutputV0: V0-format scheduler output.

    Warning:
        This function is deprecated and will be removed in {version}.
        Please migrate to V1 APIs directly.
    """.format(version=V0_DEPRECATION_VERSION)

    warnings.warn(
        "get_scheduler_output_v0_format is deprecated and will be removed "
        f"in {V0_DEPRECATION_VERSION}. Please migrate to V1 APIs. "
        f"See {V0_MIGRATION_GUIDE_URL}",
        DeprecationWarning,
        stacklevel=2
    )

    # Convert V1 output to V0 format
    # This is a best-effort conversion as V1 has different internal structure
    scheduled_seq_groups = []
    blocks_to_swap_in: dict[int, int] = {}
    blocks_to_swap_out: dict[int, int] = {}
    blocks_to_copy: list[tuple[int, int]] = []

    # Extract information from V1 output if available
    if hasattr(v1_output, 'outputs'):
        for output in v1_output.outputs:
            seq_group = SequenceGroupV0(
                request_id=getattr(output, 'request_id', ''),
            )
            scheduled_seq_groups.append(seq_group)

    return SchedulerOutputV0(
        scheduled_seq_groups=scheduled_seq_groups,
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=blocks_to_copy,
    )


def get_sequence_group_v0_format(
    request_id: str,
    sampling_params: Any,
    arrival_time: float = 0.0,
    lora_request: Any = None,
) -> SequenceGroupV0:
    """Create a V0-format sequence group.

    Args:
        request_id: Unique request identifier.
        sampling_params: Sampling parameters.
        arrival_time: Request arrival timestamp.
        lora_request: LoRA request if applicable.

    Returns:
        SequenceGroupV0: V0-format sequence group.
    """
    warnings.warn(
        "get_sequence_group_v0_format is deprecated and will be removed "
        f"in {V0_DEPRECATION_VERSION}. Please migrate to V1 APIs. "
        f"See {V0_MIGRATION_GUIDE_URL}",
        DeprecationWarning,
        stacklevel=2
    )

    return SequenceGroupV0(
        request_id=request_id,
        sampling_params=sampling_params,
        arrival_time=arrival_time,
        lora_request=lora_request,
    )


class V0EngineWrapper:
    """Wrapper to provide V0-like interface around V1 engine.

    This wrapper allows users who depend on V0 engine methods and
    attributes to continue using their code while migrating to V1.

    Warning:
        This class is deprecated and will be removed in {version}.
    """.format(version=V0_DEPRECATION_VERSION)

    def __init__(self, v1_engine: Any) -> None:
        """Initialize V0 engine wrapper.

        Args:
            v1_engine: V1 LLMEngine instance.
        """
        warnings.warn(
            "V0EngineWrapper is deprecated and will be removed "
            f"in {V0_DEPRECATION_VERSION}. Please migrate to V1 APIs. "
            f"See {V0_MIGRATION_GUIDE_URL}",
            DeprecationWarning,
            stacklevel=2
        )

        self._v1_engine = v1_engine

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to V1 engine.

        Args:
            name: Attribute name.

        Returns:
            Attribute value from V1 engine.
        """
        # Log deprecated attribute access
        logger.warning_once(
            f"Accessing engine attribute '{name}' through V0 compatibility "
            f"layer. This will be removed in {V0_DEPRECATION_VERSION}."
        )
        return getattr(self._v1_engine, name)

    @property
    def scheduler(self) -> Any:
        """Get scheduler (V0 compatibility).

        In V1, the scheduler is part of the engine core.
        This property provides a compatibility shim.

        Returns:
            Scheduler-like object or None.
        """
        warnings.warn(
            "Direct scheduler access is deprecated and will be removed "
            f"in {V0_DEPRECATION_VERSION}. Use engine_core methods instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Try to access engine core's scheduler if available
        if hasattr(self._v1_engine, 'engine_core'):
            engine_core = self._v1_engine.engine_core
            if hasattr(engine_core, 'engine_core'):
                core = engine_core.engine_core
                if hasattr(core, 'scheduler'):
                    return core.scheduler

        raise V0CompatibilityError(
            "Scheduler is not directly accessible in V1. "
            "Please migrate to V1 APIs."
        )


def wrap_v1_engine_with_v0_interface(v1_engine: Any) -> V0EngineWrapper:
    """Wrap a V1 engine with V0-compatible interface.

    This function creates a wrapper that provides V0-like attribute
    access for users migrating from V0 to V1.

    Args:
        v1_engine: V1 LLMEngine instance.

    Returns:
        V0EngineWrapper: Wrapped engine with V0 compatibility.

    Warning:
        This function is deprecated and will be removed in {version}.
    """.format(version=V0_DEPRECATION_VERSION)

    return V0EngineWrapper(v1_engine)


# Compatibility aliases for common V0 imports
# These allow `from vllm.compat.v0 import ...` for old V0 patterns

__all__ = [
    "V0CompatibilityError",
    "SchedulerOutputV0",
    "SequenceGroupV0",
    "V0EngineWrapper",
    "get_scheduler_output_v0_format",
    "get_sequence_group_v0_format",
    "wrap_v1_engine_with_v0_interface",
]
