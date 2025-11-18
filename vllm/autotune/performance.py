# SPDX-License-Identifier: Apache-2.0
"""Performance estimation for auto-tuning."""

from dataclasses import dataclass

from vllm.autotune.hardware import HardwareInfo
from vllm.autotune.memory import ModelConfig, get_bytes_per_param


@dataclass
class PerformanceEstimate:
    """Performance estimation results."""

    throughput: float  # tokens/second
    ttft: float  # Time to first token (seconds)
    itl: float  # Inter-token latency (seconds)
    confidence: float  # Estimation confidence (0-1)

    @property
    def throughput_k(self) -> float:
        """Throughput in K tokens/s."""
        return self.throughput / 1000

    @property
    def ttft_ms(self) -> float:
        """TTFT in milliseconds."""
        return self.ttft * 1000

    @property
    def itl_ms(self) -> float:
        """ITL in milliseconds."""
        return self.itl * 1000


def estimate_prefill_time(
    model_config: ModelConfig,
    hardware_info: HardwareInfo,
    batch_size: int,
    input_len: int,
    tensor_parallel_size: int = 1,
) -> float:
    """Estimate prefill time (compute-bound).

    The prefill phase is compute-bound as it processes all input tokens
    in parallel with matrix multiplications.

    Args:
        model_config: Model configuration.
        hardware_info: Hardware information.
        batch_size: Batch size.
        input_len: Input sequence length.
        tensor_parallel_size: Tensor parallelism degree.

    Returns:
        Estimated prefill time in seconds.
    """
    # FLOPs per token in prefill
    # Main compute: attention + FFN
    # Attention: 4 * hidden_size^2 (Q, K, V, O projections)
    # FFN: 3 * hidden_size * intermediate_size (up, gate, down)
    flops_per_token = (
        4 * model_config.hidden_size**2
        + 3 * model_config.hidden_size * model_config.intermediate_size
    ) * model_config.num_layers

    # Total FLOPs
    total_tokens = batch_size * input_len
    total_flops = total_tokens * flops_per_token * 2  # Multiply-add = 2 ops

    # Available compute
    # Account for tensor parallelism and compute efficiency (~70%)
    available_compute = (
        hardware_info.profile.compute
        * tensor_parallel_size
        * hardware_info.num_gpus
        / tensor_parallel_size
        * 0.7  # Efficiency factor
    )

    return total_flops / available_compute


def estimate_decode_time(
    model_config: ModelConfig,
    hardware_info: HardwareInfo,
    batch_size: int,
    tensor_parallel_size: int = 1,
) -> float:
    """Estimate decode time per token (memory-bound).

    The decode phase is memory-bound as it reads model weights and KV cache
    for generating one token at a time.

    Args:
        model_config: Model configuration.
        hardware_info: Hardware information.
        batch_size: Batch size.
        tensor_parallel_size: Tensor parallelism degree.

    Returns:
        Estimated time per decode step in seconds.
    """
    # Memory reads per decode step
    # 1. Model weights (read once per batch)
    bytes_per_param = get_bytes_per_param(model_config.dtype)
    weights_bytes = model_config.num_parameters * bytes_per_param

    # 2. KV cache reads (per sequence in batch)
    # This scales with sequence length, but we use average
    avg_kv_len = 512  # Average KV cache length
    kv_per_seq = (
        2  # K and V
        * model_config.num_layers
        * model_config.num_kv_heads
        * model_config.head_dim
        * bytes_per_param
        * avg_kv_len
    )
    kv_bytes = batch_size * kv_per_seq

    total_bytes = weights_bytes + kv_bytes

    # Available bandwidth
    # Account for tensor parallelism and efficiency (~80%)
    available_bandwidth = (
        hardware_info.profile.bandwidth
        * hardware_info.num_gpus
        / tensor_parallel_size
        * 0.8  # Efficiency factor
    )

    return total_bytes / available_bandwidth


def estimate_throughput(
    model_config: ModelConfig,
    hardware_info: HardwareInfo,
    batch_size: int,
    input_len: int,
    output_len: int,
    tensor_parallel_size: int = 1,
) -> PerformanceEstimate:
    """Estimate throughput for configuration.

    Args:
        model_config: Model configuration.
        hardware_info: Hardware information.
        batch_size: Batch size.
        input_len: Input sequence length.
        output_len: Output sequence length.
        tensor_parallel_size: Tensor parallelism.

    Returns:
        PerformanceEstimate with throughput metrics.
    """
    # Prefill time
    prefill_time = estimate_prefill_time(
        model_config, hardware_info, batch_size, input_len, tensor_parallel_size
    )

    # Decode time per token
    decode_time_per_token = estimate_decode_time(
        model_config, hardware_info, batch_size, tensor_parallel_size
    )

    # Total decode time
    total_decode_time = output_len * decode_time_per_token

    # Total time
    total_time = prefill_time + total_decode_time

    # Total tokens
    total_tokens = batch_size * (input_len + output_len)

    # Throughput
    throughput = total_tokens / total_time if total_time > 0 else 0

    # Confidence based on how well we match the hardware profile
    confidence = 0.8 if hardware_info.profile.name != "generic" else 0.5

    return PerformanceEstimate(
        throughput=throughput,
        ttft=prefill_time,
        itl=decode_time_per_token,
        confidence=confidence,
    )


def estimate_latency(
    model_config: ModelConfig,
    hardware_info: HardwareInfo,
    input_len: int,
    output_len: int,
    tensor_parallel_size: int = 1,
) -> PerformanceEstimate:
    """Estimate latency for single request.

    Args:
        model_config: Model configuration.
        hardware_info: Hardware information.
        input_len: Input sequence length.
        output_len: Output sequence length.
        tensor_parallel_size: Tensor parallelism.

    Returns:
        PerformanceEstimate with latency metrics.
    """
    # Single request latency
    return estimate_throughput(
        model_config,
        hardware_info,
        batch_size=1,
        input_len=input_len,
        output_len=output_len,
        tensor_parallel_size=tensor_parallel_size,
    )


def optimize_for_throughput(
    model_config: ModelConfig,
    hardware_info: HardwareInfo,
    available_memory: int,
    input_len: int = 500,
    output_len: int = 200,
) -> dict:
    """Find configuration optimized for throughput.

    Args:
        model_config: Model configuration.
        hardware_info: Hardware information.
        available_memory: Available GPU memory.
        input_len: Expected input length.
        output_len: Expected output length.

    Returns:
        Dict with optimized parameters.
    """
    from vllm.autotune.memory import calculate_max_batch_size

    best_throughput = 0
    best_config = {}

    # Try different tensor parallel sizes
    for tp_size in [1, 2, 4, 8]:
        if tp_size > hardware_info.num_gpus:
            continue

        # Calculate max batch size for this TP
        memory_per_tp = available_memory // tp_size
        max_batch = calculate_max_batch_size(
            model_config, memory_per_tp, input_len + output_len
        )

        # Try different batch sizes
        for batch_size in [max_batch, max_batch // 2, max_batch // 4]:
            if batch_size < 1:
                continue

            estimate = estimate_throughput(
                model_config,
                hardware_info,
                batch_size,
                input_len,
                output_len,
                tp_size,
            )

            if estimate.throughput > best_throughput:
                best_throughput = estimate.throughput
                best_config = {
                    "tensor_parallel_size": tp_size,
                    "max_num_seqs": batch_size,
                    "estimated_throughput": estimate.throughput,
                    "estimated_ttft": estimate.ttft,
                }

    return best_config


def optimize_for_latency(
    model_config: ModelConfig,
    hardware_info: HardwareInfo,
    available_memory: int,
    input_len: int = 500,
    output_len: int = 200,
) -> dict:
    """Find configuration optimized for latency.

    Args:
        model_config: Model configuration.
        hardware_info: Hardware information.
        available_memory: Available GPU memory.
        input_len: Expected input length.
        output_len: Expected output length.

    Returns:
        Dict with optimized parameters.
    """
    best_latency = float("inf")
    best_config = {}

    # For latency, maximize parallelism
    for tp_size in [8, 4, 2, 1]:
        if tp_size > hardware_info.num_gpus:
            continue

        estimate = estimate_latency(
            model_config,
            hardware_info,
            input_len,
            output_len,
            tp_size,
        )

        total_latency = estimate.ttft + output_len * estimate.itl

        if total_latency < best_latency:
            best_latency = total_latency
            best_config = {
                "tensor_parallel_size": tp_size,
                "max_num_seqs": 32,  # Lower batch for latency
                "estimated_ttft": estimate.ttft,
                "estimated_itl": estimate.itl,
            }

    return best_config
