# SPDX-License-Identifier: Apache-2.0
"""Hardware detection and profiles for auto-tuning."""

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class HardwareProfile:
    """Hardware profile with performance characteristics."""

    name: str
    memory: int  # bytes
    bandwidth: float  # bytes/s
    compute: float  # FLOPS (BF16)
    nvlink: bool
    recommended_attention: str
    recommended_quantization: Optional[str]

    @property
    def memory_gb(self) -> float:
        """Memory in GB."""
        return self.memory / (1024**3)

    @property
    def bandwidth_gbps(self) -> float:
        """Bandwidth in GB/s."""
        return self.bandwidth / (1024**3)

    @property
    def compute_tflops(self) -> float:
        """Compute in TFLOPS."""
        return self.compute / 1e12


# Hardware profiles with performance characteristics
HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "A100-80GB": {
        "memory": 80 * (1024**3),
        "bandwidth": 2039 * (1024**3),  # GB/s
        "compute": 312e12,  # TFLOPS BF16
        "nvlink": True,
        "recommended_attention": "FLASH_ATTN",
        "recommended_quantization": "fp8",
    },
    "A100-40GB": {
        "memory": 40 * (1024**3),
        "bandwidth": 1555 * (1024**3),
        "compute": 312e12,
        "nvlink": True,
        "recommended_attention": "FLASH_ATTN",
        "recommended_quantization": "fp8",
    },
    "H100-80GB": {
        "memory": 80 * (1024**3),
        "bandwidth": 3350 * (1024**3),
        "compute": 989e12,
        "nvlink": True,
        "recommended_attention": "FLASHINFER",
        "recommended_quantization": "fp8",
    },
    "H100-SXM": {
        "memory": 80 * (1024**3),
        "bandwidth": 3350 * (1024**3),
        "compute": 989e12,
        "nvlink": True,
        "recommended_attention": "FLASHINFER",
        "recommended_quantization": "fp8",
    },
    "H200": {
        "memory": 141 * (1024**3),
        "bandwidth": 4800 * (1024**3),
        "compute": 989e12,
        "nvlink": True,
        "recommended_attention": "FLASHINFER",
        "recommended_quantization": "fp8",
    },
    "L40S": {
        "memory": 48 * (1024**3),
        "bandwidth": 864 * (1024**3),
        "compute": 362e12,
        "nvlink": False,
        "recommended_attention": "FLASH_ATTN",
        "recommended_quantization": "awq",
    },
    "A10G": {
        "memory": 24 * (1024**3),
        "bandwidth": 600 * (1024**3),
        "compute": 125e12,
        "nvlink": False,
        "recommended_attention": "FLASH_ATTN",
        "recommended_quantization": "awq",
    },
    "RTX 4090": {
        "memory": 24 * (1024**3),
        "bandwidth": 1008 * (1024**3),
        "compute": 330e12,
        "nvlink": False,
        "recommended_attention": "FLASH_ATTN",
        "recommended_quantization": "awq",
    },
    "RTX 3090": {
        "memory": 24 * (1024**3),
        "bandwidth": 936 * (1024**3),
        "compute": 142e12,
        "nvlink": False,
        "recommended_attention": "FLASH_ATTN",
        "recommended_quantization": "gptq",
    },
    "V100-32GB": {
        "memory": 32 * (1024**3),
        "bandwidth": 900 * (1024**3),
        "compute": 125e12,
        "nvlink": True,
        "recommended_attention": "FLASH_ATTN",
        "recommended_quantization": "gptq",
    },
    "V100-16GB": {
        "memory": 16 * (1024**3),
        "bandwidth": 900 * (1024**3),
        "compute": 125e12,
        "nvlink": True,
        "recommended_attention": "FLASH_ATTN",
        "recommended_quantization": "gptq",
    },
    # Generic fallback profile
    "generic": {
        "memory": 16 * (1024**3),
        "bandwidth": 500 * (1024**3),
        "compute": 100e12,
        "nvlink": False,
        "recommended_attention": "FLASH_ATTN",
        "recommended_quantization": None,
    },
}


@dataclass
class HardwareInfo:
    """Detected hardware information."""

    profile: HardwareProfile
    num_gpus: int
    gpu_name: str
    total_memory: int  # Total memory across all GPUs

    @property
    def memory_per_gpu_gb(self) -> float:
        """Memory per GPU in GB."""
        return self.profile.memory_gb

    @property
    def total_memory_gb(self) -> float:
        """Total memory in GB."""
        return self.total_memory / (1024**3)


def get_hardware_profile(gpu_name: Optional[str] = None) -> HardwareProfile:
    """Get hardware profile for GPU.

    Args:
        gpu_name: GPU name string. If None, auto-detect.

    Returns:
        HardwareProfile for the GPU.
    """
    if gpu_name is None:
        if not torch.cuda.is_available():
            return HardwareProfile(name="generic", **HARDWARE_PROFILES["generic"])
        gpu_name = torch.cuda.get_device_name()

    # Try to match known profiles
    for profile_name, profile_data in HARDWARE_PROFILES.items():
        if profile_name == "generic":
            continue
        # Check if profile name is in GPU name
        if profile_name.replace("-", " ") in gpu_name or profile_name in gpu_name:
            return HardwareProfile(name=profile_name, **profile_data)

    # Check for partial matches
    gpu_name_upper = gpu_name.upper()
    if "A100" in gpu_name_upper:
        if "80" in gpu_name or "SXM" in gpu_name_upper:
            return HardwareProfile(name="A100-80GB", **HARDWARE_PROFILES["A100-80GB"])
        return HardwareProfile(name="A100-40GB", **HARDWARE_PROFILES["A100-40GB"])

    if "H100" in gpu_name_upper:
        return HardwareProfile(name="H100-80GB", **HARDWARE_PROFILES["H100-80GB"])

    if "H200" in gpu_name_upper:
        return HardwareProfile(name="H200", **HARDWARE_PROFILES["H200"])

    if "V100" in gpu_name_upper:
        if "32" in gpu_name:
            return HardwareProfile(name="V100-32GB", **HARDWARE_PROFILES["V100-32GB"])
        return HardwareProfile(name="V100-16GB", **HARDWARE_PROFILES["V100-16GB"])

    if "L40" in gpu_name_upper:
        return HardwareProfile(name="L40S", **HARDWARE_PROFILES["L40S"])

    if "A10" in gpu_name_upper:
        return HardwareProfile(name="A10G", **HARDWARE_PROFILES["A10G"])

    if "4090" in gpu_name:
        return HardwareProfile(name="RTX 4090", **HARDWARE_PROFILES["RTX 4090"])

    if "3090" in gpu_name:
        return HardwareProfile(name="RTX 3090", **HARDWARE_PROFILES["RTX 3090"])

    # Fallback to generic profile
    return HardwareProfile(name="generic", **HARDWARE_PROFILES["generic"])


def get_hardware_info() -> HardwareInfo:
    """Detect hardware and return info.

    Returns:
        HardwareInfo with detected hardware characteristics.
    """
    if not torch.cuda.is_available():
        profile = get_hardware_profile()
        return HardwareInfo(
            profile=profile,
            num_gpus=0,
            gpu_name="CPU",
            total_memory=0,
        )

    num_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name()
    profile = get_hardware_profile(gpu_name)

    # Get actual memory from first GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory * num_gpus

    return HardwareInfo(
        profile=profile,
        num_gpus=num_gpus,
        gpu_name=gpu_name,
        total_memory=total_memory,
    )
