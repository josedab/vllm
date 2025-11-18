#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V0 to V1 feature parity checker.

This script checks feature parity between the deprecated V0 engine
and the current V1 engine. It helps identify migration gaps and
tracks the deprecation progress.

Usage:
    python scripts/v0_v1_parity.py [--verbose] [--json]

Options:
    --verbose    Show detailed information about each feature
    --json       Output results as JSON
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum


class FeatureStatus(str, Enum):
    """Feature support status."""
    FULL = "full"  # Fully supported in V1
    PARTIAL = "partial"  # Partially supported, some gaps
    UNSUPPORTED = "unsupported"  # Not supported in V1
    DEPRECATED = "deprecated"  # Deprecated, will be removed


@dataclass
class Feature:
    """Feature definition for parity checking."""
    name: str
    description: str
    v1_status: FeatureStatus
    notes: str = ""
    migration_guide: str = ""


@dataclass
class ParityReport:
    """Parity check report."""
    total_features: int = 0
    full_support: int = 0
    partial_support: int = 0
    unsupported: int = 0
    deprecated: int = 0
    features: list[Feature] = field(default_factory=list)
    parity_percentage: float = 0.0


# V0 Features to check for V1 parity
V0_FEATURES: list[Feature] = [
    Feature(
        name="continuous_batching",
        description="Dynamic batching of incoming requests",
        v1_status=FeatureStatus.FULL,
        notes="Fully supported with improved implementation in V1",
    ),
    Feature(
        name="prefix_caching",
        description="Cache common prefixes to reduce computation",
        v1_status=FeatureStatus.FULL,
        notes="Supported with automatic prefix detection",
    ),
    Feature(
        name="speculative_decoding",
        description="Use draft model to speculate future tokens",
        v1_status=FeatureStatus.FULL,
        notes="Fully supported in V1",
    ),
    Feature(
        name="lora",
        description="LoRA adapter support for fine-tuning",
        v1_status=FeatureStatus.FULL,
        notes="Full LoRA support including multi-LoRA",
    ),
    Feature(
        name="multimodal",
        description="Support for multimodal models (images, video)",
        v1_status=FeatureStatus.FULL,
        notes="Comprehensive multimodal support in V1",
    ),
    Feature(
        name="tensor_parallelism",
        description="Distribute model across multiple GPUs",
        v1_status=FeatureStatus.FULL,
        notes="Improved tensor parallelism in V1",
    ),
    Feature(
        name="pipeline_parallelism",
        description="Pipeline model layers across GPUs",
        v1_status=FeatureStatus.FULL,
        notes="Supported in V1",
    ),
    Feature(
        name="quantization",
        description="Model quantization (AWQ, GPTQ, FP8)",
        v1_status=FeatureStatus.FULL,
        notes="All quantization methods supported",
    ),
    Feature(
        name="streaming",
        description="Stream generated tokens as they're produced",
        v1_status=FeatureStatus.FULL,
        notes="Improved streaming with lower latency",
    ),
    Feature(
        name="best_of",
        description="Generate multiple sequences and return best",
        v1_status=FeatureStatus.DEPRECATED,
        notes="V0-only feature, not supported in V1",
        migration_guide="Use n parameter with custom ranking instead",
    ),
    Feature(
        name="beam_search",
        description="Beam search decoding",
        v1_status=FeatureStatus.FULL,
        notes="Supported via BeamSearchParams",
    ),
    Feature(
        name="structured_output",
        description="JSON schema, regex, and grammar constraints",
        v1_status=FeatureStatus.FULL,
        notes="Full structured output support with multiple backends",
    ),
    Feature(
        name="async_api",
        description="Asynchronous API for high throughput",
        v1_status=FeatureStatus.FULL,
        notes="AsyncLLMEngine with improved performance",
    ),
    Feature(
        name="openai_api",
        description="OpenAI-compatible API server",
        v1_status=FeatureStatus.FULL,
        notes="Full OpenAI API compatibility",
    ),
    Feature(
        name="flash_attention",
        description="Flash attention kernel support",
        v1_status=FeatureStatus.FULL,
        notes="Multiple attention backends supported",
    ),
    Feature(
        name="paged_attention",
        description="Paged KV cache management",
        v1_status=FeatureStatus.FULL,
        notes="Core feature of vLLM",
    ),
    Feature(
        name="cuda_graphs",
        description="CUDA graph capture for reduced overhead",
        v1_status=FeatureStatus.FULL,
        notes="Automatic CUDA graph optimization",
    ),
    Feature(
        name="embeddings",
        description="Generate embeddings from models",
        v1_status=FeatureStatus.FULL,
        notes="Embedding API fully supported",
    ),
    Feature(
        name="pooling",
        description="Pooling for embedding models",
        v1_status=FeatureStatus.FULL,
        notes="Multiple pooling strategies supported",
    ),
    Feature(
        name="classification",
        description="Classification task support",
        v1_status=FeatureStatus.FULL,
        notes="Supported for classifier models",
    ),
]


def check_parity() -> ParityReport:
    """Check feature parity between V0 and V1.

    Returns:
        ParityReport: Detailed parity report.
    """
    report = ParityReport()
    report.total_features = len(V0_FEATURES)
    report.features = V0_FEATURES

    for feature in V0_FEATURES:
        if feature.v1_status == FeatureStatus.FULL:
            report.full_support += 1
        elif feature.v1_status == FeatureStatus.PARTIAL:
            report.partial_support += 1
        elif feature.v1_status == FeatureStatus.UNSUPPORTED:
            report.unsupported += 1
        elif feature.v1_status == FeatureStatus.DEPRECATED:
            report.deprecated += 1

    # Calculate parity percentage (full + partial)
    if report.total_features > 0:
        supported = report.full_support + (report.partial_support * 0.5)
        report.parity_percentage = (supported / report.total_features) * 100

    return report


def get_missing_features() -> list[str]:
    """Get list of V0 features not fully supported in V1.

    Returns:
        list[str]: List of feature names that are not fully supported.
    """
    missing = []
    for feature in V0_FEATURES:
        if feature.v1_status in (FeatureStatus.UNSUPPORTED, FeatureStatus.DEPRECATED):
            missing.append(feature.name)
    return missing


def get_migration_blockers() -> list[Feature]:
    """Get features that may block migration from V0 to V1.

    Returns:
        list[Feature]: Features that are unsupported or deprecated.
    """
    blockers = []
    for feature in V0_FEATURES:
        if feature.v1_status in (FeatureStatus.UNSUPPORTED, FeatureStatus.DEPRECATED):
            blockers.append(feature)
    return blockers


def print_report(report: ParityReport, verbose: bool = False) -> None:
    """Print parity report to console.

    Args:
        report: Parity report to print.
        verbose: Whether to show detailed feature information.
    """
    print("=" * 60)
    print("V0 to V1 Feature Parity Report")
    print("=" * 60)
    print()
    print(f"Total Features Checked: {report.total_features}")
    print(f"Full Support:           {report.full_support}")
    print(f"Partial Support:        {report.partial_support}")
    print(f"Unsupported:            {report.unsupported}")
    print(f"Deprecated:             {report.deprecated}")
    print()
    print(f"Parity Percentage:      {report.parity_percentage:.1f}%")
    print()

    if verbose:
        print("Feature Details:")
        print("-" * 60)
        for feature in report.features:
            status_symbol = {
                FeatureStatus.FULL: "✓",
                FeatureStatus.PARTIAL: "◐",
                FeatureStatus.UNSUPPORTED: "✗",
                FeatureStatus.DEPRECATED: "⚠",
            }.get(feature.v1_status, "?")

            print(f"\n{status_symbol} {feature.name}")
            print(f"  Status: {feature.v1_status.value}")
            print(f"  Description: {feature.description}")
            if feature.notes:
                print(f"  Notes: {feature.notes}")
            if feature.migration_guide:
                print(f"  Migration: {feature.migration_guide}")

    # Print migration blockers
    blockers = get_migration_blockers()
    if blockers:
        print()
        print("Migration Blockers:")
        print("-" * 60)
        for blocker in blockers:
            print(f"  - {blocker.name}: {blocker.notes}")
            if blocker.migration_guide:
                print(f"    Migration: {blocker.migration_guide}")


def print_json_report(report: ParityReport) -> None:
    """Print parity report as JSON.

    Args:
        report: Parity report to print.
    """
    # Convert to serializable format
    output = {
        "total_features": report.total_features,
        "full_support": report.full_support,
        "partial_support": report.partial_support,
        "unsupported": report.unsupported,
        "deprecated": report.deprecated,
        "parity_percentage": report.parity_percentage,
        "features": [
            {
                "name": f.name,
                "description": f.description,
                "v1_status": f.v1_status.value,
                "notes": f.notes,
                "migration_guide": f.migration_guide,
            }
            for f in report.features
        ],
    }
    print(json.dumps(output, indent=2))


def main() -> int:
    """Main entry point.

    Returns:
        int: Exit code (0 for success, 1 if blockers found).
    """
    parser = argparse.ArgumentParser(
        description="Check V0 to V1 feature parity"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed feature information"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()

    report = check_parity()

    if args.json:
        print_json_report(report)
    else:
        print_report(report, verbose=args.verbose)

    # Return non-zero if there are migration blockers
    blockers = get_migration_blockers()
    if blockers:
        if not args.json:
            print()
            print(f"Warning: {len(blockers)} feature(s) may block migration")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
