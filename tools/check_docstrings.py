#!/usr/bin/env python3
"""
Documentation coverage checker for vLLM.

This tool checks docstring coverage for public functions and classes
in the vLLM codebase. It reports coverage statistics and can be used
in CI to enforce documentation standards.

Usage:
    python tools/check_docstrings.py [--threshold=90] [--verbose]

Exit codes:
    0: Coverage meets threshold
    1: Coverage below threshold
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import NamedTuple


class CoverageResult(NamedTuple):
    """Result of checking a single file."""
    file_path: str
    documented: int
    total: int
    undocumented_items: list[str]


def check_module(path: Path, verbose: bool = False) -> CoverageResult:
    """Check docstring coverage for a single module.

    Args:
        path: Path to the Python file to check.
        verbose: If True, collect names of undocumented items.

    Returns:
        CoverageResult with coverage statistics.
    """
    try:
        with open(path, encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        if verbose:
            print(f"Warning: Could not parse {path}: {e}", file=sys.stderr)
        return CoverageResult(str(path), 0, 0, [])

    total = 0
    documented = 0
    undocumented_items = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Skip private items (starting with underscore)
            if node.name.startswith('_'):
                continue

            # Skip items defined inside other functions (local functions)
            # We only want top-level and class-level items
            total += 1

            if ast.get_docstring(node):
                documented += 1
            elif verbose:
                undocumented_items.append(f"{node.name} (line {node.lineno})")

    return CoverageResult(str(path), documented, total, undocumented_items)


def check_directory(
    directory: Path,
    verbose: bool = False,
    exclude_patterns: list[str] | None = None
) -> list[CoverageResult]:
    """Check docstring coverage for all Python files in a directory.

    Args:
        directory: Root directory to check.
        verbose: If True, collect names of undocumented items.
        exclude_patterns: Patterns to exclude from checking.

    Returns:
        List of CoverageResult objects.
    """
    if exclude_patterns is None:
        exclude_patterns = ['test', 'tests', '__pycache__', '.git']

    results = []

    for py_file in directory.rglob("*.py"):
        # Skip excluded patterns
        path_str = str(py_file)
        if any(pattern in path_str for pattern in exclude_patterns):
            continue

        result = check_module(py_file, verbose)
        if result.total > 0:
            results.append(result)

    return results


def print_summary(
    results: list[CoverageResult],
    threshold: float,
    verbose: bool = False
) -> bool:
    """Print coverage summary and return whether threshold is met.

    Args:
        results: List of coverage results.
        threshold: Minimum coverage percentage required.
        verbose: If True, print detailed information.

    Returns:
        True if coverage meets threshold, False otherwise.
    """
    total_documented = sum(r.documented for r in results)
    total_items = sum(r.total for r in results)

    if total_items == 0:
        print("No public items found to check.")
        return True

    coverage = total_documented / total_items * 100

    print("=" * 60)
    print("vLLM Docstring Coverage Report")
    print("=" * 60)
    print(f"Total public items: {total_items}")
    print(f"Documented items:   {total_documented}")
    print(f"Undocumented items: {total_items - total_documented}")
    print(f"Coverage:           {coverage:.1f}%")
    print(f"Threshold:          {threshold:.1f}%")
    print("=" * 60)

    if verbose:
        # Show files with lowest coverage
        sorted_results = sorted(
            results,
            key=lambda r: (r.documented / r.total if r.total > 0 else 1.0)
        )

        print("\nFiles with lowest coverage:")
        print("-" * 60)

        for result in sorted_results[:10]:
            if result.total > 0:
                file_coverage = result.documented / result.total * 100
                print(f"  {result.file_path}")
                print(f"    Coverage: {file_coverage:.1f}% ({result.documented}/{result.total})")
                if result.undocumented_items:
                    print(f"    Missing docs for: {', '.join(result.undocumented_items[:3])}")
                    if len(result.undocumented_items) > 3:
                        print(f"      ... and {len(result.undocumented_items) - 3} more")
                print()

    if coverage < threshold:
        print(f"\n❌ FAIL: Coverage {coverage:.1f}% is below threshold {threshold:.1f}%")
        return False
    else:
        print(f"\n✓ PASS: Coverage {coverage:.1f}% meets threshold {threshold:.1f}%")
        return True


def main():
    """Main entry point for docstring coverage checker."""
    parser = argparse.ArgumentParser(
        description="Check docstring coverage for vLLM codebase."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Minimum coverage percentage required (default: 90)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including undocumented items"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="vllm",
        help="Path to check (default: vllm)"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=["test", "tests", "__pycache__", ".git", "third_party"],
        help="Patterns to exclude from checking"
    )

    args = parser.parse_args()

    # Find the vllm directory
    vllm_path = Path(args.path)
    if not vllm_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        vllm_path = script_dir / args.path

    if not vllm_path.exists():
        print(f"Error: Path '{args.path}' not found", file=sys.stderr)
        sys.exit(1)

    print(f"Checking docstring coverage in: {vllm_path.absolute()}")
    print()

    results = check_directory(
        vllm_path,
        verbose=args.verbose,
        exclude_patterns=args.exclude
    )

    if not results:
        print("No Python files found to check.")
        sys.exit(0)

    passed = print_summary(results, args.threshold, args.verbose)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
