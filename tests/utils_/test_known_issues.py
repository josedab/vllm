# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the known issues database and error enhancement."""

import pytest

from vllm.exceptions import (
    VLLMConfigurationError,
    VLLMDistributedError,
    VLLMError,
    VLLMMemoryError,
    VLLMModelLoadError,
)
from vllm.utils.known_issues import (
    DOCS_BASE_URL,
    KNOWN_ISSUES,
    enhance_error,
    get_issue_info,
    list_known_issues,
)


class TestKnownIssuesDatabase:
    """Tests for the KNOWN_ISSUES database structure."""

    def test_known_issues_has_required_keys(self):
        """Test that all known issues have required keys."""
        required_keys = {"pattern", "message", "solutions", "docs"}

        for issue_id, issue in KNOWN_ISSUES.items():
            for key in required_keys:
                assert key in issue, f"Issue {issue_id} missing key: {key}"

    def test_known_issues_solutions_are_lists(self):
        """Test that all solutions are lists."""
        for issue_id, issue in KNOWN_ISSUES.items():
            assert isinstance(
                issue["solutions"], list
            ), f"Issue {issue_id} solutions should be a list"
            assert len(issue["solutions"]) > 0, f"Issue {issue_id} has no solutions"

    def test_known_issues_patterns_are_valid_regex(self):
        """Test that all patterns are valid regex."""
        import re

        for issue_id, issue in KNOWN_ISSUES.items():
            try:
                re.compile(issue["pattern"])
            except re.error as e:
                pytest.fail(f"Issue {issue_id} has invalid regex: {e}")

    def test_known_issues_have_exception_classes(self):
        """Test that issues have valid exception classes."""
        for issue_id, issue in KNOWN_ISSUES.items():
            if "exception_class" in issue:
                assert issubclass(
                    issue["exception_class"], VLLMError
                ), f"Issue {issue_id} exception_class must be VLLMError subclass"


class TestEnhanceError:
    """Tests for the enhance_error function."""

    def test_enhance_cuda_oom_error(self):
        """Test enhancing CUDA out of memory error."""
        original = RuntimeError("CUDA out of memory. Tried to allocate 2GB")
        enhanced = enhance_error(original)

        assert isinstance(enhanced, VLLMMemoryError)
        assert "GPU memory" in str(enhanced)
        assert "Solutions:" in str(enhanced)
        assert "Documentation:" in str(enhanced)

    def test_enhance_nccl_timeout_error(self):
        """Test enhancing NCCL timeout error."""
        original = RuntimeError("NCCL error: timeout occurred")
        enhanced = enhance_error(original)

        assert isinstance(enhanced, VLLMDistributedError)
        assert "Distributed" in str(enhanced) or "timeout" in str(enhanced).lower()

    def test_enhance_model_not_found_error(self):
        """Test enhancing model not found error."""
        original = OSError("Repository not found for model xyz")
        enhanced = enhance_error(original)

        assert isinstance(enhanced, VLLMModelLoadError)
        assert "Solutions:" in str(enhanced)

    def test_enhance_auth_error(self):
        """Test enhancing authentication error."""
        original = OSError("401 unauthorized access denied")
        enhanced = enhance_error(original)

        assert isinstance(enhanced, VLLMModelLoadError)

    def test_enhance_tensor_parallel_error(self):
        """Test enhancing tensor parallel configuration error."""
        original = ValueError("tensor parallel size exceeds available GPUs")
        enhanced = enhance_error(original)

        assert isinstance(enhanced, VLLMConfigurationError)

    def test_enhance_context_length_error(self):
        """Test enhancing context length error."""
        original = ValueError("context length exceeds maximum supported")
        enhanced = enhance_error(original)

        assert isinstance(enhanced, VLLMConfigurationError)

    def test_enhance_unknown_error(self):
        """Test that unknown errors are wrapped in generic VLLMError."""
        original = RuntimeError("Some completely unknown error xyz123")
        enhanced = enhance_error(original)

        # Should return a VLLMError wrapping the original
        assert isinstance(enhanced, VLLMError)
        assert "xyz123" in str(enhanced)

    def test_enhance_preserves_original_in_context(self):
        """Test that enhanced error includes original error in context."""
        original = RuntimeError("CUDA out of memory")
        enhanced = enhance_error(original)

        assert "Original error" in enhanced.context or len(enhanced.context) > 0

    def test_enhance_includes_docs_url(self):
        """Test that enhanced error includes documentation URL."""
        original = RuntimeError("CUDA out of memory")
        enhanced = enhance_error(original)

        assert enhanced.docs_url is not None
        assert DOCS_BASE_URL in enhanced.docs_url

    def test_enhance_case_insensitive(self):
        """Test that pattern matching is case insensitive."""
        # Test with different case variations
        errors = [
            RuntimeError("cuda OUT OF memory"),
            RuntimeError("Cuda Out Of Memory"),
            RuntimeError("CUDA OUT OF MEMORY"),
        ]

        for error in errors:
            enhanced = enhance_error(error)
            assert isinstance(enhanced, VLLMMemoryError)


class TestGetIssueInfo:
    """Tests for the get_issue_info function."""

    def test_get_existing_issue(self):
        """Test retrieving an existing issue."""
        info = get_issue_info("CUDA_OOM")

        assert info is not None
        assert "pattern" in info
        assert "solutions" in info

    def test_get_nonexistent_issue(self):
        """Test retrieving a non-existent issue."""
        info = get_issue_info("NONEXISTENT_ISSUE_XYZ")

        assert info is None


class TestListKnownIssues:
    """Tests for the list_known_issues function."""

    def test_list_returns_all_issues(self):
        """Test that list returns all known issue IDs."""
        issue_list = list_known_issues()

        assert isinstance(issue_list, list)
        assert len(issue_list) == len(KNOWN_ISSUES)

    def test_list_contains_expected_issues(self):
        """Test that list contains expected common issues."""
        issue_list = list_known_issues()

        expected = ["CUDA_OOM", "NCCL_TIMEOUT", "MODEL_NOT_FOUND"]
        for issue_id in expected:
            assert issue_id in issue_list, f"Expected issue {issue_id} not found"


class TestDocsBaseUrl:
    """Tests for the DOCS_BASE_URL constant."""

    def test_docs_base_url_format(self):
        """Test that docs base URL is properly formatted."""
        assert DOCS_BASE_URL.startswith("https://")
        assert "vllm" in DOCS_BASE_URL
        assert not DOCS_BASE_URL.endswith("/")
