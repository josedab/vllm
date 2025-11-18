# SPDX-License-Identifier: Apache-2.0
"""Tests for the autotune CLI command."""

import argparse
import tempfile
from pathlib import Path

import pytest

from vllm.entrypoints.cli.tune import TuneSubcommand


class TestTuneSubcommand:
    """Tests for the tune CLI subcommand."""

    def test_subcommand_name(self):
        """Test subcommand name."""
        cmd = TuneSubcommand()
        assert cmd.name == "tune"

    def test_subparser_init(self):
        """Test subparser initialization."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        cmd = TuneSubcommand()
        tune_parser = cmd.subparser_init(subparsers)

        assert tune_parser is not None

    def test_validate_valid_optimize_for(self):
        """Test validation with valid optimize_for values."""
        cmd = TuneSubcommand()

        for opt in ["throughput", "latency", "memory"]:
            args = argparse.Namespace(optimize_for=opt)
            cmd.validate(args)  # Should not raise

    def test_validate_invalid_optimize_for(self):
        """Test validation with invalid optimize_for value."""
        cmd = TuneSubcommand()

        args = argparse.Namespace(optimize_for="invalid")
        with pytest.raises(ValueError, match="Invalid optimize_for value"):
            cmd.validate(args)

    def test_cmd_execution(self, capsys):
        """Test command execution."""
        cmd = TuneSubcommand()

        args = argparse.Namespace(
            model="gpt2",
            optimize_for="throughput",
            expected_batch_size=None,
            expected_input_length=500,
            expected_output_length=200,
            output=None,
        )

        cmd.cmd(args)

        captured = capsys.readouterr()
        assert "Model: gpt2" in captured.out
        assert "Recommendations:" in captured.out
        assert "tensor_parallel_size:" in captured.out

    def test_cmd_with_output_file(self, capsys):
        """Test command execution with output file."""
        cmd = TuneSubcommand()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            args = argparse.Namespace(
                model="gpt2",
                optimize_for="throughput",
                expected_batch_size=None,
                expected_input_length=500,
                expected_output_length=200,
                output=output_path,
            )

            cmd.cmd(args)

            # Check that file was created
            assert Path(output_path).exists()

            # Check file content
            with open(output_path) as f:
                content = f.read()
                assert "model:" in content
                assert "tensor_parallel_size:" in content

            captured = capsys.readouterr()
            assert f"Configuration saved to: {output_path}" in captured.out
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_cmd_latency_optimization(self, capsys):
        """Test command with latency optimization."""
        cmd = TuneSubcommand()

        args = argparse.Namespace(
            model="gpt2",
            optimize_for="latency",
            expected_batch_size=None,
            expected_input_length=500,
            expected_output_length=200,
            output=None,
        )

        cmd.cmd(args)

        captured = capsys.readouterr()
        assert "Model: gpt2" in captured.out


class TestCmdInit:
    """Tests for cmd_init function."""

    def test_cmd_init_returns_list(self):
        """Test that cmd_init returns a list of subcommands."""
        from vllm.entrypoints.cli.tune import cmd_init

        cmds = cmd_init()

        assert isinstance(cmds, list)
        assert len(cmds) == 1
        assert isinstance(cmds[0], TuneSubcommand)
