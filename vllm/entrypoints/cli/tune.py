# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-tune CLI command for vLLM."""

import argparse

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

DESCRIPTION = """Analyze a model and hardware to recommend optimal vLLM configuration.

This command analyzes the model architecture, available hardware, and workload
hints to generate configuration recommendations for optimal performance.

Examples:
  vllm tune meta-llama/Llama-2-7b-hf
  vllm tune meta-llama/Llama-2-70b-hf --optimize-for throughput
  vllm tune mistralai/Mixtral-8x7B-v0.1 --optimize-for latency --output config.yaml
"""


class TuneSubcommand(CLISubcommand):
    """The `tune` subcommand for the vLLM CLI."""

    name = "tune"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm.autotune import AutoTuner, TuningHints

        # Build hints from args
        hints = TuningHints(
            expected_batch_size=args.expected_batch_size,
            expected_input_length=args.expected_input_length,
            expected_output_length=args.expected_output_length,
        )

        # Run analysis
        tuner = AutoTuner()
        result = tuner.analyzer.analyze(
            model_name=args.model,
            optimize_for=args.optimize_for,
            hints=hints,
        )

        # Get hardware info
        hardware_info = tuner.analyzer.get_hardware_info()

        # Print results
        print(f"\nModel: {args.model}")
        print(f"GPU: {hardware_info.num_gpus}x {hardware_info.gpu_name}")
        print()
        print("Recommendations:")
        print(f"  tensor_parallel_size: {result.tensor_parallel_size} "
              f"({result.explanations.get('tensor_parallel_size', '')})")
        print(f"  max_num_seqs: {result.max_num_seqs} "
              f"({result.explanations.get('max_num_seqs', '')})")
        print(f"  max_model_len: {result.max_model_len} "
              f"({result.explanations.get('max_model_len', '')})")
        print(f"  quantization: {result.quantization or 'None'} "
              f"({result.explanations.get('quantization', '')})")
        print(f"  attention_backend: {result.attention_backend} "
              f"({result.explanations.get('attention_backend', '')})")
        print(f"  enable_prefix_caching: {result.enable_prefix_caching}")
        print(f"  gpu_memory_utilization: {result.gpu_memory_utilization}")
        print()
        print("Estimated Performance:")
        print(f"  Throughput: {result.estimated_throughput:,.0f} tok/s")
        print(f"  TTFT: {result.estimated_ttft * 1000:.1f}ms")
        print(f"  Memory: {result.estimated_memory:.1f}GB")
        print(f"  Confidence: {result.confidence:.0%}")
        print()

        # Output configuration file if requested
        if args.output:
            import yaml

            config = result.to_engine_args()
            config["model"] = args.model

            with open(args.output, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Configuration saved to: {args.output}")
            print()

        # Print usage example
        print("To use these recommendations:")
        quant_arg = f" --quantization {result.quantization}" if result.quantization else ""
        print(f"  vllm serve {args.model} "
              f"--tensor-parallel-size {result.tensor_parallel_size} "
              f"--max-num-seqs {result.max_num_seqs} "
              f"--max-model-len {result.max_model_len}{quant_arg}")

    def validate(self, args: argparse.Namespace) -> None:
        if args.optimize_for not in ("throughput", "latency", "memory"):
            raise ValueError(
                f"Invalid optimize_for value: {args.optimize_for}. "
                "Must be 'throughput', 'latency', or 'memory'."
            )

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        tune_parser = subparsers.add_parser(
            self.name,
            description=DESCRIPTION,
            usage="vllm tune <model> [options]",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Required arguments
        tune_parser.add_argument(
            "model",
            type=str,
            help="Model name or path to analyze",
        )

        # Optimization target
        tune_parser.add_argument(
            "--optimize-for",
            type=str,
            default="throughput",
            choices=["throughput", "latency", "memory"],
            help="Optimization target (default: throughput)",
        )

        # Workload hints
        tune_parser.add_argument(
            "--expected-batch-size",
            type=int,
            default=None,
            help="Expected batch size (default: auto-detect)",
        )
        tune_parser.add_argument(
            "--expected-input-length",
            type=int,
            default=500,
            help="Expected input sequence length (default: 500)",
        )
        tune_parser.add_argument(
            "--expected-output-length",
            type=int,
            default=200,
            help="Expected output sequence length (default: 200)",
        )

        # Output options
        tune_parser.add_argument(
            "--output",
            "-o",
            type=str,
            default=None,
            help="Output configuration to YAML file",
        )

        tune_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return tune_parser


def cmd_init() -> list[CLISubcommand]:
    return [TuneSubcommand()]
