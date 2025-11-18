"""
Production Deployment Example

This example demonstrates how to deploy vLLM for production use with:
- Optimized configuration for throughput
- Health checks and monitoring
- Graceful shutdown handling
- Logging best practices

Usage:
    python examples/offline_inference/production_deployment.py

For API server deployment, see the online_serving examples.
"""

import logging
import signal
import sys
import time
from typing import Optional

from vllm import LLM, SamplingParams

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class ProductionLLM:
    """Wrapper around vLLM LLM class with production-ready features."""

    def __init__(
        self,
        model: str,
        gpu_memory_utilization: float = 0.95,
        max_model_len: Optional[int] = 4096,
        max_num_seqs: int = 256,
        enable_prefix_caching: bool = True,
        **kwargs
    ):
        """Initialize production LLM.

        Args:
            model: HuggingFace model name or path.
            gpu_memory_utilization: Fraction of GPU memory to use (default: 0.95).
            max_model_len: Maximum sequence length (default: 4096).
            max_num_seqs: Maximum concurrent sequences (default: 256).
            enable_prefix_caching: Enable prefix caching for better throughput.
            **kwargs: Additional arguments passed to LLM.
        """
        logger.info(f"Initializing LLM with model: {model}")
        start_time = time.time()

        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=enable_prefix_caching,
            **kwargs
        )

        load_time = time.time() - start_time
        logger.info(f"LLM initialized in {load_time:.2f} seconds")

    def health_check(self) -> bool:
        """Verify LLM is responding correctly.

        Returns:
            True if health check passes, False otherwise.
        """
        try:
            output = self.llm.generate(
                ["Health check"],
                SamplingParams(max_tokens=10, temperature=0)
            )
            is_healthy = len(output) > 0 and len(output[0].outputs) > 0
            if is_healthy:
                logger.debug("Health check passed")
            else:
                logger.warning("Health check returned empty output")
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def generate(
        self,
        prompts: list[str],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs
    ):
        """Generate completions with logging.

        Args:
            prompts: List of input prompts.
            sampling_params: Sampling parameters for generation.
            **kwargs: Additional arguments passed to generate.

        Returns:
            List of RequestOutput objects.
        """
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=512
            )

        logger.info(f"Generating completions for {len(prompts)} prompts")
        start_time = time.time()

        outputs = self.llm.generate(prompts, sampling_params, **kwargs)

        generation_time = time.time() - start_time
        total_tokens = sum(
            len(output.outputs[0].token_ids)
            for output in outputs
            if output.outputs
        )
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

        logger.info(
            f"Generated {total_tokens} tokens in {generation_time:.2f}s "
            f"({tokens_per_second:.1f} tokens/s)"
        )

        return outputs


def create_default_sampling_params() -> SamplingParams:
    """Create default sampling parameters for production.

    Returns:
        SamplingParams configured for production use.
    """
    return SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
        # Useful stopping conditions
        stop=["\n\n", "###", "<|endoftext|>"],
        # Skip special tokens in output
        skip_special_tokens=True,
    )


def setup_signal_handlers():
    """Set up graceful shutdown signal handlers."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point demonstrating production deployment."""
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()

    # Initialize with production configuration
    # Using a small model for demonstration - replace with your model
    logger.info("Starting production LLM deployment")

    try:
        llm = ProductionLLM(
            model="facebook/opt-125m",  # Replace with your model
            gpu_memory_utilization=0.95,
            max_model_len=2048,
            max_num_seqs=128,
            enable_prefix_caching=True,
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        sys.exit(1)

    # Verify health before accepting requests
    if not llm.health_check():
        logger.error("Health check failed, exiting")
        sys.exit(1)

    logger.info("LLM ready for production use")

    # Example generation
    prompts = [
        "The key to successful machine learning deployment is",
        "Best practices for production systems include",
        "When scaling inference workloads, consider",
    ]

    sampling_params = create_default_sampling_params()

    try:
        outputs = llm.generate(prompts, sampling_params)

        # Process outputs
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            logger.info(f"Prompt {i + 1}: {prompt[:50]}...")
            logger.info(f"Generated: {generated_text[:100]}...")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)

    logger.info("Production deployment example completed successfully")


if __name__ == "__main__":
    main()
