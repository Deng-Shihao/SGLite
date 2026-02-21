import argparse
import asyncio
import random
import sys
from typing import List

from minisgl.benchmark.client import (
    benchmark_one,
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)


async def main(args):
    """
    Benchmark script for vLLM using OpenAI-compatible API.

    This script tests vLLM's performance with various batch sizes.
    vLLM should be running on the specified port with OpenAI-compatible API enabled.

    To run vLLM server:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-4B-AWQ \
        --port 8000 \
        --gpu-memory-utilization 0.95
    """
    try:
        random.seed(42)  # reproducibility

        async def generate_task(max_bs: int) -> List[str]:
            """Generate a list of tasks with random lengths."""
            result = []
            for _ in range(max_bs):
                length = random.randint(1, args.max_input)
                message = generate_prompt(tokenizer, length)
                result.append(message)
                await asyncio.sleep(0)
            return result

        # Configuration from args
        TEST_BS = args.batch_sizes
        PORT = args.port
        MODEL_PATH = args.model  # Model path for tokenizer fallback

        # Create the async client for vLLM
        async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="EMPTY") as client:
            logger.info("Connecting to vLLM server...")

            # Get model name from server
            try:
                MODEL = await get_model_name(client)
                logger.info(f"Connected to model: {MODEL}")
            except Exception as e:
                logger.error(f"Failed to get model name from vLLM server: {e}")
                logger.error(f"Make sure vLLM is running on http://127.0.0.1:{PORT}")
                sys.exit(1)

            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                logger.info(f"Loaded tokenizer from {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from {MODEL_PATH}: {e}")
                logger.warning(f"Trying to use server model name: {MODEL}...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(MODEL)
                    logger.info(f"Loaded tokenizer from {MODEL}")
                except Exception as e2:
                    logger.warning(f"Failed to load tokenizer from {MODEL}: {e2}")
                    logger.warning("Using Llama-2 tokenizer as fallback...")
                    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                    logger.info("Using Llama-2 tokenizer as fallback")

            logger.info("Testing connection to vLLM server...")

            # Test connection with a simple request first
            try:
                # Generate test tasks in advance
                gen_task = asyncio.create_task(generate_task(max(TEST_BS)))

                test_msg = generate_prompt(tokenizer, 100)
                test_result = await benchmark_one(
                    client, test_msg, 2, MODEL, pbar=False
                )

                if len(test_result.tics) <= 2:
                    logger.error("Server connection test failed - no tokens generated")
                    sys.exit(1)

                logger.info("✓ vLLM server connection successful")
            except Exception as e:
                logger.error("✗ vLLM server connection failed")
                logger.error(f"Error: {e}")
                logger.error(
                    f"Make sure vLLM is running on http://127.0.0.1:{PORT}\n"
                    "Start vLLM with:\n"
                    f"  python -m vllm.entrypoints.openai.api_server --model <model> --port {PORT}"
                )
                sys.exit(1)

            # Generate test messages and output lengths
            msgs = await gen_task
            output_lengths = [
                random.randint(args.output_min, args.output_max) for _ in range(max(TEST_BS))
            ]
            logger.info(f"Generated {len(msgs)} test messages")
            logger.info(f"Output lengths range: {args.output_min}-{args.output_max} tokens")

            # Run benchmark for different batch sizes
            logger.info("=" * 80)
            logger.info("Starting vLLM benchmark...")
            logger.info("=" * 80)

            for batch_size in TEST_BS:
                logger.info(f"\n{'='*80}")
                logger.info(f"Testing batch size: {batch_size}")
                logger.info(f"{'='*80}")

                try:
                    results = await benchmark_one_batch(
                        client,
                        msgs[:batch_size],
                        output_lengths[:batch_size],
                        MODEL,
                        # vLLM-specific extra_body parameters can be added here
                        extra_body={
                            "ignore_eos": True,  # For consistent comparison
                            "top_k": 1,  # Greedy decoding
                        },
                    )
                    process_benchmark_results(results)

                except Exception as e:
                    logger.error(f"✗ Error with batch size {batch_size}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            logger.info("\n" + "=" * 80)
            logger.info("vLLM Benchmark completed.")
            logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM server using OpenAI-compatible API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-AWQ",
        help="Model name or path (used for loading tokenizer)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port where vLLM server is running",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--max-input",
        type=int,
        default=8192,
        help="Maximum input length in tokens",
    )
    parser.add_argument(
        "--output-min",
        type=int,
        default=16,
        help="Minimum output length in tokens",
    )
    parser.add_argument(
        "--output-max",
        type=int,
        default=1024,
        help="Maximum output length in tokens",
    )

    args = parser.parse_args()

    logger.info("vLLM Benchmark Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Batch sizes: {args.batch_sizes}")
    logger.info(f"  Max input: {args.max_input}")
    logger.info(f"  Output range: {args.output_min}-{args.output_max}")
    logger.info("")

    asyncio.run(main(args))
