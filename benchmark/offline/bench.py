# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import time
import torch
from random import randint, seed
from datetime import datetime

from minisgl.core import SamplingParams
from minisgl.llm import LLM


# Benchmark Configuration
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MODELS = [
    {"name": "Qwen/Qwen3-0.6B", "dtype": None},
    # {"name": "Qwen/Qwen3-4B", "dtype": None},
    # {"name": "Qwen/Qwen3-4B-AWQ", "dtype": torch.float16},
]


def run_benchmark(llm, num_seqs, max_input_len, max_output_len):
    """Run a single benchmark with given parameters."""
    seed(0)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
        for _ in range(num_seqs)
    ]

    # Warm up
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))

    # Benchmark
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t

    return {"num_seqs": num_seqs, "total_tokens": total_tokens, "time": t, "throughput": throughput}


def main():
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.txt"

    results = []

    # Header
    header = f"""
================================================================================
                         OFFLINE BENCHMARK RESULTS
================================================================================
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Max Input Length: {MAX_INPUT_LEN}
Max Output Length: {MAX_OUTPUT_LEN}
Batch Sizes: {BATCH_SIZES}
Models: {[m["name"] for m in MODELS]}
================================================================================
"""
    print(header)
    results.append(header)

    for model_config in MODELS:
        model_name = model_config["name"]
        model_dtype = model_config["dtype"]

        model_header = f"\n{'=' * 80}\nModel: {model_name}\n{'=' * 80}\n"
        print(model_header)
        results.append(model_header)

        try:
            # Initialize LLM with appropriate dtype
            if model_dtype is not None:
                llm = LLM(
                    model_name,
                    dtype=model_dtype,
                    max_seq_len_override=4096,
                    max_extend_tokens=16384,
                    cuda_graph_max_bs=256,
                )
            else:
                llm = LLM(
                    model_name,
                    max_seq_len_override=4096,
                    max_extend_tokens=16384,
                    cuda_graph_max_bs=256,
                )

            # Table header
            table_header = f"{'Batch Size':>12} | {'Total Tokens':>14} | {'Time (s)':>10} | {'Throughput (tok/s)':>20}"
            separator = "-" * len(table_header)
            print(table_header)
            print(separator)
            results.append(table_header)
            results.append(separator)

            for batch_size in BATCH_SIZES:
                try:
                    result = run_benchmark(llm, batch_size, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
                    row = f"{result['num_seqs']:>12} | {result['total_tokens']:>14} | {result['time']:>10.2f} | {result['throughput']:>20.2f}"
                    print(row)
                    results.append(row)
                except Exception as e:
                    error_row = f"{batch_size:>12} | ERROR: {str(e)[:50]}"
                    print(error_row)
                    results.append(error_row)

            # Clean up model to free GPU memory
            llm.shutdown()
            del llm
            torch.cuda.empty_cache()

        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            print(error_msg)
            results.append(error_msg)

    # Footer
    footer = f"\n{'=' * 80}\nBenchmark completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 80}\n"
    print(footer)
    results.append(footer)

    # Save results to file
    with open(output_file, "w") as f:
        f.write("\n".join(results))

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
