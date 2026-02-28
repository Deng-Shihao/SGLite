"""Offline throughput benchmark for SGLite."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import torch

from sglite.core import SamplingParams
from sglite.llm import LLM


MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024
PROMPT_TOKEN_ID = 100
WARMUP_SEQS = 32
WARMUP_TOKENS = 128

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
MODELS = [
    {"name": "Qwen/Qwen3-0.6B", "dtype": None},
    {"name": "Qwen/Qwen3-4B", "dtype": None},
    {"name": "Qwen/Qwen3-4B-AWQ", "dtype": torch.float16},
    {"name": "Qwen/Qwen3-8B", "dtype": None},
    {"name": "Qwen/Qwen3-8B-AWQ", "dtype": torch.float16},
]


@dataclass
class BenchmarkResult:
    num_seqs: int
    total_tokens: int
    time_s: float
    throughput: float


def build_prompts(num_seqs: int, input_len: int) -> list[list[int]]:
    prompt_token_ids = [PROMPT_TOKEN_ID] * input_len
    return [prompt_token_ids.copy() for _ in range(num_seqs)]


def build_sampling_params(num_seqs: int, max_output_len: int) -> list[SamplingParams]:
    return [
        SamplingParams(
            max_tokens=max_output_len,
            temperature=0.6,
            top_p=0.95,
            ignore_eos=True,
        )
        for _ in range(num_seqs)
    ]


def warmup(llm: LLM, max_input_len: int) -> None:
    print(f"Running warmup with {WARMUP_SEQS} sequences...")
    prompts = build_prompts(WARMUP_SEQS, max_input_len)
    sampling_params = [
        SamplingParams(
            max_tokens=WARMUP_TOKENS,
            temperature=0.6,
            top_p=0.95,
            ignore_eos=True,
        )
        for _ in range(WARMUP_SEQS)
    ]

    start_time = time.perf_counter()
    llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start_time
    print(f"Warmup completed in {elapsed:.2f} seconds\n")


def run_benchmark(llm: LLM, num_seqs: int, max_input_len: int, max_output_len: int) -> BenchmarkResult:
    prompts = build_prompts(num_seqs, max_input_len)
    sampling_params = build_sampling_params(num_seqs, max_output_len)

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.perf_counter() - start_time

    total_generated_tokens = sum(len(output["token_ids"]) for output in outputs)
    throughput = total_generated_tokens / total_time

    return BenchmarkResult(
        num_seqs=num_seqs,
        total_tokens=total_generated_tokens,
        time_s=total_time,
        throughput=throughput,
    )


def create_llm(model_name: str, model_dtype: torch.dtype | None) -> LLM:
    kwargs = {
        "max_seq_len_override": 4096,
        "max_extend_tokens": 16384,
        "cuda_graph_max_bs": 256,
    }
    if model_dtype is not None:
        kwargs["dtype"] = model_dtype
    return LLM(model_name, **kwargs)


def format_header() -> str:
    return f"""
================================================================================
                         OFFLINE BENCHMARK RESULTS
================================================================================
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Max Input Length: {MAX_INPUT_LEN}
Max Output Length: {MAX_OUTPUT_LEN}
Warmup: {WARMUP_SEQS} seqs x {WARMUP_TOKENS} tokens
Batch Sizes: {BATCH_SIZES}
Models: {[m["name"] for m in MODELS]}
================================================================================
"""


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.txt"
    results: list[str] = []

    header = format_header()
    print(header)
    results.append(header)

    for model_config in MODELS:
        model_name = model_config["name"]
        model_dtype = model_config["dtype"]
        llm: LLM | None = None

        model_header = f"\n{'=' * 80}\nModel: {model_name}\n{'=' * 80}\n"
        print(model_header)
        results.append(model_header)

        try:
            llm = create_llm(model_name, model_dtype)
            warmup(llm, MAX_INPUT_LEN)

            table_header = (
                f"{'Batch Size':>12} | {'Total Tokens':>14} | {'Time (s)':>10} | "
                f"{'Throughput (tok/s)':>20}"
            )
            separator = "-" * len(table_header)
            print(table_header)
            print(separator)
            results.append(table_header)
            results.append(separator)

            for batch_size in BATCH_SIZES:
                try:
                    result = run_benchmark(llm, batch_size, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
                    row = (
                        f"{result.num_seqs:>12} | {result.total_tokens:>14} | "
                        f"{result.time_s:>10.2f} | {result.throughput:>20.2f}"
                    )
                    print(row)
                    results.append(row)
                except Exception as exc:
                    error_row = f"{batch_size:>12} | ERROR: {str(exc)[:50]}"
                    print(error_row)
                    results.append(error_row)
        except Exception as exc:
            error_msg = f"Failed to load model {model_name}: {exc}"
            print(error_msg)
            results.append(error_msg)
        finally:
            if llm is not None:
                llm.shutdown()
                del llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    footer = (
        f"\n{'=' * 80}\nBenchmark completed at: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 80}\n"
    )
    print(footer)
    results.append(footer)

    with open(output_file, "w") as file:
        file.write("\n".join(results))

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
