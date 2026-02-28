"""
vLLM Throughput Benchmark
Includes warmup phase to stabilize GPU performance before measurement
"""

import time
import torch
from datetime import datetime
from vllm import LLM, SamplingParams


# Benchmark Configuration
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
ATTENTION_BACKENDS = ["FLASHINFER", "FLASH_ATTN"]
TP_SIZES = [1]
MODELS = [
    {"name": "Qwen/Qwen3-0.6B"},
    {"name": "Qwen/Qwen3-4B"},
    {"name": "Qwen/Qwen3-4B-AWQ"},
    {"name": "Qwen/Qwen3-8B"},
    {"name": "Qwen/Qwen3-8B-AWQ"},
]

# Warmup configuration
WARMUP_SEQS = 32
WARMUP_TOKENS = 128


def run_benchmark(llm, num_seqs, max_input_len, max_output_len):
    """Run a single benchmark with given parameters."""
    sampling_params = SamplingParams(
        max_tokens=max_output_len,
        temperature=0.6,
        top_p=0.95,
    )
    
    dummy_prompt_token_ids = [100] * max_input_len
    prompts = [{"prompt_token_ids": dummy_prompt_token_ids} for _ in range(num_seqs)]
    
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_generated_tokens / total_time
    
    return {
        "num_seqs": num_seqs,
        "total_tokens": total_generated_tokens,
        "time": total_time,
        "throughput": throughput
    }


def warmup(llm, max_input_len):
    """Run warmup to stabilize GPU performance."""
    print(f"Running warmup with {WARMUP_SEQS} sequences...")
    warmup_sampling_params = SamplingParams(
        max_tokens=WARMUP_TOKENS,
        temperature=0.6,
        top_p=0.95,
        presence_penalty=1.5,
    )
    warmup_prompt_token_ids = [100] * max_input_len
    warmup_prompts = [{"prompt_token_ids": warmup_prompt_token_ids} for _ in range(WARMUP_SEQS)]
    
    warmup_start = time.perf_counter()
    _ = llm.generate(warmup_prompts, warmup_sampling_params)
    warmup_end = time.perf_counter()
    print(f"Warmup completed in {warmup_end - warmup_start:.2f} seconds\n")


def format_comparison_row(batch_size, lhs_result, rhs_result):
    if lhs_result is None or rhs_result is None:
        return f"{batch_size:>12} | {'N/A':>14} | {'N/A':>14} | {'N/A':>12}"

    lhs_throughput = lhs_result["throughput"]
    rhs_throughput = rhs_result["throughput"]
    ratio = lhs_throughput / rhs_throughput if rhs_throughput else float("inf")
    return (
        f"{batch_size:>12} | {lhs_throughput:>14.2f} | {rhs_throughput:>14.2f} | "
        f"{ratio:>12.2f}"
    )


def main():
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_vllm_results_{timestamp}.txt"
    
    results = []
    
    # Header
    header = f"""
================================================================================
                      VLLM OFFLINE BENCHMARK RESULTS
================================================================================
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Max Input Length: {MAX_INPUT_LEN}
Max Output Length: {MAX_OUTPUT_LEN}
Batch Sizes: {BATCH_SIZES}
Attention Backends: {ATTENTION_BACKENDS}
TP Sizes: {TP_SIZES}
Models: {[m['name'] for m in MODELS]}
================================================================================
"""
    print(header)
    results.append(header)
    
    for model_config in MODELS:
        model_name = model_config["name"]
        model_results = {}
        
        model_header = f"\n{'='*80}\nModel: {model_name}\n{'='*80}\n"
        print(model_header)
        results.append(model_header)
        
        for tp_size in TP_SIZES:
            tp_header = f"\n{'-' * 80}\nTensor Parallel Size: {tp_size}\n{'-' * 80}"
            print(tp_header)
            results.append(tp_header)

            for attention_backend in ATTENTION_BACKENDS:
                combo_results = {}
                combo_key = (attention_backend, tp_size)
                combo_header = (
                    f"\nAttention Backend: {attention_backend} | TP Size: {tp_size}\n"
                    f"{'-' * 80}"
                )
                print(combo_header)
                results.append(combo_header)

                llm = None
                try:
                    print("Initializing model...")
                    llm = LLM(
                        model=model_name,
                        attention_backend=attention_backend,
                        tensor_parallel_size=tp_size,
                        max_model_len=4096,
                        max_num_seqs=max(BATCH_SIZES),
                        enable_chunked_prefill=True,
                        trust_remote_code=True,
                        gpu_memory_utilization=0.9,
                    )

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
                            combo_results[batch_size] = result
                            row = (
                                f"{result['num_seqs']:>12} | {result['total_tokens']:>14} | "
                                f"{result['time']:>10.2f} | {result['throughput']:>20.2f}"
                            )
                            print(row)
                            results.append(row)
                        except Exception as e:
                            error_row = f"{batch_size:>12} | ERROR: {str(e)[:50]}"
                            print(error_row)
                            results.append(error_row)
                except Exception as e:
                    error_msg = (
                        "Failed to load model "
                        f"{model_name} with backend {attention_backend} and tp={tp_size}: {str(e)}"
                    )
                    print(error_msg)
                    results.append(error_msg)
                finally:
                    if llm is not None:
                        del llm
                        torch.cuda.empty_cache()

                model_results[combo_key] = combo_results

        for tp_size in TP_SIZES:
            comparison_header = (
                f"\n{'-' * 80}\nBackend Comparison at TP={tp_size} (tok/s)\n{'-' * 80}"
            )
            comparison_table_header = (
                f"{'Batch Size':>12} | {'FLASHINFER':>14} | {'FLASH_ATTN':>14} | {'FI/FA Ratio':>12}"
            )
            comparison_separator = "-" * len(comparison_table_header)
            print(comparison_header)
            print(comparison_table_header)
            print(comparison_separator)
            results.append(comparison_header)
            results.append(comparison_table_header)
            results.append(comparison_separator)

            for batch_size in BATCH_SIZES:
                comparison_row = format_comparison_row(
                    batch_size,
                    model_results.get(("FLASHINFER", tp_size), {}).get(batch_size),
                    model_results.get(("FLASH_ATTN", tp_size), {}).get(batch_size),
                )
                print(comparison_row)
                results.append(comparison_row)

        for attention_backend in ATTENTION_BACKENDS:
            comparison_header = (
                f"\n{'-' * 80}\nTP Comparison for {attention_backend} (tok/s)\n{'-' * 80}"
            )
            comparison_table_header = (
                f"{'Batch Size':>12} | {'TP=1':>14} | {'TP=2':>14} | {'TP2/TP1 Ratio':>12}"
            )
            comparison_separator = "-" * len(comparison_table_header)
            print(comparison_header)
            print(comparison_table_header)
            print(comparison_separator)
            results.append(comparison_header)
            results.append(comparison_table_header)
            results.append(comparison_separator)

            for batch_size in BATCH_SIZES:
                comparison_row = format_comparison_row(
                    batch_size,
                    model_results.get((attention_backend, 2), {}).get(batch_size),
                    model_results.get((attention_backend, 1), {}).get(batch_size),
                )
                print(comparison_row)
                results.append(comparison_row)
    
    # Footer
    footer = f"\n{'='*80}\nBenchmark completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n"
    print(footer)
    results.append(footer)
    
    # Save results to file
    with open(output_file, "w") as f:
        f.write("\n".join(results))
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
