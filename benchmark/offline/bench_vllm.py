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
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MODELS = [
    {"name": "Qwen/Qwen3-0.6B"},
    {"name": "Qwen/Qwen3-4B"},
    {"name": "Qwen/Qwen3-4B-AWQ"},
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
        presence_penalty=1.5,
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
Models: {[m['name'] for m in MODELS]}
================================================================================
"""
    print(header)
    results.append(header)
    
    for model_config in MODELS:
        model_name = model_config["name"]
        
        model_header = f"\n{'='*80}\nModel: {model_name}\n{'='*80}\n"
        print(model_header)
        results.append(model_header)
        
        try:
            print("Initializing model...")
            llm = LLM(
                model=model_name,
                max_model_len=4096,
                max_num_seqs=max(BATCH_SIZES),
                enable_chunked_prefill=True,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
            )
            
            # Warmup
            warmup(llm, MAX_INPUT_LEN)
            
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
            del llm
            torch.cuda.empty_cache()
            
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            print(error_msg)
            results.append(error_msg)
    
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
