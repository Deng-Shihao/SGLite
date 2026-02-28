"""Offline throughput benchmark for SGLite with optional tensor parallelism."""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import queue
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime

import torch
from transformers import AutoTokenizer

from sglite.core import SamplingParams
from sglite.distributed import DistributedInfo
from sglite.message import BaseFrontendMsg, BaseTokenizerMsg, TokenizeMsg
from sglite.server.api_server import FrontendManager
from sglite.server.args import ServerArgs
from sglite.tokenizer import tokenize_worker
from sglite.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue


MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024
WARMUP_SEQS = 5
WARMUP_TOKENS = 128

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
TP_SIZES = [1]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-index",
        type=int,
        default=None,
        help="Run only a single model entry from MODELS inside the current process.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Run only a single tensor parallel size from TP_SIZES inside the current process.",
    )
    return parser.parse_args()


def resolve_dtype(model_dtype: torch.dtype | None) -> torch.dtype:
    return torch.bfloat16 if model_dtype is None else model_dtype


def generate_prompt(tokenizer, num_tokens: int, *, seed: int = 0) -> str:
    rng = random.Random(seed)
    vocab_size = max(1, tokenizer.vocab_size // 2)
    token_ids = [rng.randint(0, vocab_size) for _ in range(num_tokens - 1)]

    for _ in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == num_tokens:
            return prompt
        if len(token_ids) < num_tokens:
            need = num_tokens - len(token_ids)
            token_ids.extend(rng.randint(0, vocab_size) for _ in range(need))
        else:
            token_ids = token_ids[:num_tokens]

    raise ValueError("Failed to generate a prompt of the desired length.")


def build_prompts(prompt: str, num_seqs: int) -> list[str]:
    return [prompt] * num_seqs


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
TP Sizes: {TP_SIZES}
Models: {[m["name"] for m in MODELS]}
================================================================================
"""


def format_table_header() -> tuple[str, str]:
    table_header = (
        f"{'Batch Size':>12} | {'Total Tokens':>14} | {'Time (s)':>10} | "
        f"{'Throughput (tok/s)':>20}"
    )
    return table_header, "-" * len(table_header)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def create_server_args(
    model_name: str,
    model_dtype: torch.dtype | None,
    tp_size: int,
) -> ServerArgs:
    return ServerArgs(
        model_path=model_name,
        tp_info=DistributedInfo(0, tp_size),
        dtype=resolve_dtype(model_dtype),
        server_port=find_free_port(),
        max_running_req=max(BATCH_SIZES),
        max_seq_len_override=4096,
        max_extend_tokens=16384,
        cuda_graph_max_bs=256,
        num_tokenizer=0,
        silent_output=True,
    )


def create_frontend(config: ServerArgs) -> FrontendManager:
    return FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )


def _run_scheduler_worker(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from sglite.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            scheduler.shutdown()


def start_backend_processes(config: ServerArgs) -> tuple[list[mp.Process], mp.Queue[str]]:
    mp.set_start_method("spawn", force=True)
    ack_queue: mp.Queue[str] = mp.Queue()
    processes: list[mp.Process] = []
    world_size = config.tp_info.size

    for rank in range(world_size):
        process = mp.Process(
            target=_run_scheduler_worker,
            args=(replace(config, tp_info=DistributedInfo(rank, world_size)), ack_queue),
            daemon=False,
            name=f"sglite-TP{rank}-scheduler",
        )
        process.start()
        processes.append(process)

    process = mp.Process(
        target=tokenize_worker,
        kwargs={
            "tokenizer_path": config.model_path,
            "addr": config.zmq_detokenizer_addr,
            "backend_addr": config.zmq_backend_addr,
            "frontend_addr": config.zmq_frontend_addr,
            "local_bs": 1,
            "create": config.tokenizer_create_addr,
            "tokenizer_id": 0,
            "ack_queue": ack_queue,
        },
        daemon=False,
        name="sglite-tokenizer-0",
    )
    process.start()
    processes.append(process)

    expected_acks = config.num_tokenizer + 2
    for _ in range(expected_acks):
        ack_queue.get(timeout=120)

    return processes, ack_queue


def stop_backend_processes(processes: list[mp.Process], ack_queue: mp.Queue[str] | None) -> None:
    if ack_queue is not None:
        ack_queue.close()
        ack_queue.join_thread()

    for process in processes:
        if process.is_alive():
            process.kill()

    for process in processes:
        process.join(timeout=5)


async def _send_batch(
    frontend: FrontendManager,
    prompts: list[str],
    sampling_params: list[SamplingParams],
) -> list[int]:
    async def wait_for_completion(uid: int) -> int:
        final_ack = None
        async for ack in frontend.wait_for_ack(uid):
            final_ack = ack
        assert final_ack is not None
        return final_ack.num_output_tokens

    pending_uids: list[int] = []
    for prompt, sampling_param in zip(prompts, sampling_params, strict=True):
        uid = frontend.new_user()
        pending_uids.append(uid)
        await frontend.send_one(
            TokenizeMsg(
                uid=uid,
                text=prompt,
                sampling_params=sampling_param,
            )
        )

    return await asyncio.gather(*(wait_for_completion(uid) for uid in pending_uids))


async def warmup(frontend: FrontendManager, prompt: str) -> None:
    print(f"Running warmup with {WARMUP_SEQS} sequences...")
    prompts = build_prompts(prompt, WARMUP_SEQS)
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
    await _send_batch(frontend, prompts, sampling_params)
    elapsed = time.perf_counter() - start_time
    print(f"Warmup completed in {elapsed:.2f} seconds\n")


async def run_benchmark(
    frontend: FrontendManager,
    prompt: str,
    num_seqs: int,
    max_output_len: int,
) -> BenchmarkResult:
    prompts = build_prompts(prompt, num_seqs)
    sampling_params = build_sampling_params(num_seqs, max_output_len)

    start_time = time.perf_counter()
    output_lengths = await _send_batch(frontend, prompts, sampling_params)
    total_time = time.perf_counter() - start_time

    total_generated_tokens = sum(output_lengths)
    throughput = total_generated_tokens / total_time

    return BenchmarkResult(
        num_seqs=num_seqs,
        total_tokens=total_generated_tokens,
        time_s=total_time,
        throughput=throughput,
    )


async def run_model_async(
    frontend: FrontendManager,
    prompt: str,
) -> list[BenchmarkResult | str]:
    await warmup(frontend, prompt)

    results: list[BenchmarkResult | str] = []
    for batch_size in BATCH_SIZES:
        try:
            results.append(await run_benchmark(frontend, prompt, batch_size, MAX_OUTPUT_LEN))
        except Exception as exc:
            results.append(f"{batch_size:>12} | ERROR: {str(exc)[:50]}")
    return results


def run_model(model_config: dict[str, str | torch.dtype | None], tp_size: int) -> list[str]:
    model_name = str(model_config["name"])
    model_dtype = model_config["dtype"]
    lines: list[str] = []
    frontend: FrontendManager | None = None
    processes: list[mp.Process] = []
    ack_queue: mp.Queue[str] | None = None

    model_header = (
        f"\n{'=' * 80}\nModel: {model_name}\nTensor Parallel Size: {tp_size}\n{'=' * 80}\n"
    )
    print(model_header)
    lines.append(model_header)

    try:
        config = create_server_args(model_name, model_dtype, tp_size)
        frontend = create_frontend(config)
        processes, ack_queue = start_backend_processes(config)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        prompt = generate_prompt(tokenizer, MAX_INPUT_LEN)

        table_header, separator = format_table_header()
        run_results = asyncio.run(run_model_async(frontend, prompt))

        print(table_header)
        print(separator)
        lines.append(table_header)
        lines.append(separator)

        for result in run_results:
            if isinstance(result, str):
                print(result)
                lines.append(result)
                continue

            row = (
                f"{result.num_seqs:>12} | {result.total_tokens:>14} | "
                f"{result.time_s:>10.2f} | {result.throughput:>20.2f}"
            )
            print(row)
            lines.append(row)
    except queue.Empty:
        error_msg = f"Failed to load model {model_name} with tp={tp_size}: backend startup timed out"
        print(error_msg)
        lines.append(error_msg)
    except Exception as exc:
        error_msg = f"Failed to load model {model_name} with tp={tp_size}: {exc}"
        print(error_msg)
        lines.append(error_msg)
    finally:
        if frontend is not None:
            frontend.shutdown()
        stop_backend_processes(processes, ack_queue)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return lines


def stream_model_subprocess(model_index: int, tp_size: int) -> tuple[int, list[str]]:
    command = [
        sys.executable,
        __file__,
        "--model-index",
        str(model_index),
        "--tp-size",
        str(tp_size),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    output_lines: list[str] = []
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line.rstrip("\n"))

    return_code = process.wait()
    return return_code, output_lines


def main() -> None:
    args = parse_args()

    if args.model_index is not None or args.tp_size is not None:
        if args.model_index is None or args.tp_size is None:
            raise SystemExit("--model-index and --tp-size must be provided together")
        if not 0 <= args.model_index < len(MODELS):
            raise SystemExit(f"model-index must be in [0, {len(MODELS) - 1}]")
        if args.tp_size not in TP_SIZES:
            raise SystemExit(f"tp-size must be one of {TP_SIZES}")
        run_model(MODELS[args.model_index], args.tp_size)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.txt"
    results: list[str] = []

    header = format_header()
    print(header)
    results.append(header)

    for model_index, model_config in enumerate(MODELS):
        for tp_size in TP_SIZES:
            return_code, output_lines = stream_model_subprocess(model_index, tp_size)
            results.extend(output_lines)
            if return_code != 0:
                failure_line = (
                    f"Model subprocess failed for {model_config['name']} with tp={tp_size} "
                    f"and exit code {return_code}"
                )
                print(failure_line)
                results.append(failure_line)

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
