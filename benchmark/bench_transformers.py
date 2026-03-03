import argparse
import time
from threading import Thread
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


def build_prompt(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    enable_thinking: bool,
) -> str:
    """Prefer chat template if available; otherwise fall back to the last user message."""
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking:
            kwargs["enable_thinking"] = True
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)
    return messages[-1]["content"]


def get_device_config(device_arg: str) -> Tuple[torch.device, torch.dtype, str | None]:
    if device_arg == "cpu":
        return torch.device("cpu"), torch.float32, None
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA selected but torch.cuda.is_available() is False.")
        return torch.device("cuda"), torch.float16, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    device_map = "auto" if device.type == "cuda" else None
    return device, torch_dtype, device_map


def build_messages(
    history: Sequence[Tuple[str, str]],
    user_prompt: str,
    system_prompt: str | None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def get_input_device(model, fallback_device: torch.device) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            if mapped_device == "disk":
                continue
            if isinstance(mapped_device, torch.device):
                return mapped_device
            if isinstance(mapped_device, str):
                return torch.device(mapped_device)
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")

    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    if isinstance(model_device, str):
        return torch.device(model_device)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def print_benchmark(num_input_tokens: int, num_output_tokens: int, ttft: float, t_end: float, t_first_token: float) -> None:
    gen_duration = t_end - t_first_token
    if gen_duration > 0 and num_output_tokens > 1:
        decode_tokens = num_output_tokens - 1
        speed_tok_s = decode_tokens / gen_duration
        speed_ms_tok = 1000.0 / speed_tok_s
    else:
        speed_tok_s = 0.0
        speed_ms_tok = 0.0

    print("=" * 40)
    print("Speed of Inference")
    print("-" * 40)
    print(f"TTFT          : {ttft:.3f}s for {num_input_tokens} tokens")
    print(f"Generation    : {speed_tok_s:.1f} tok/s  ({speed_ms_tok:.1f} ms/tok)")
    print("=" * 40)


def run_generation(
    model,
    tokenizer,
    args,
    device: torch.device,
    device_map: str | None,
    messages: Sequence[Dict[str, str]],
) -> str:
    prompt_text = build_prompt(tokenizer, messages, enable_thinking=args.enable_thinking)
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    num_input_tokens = int(enc["input_ids"].shape[-1])
    input_device = get_input_device(model, device)
    enc = {k: v.to(input_device) for k, v in enc.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    gen_kwargs = {
        **enc,
        "max_new_tokens": args.max_new_tokens,
        "streamer": streamer,
        "return_dict_in_generate": True,
        "pad_token_id": pad_token_id,
    }
    if args.temperature > 0:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        gen_kwargs["do_sample"] = False

    result: Dict[str, object] = {}

    def generate_in_background() -> None:
        try:
            result["output"] = model.generate(**gen_kwargs)
        except Exception as exc:  # pragma: no cover - surfaced to caller immediately.
            result["error"] = exc

    t_start = time.perf_counter()
    thread = Thread(target=generate_in_background, daemon=True)
    thread.start()

    generated_text = ""
    t_first_token: float | None = None
    for piece in streamer:
        if t_first_token is None:
            t_first_token = time.perf_counter()
        generated_text += piece
        print(piece, end="", flush=True)

    thread.join()
    t_end = time.perf_counter()
    print("", flush=True)

    error = result.get("error")
    if isinstance(error, Exception):
        raise error

    output = result.get("output")
    sequences = getattr(output, "sequences", output)
    if sequences is None:
        num_output_tokens = 0
    else:
        num_output_tokens = max(int(sequences.shape[-1]) - num_input_tokens, 0)

    if t_first_token is not None:
        ttft = t_first_token - t_start
        print_benchmark(
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            ttft=ttft,
            t_end=t_end,
            t_first_token=t_first_token,
        )

    return generated_text


def interactive_shell(model, tokenizer, args, device: torch.device, device_map: str | None) -> None:
    history: List[Tuple[str, str]] = []
    while True:
        try:
            user_prompt = input(">>> ").strip()
        except EOFError:
            print("\nExiting shell...")
            return

        if not user_prompt:
            continue
        if user_prompt == "/exit":
            print("Exiting shell...")
            return
        if user_prompt == "/clear":
            history = []
            continue
        if user_prompt.startswith("/"):
            print(f"Unknown command: {user_prompt}")
            continue

        messages = build_messages(history, user_prompt, args.system_prompt)
        assistant_text = run_generation(
            model=model,
            tokenizer=tokenizer,
            args=args,
            device=device,
            device_map=device_map,
            messages=messages,
        )
        history.append((user_prompt, assistant_text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B", help="HF model id")
    parser.add_argument(
        "--prompt",
        default="Explain streaming inference in one paragraph.",
        help="User prompt for one-shot mode",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt used to build chat history",
    )
    parser.add_argument("--interactive", action="store_true", help="Run a CLI shell with /clear and /exit")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode if the tokenizer chat template supports it",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    args = parser.parse_args()

    device, torch_dtype, device_map = get_device_config(args.device)

    print(f"[info] model={args.model}")
    print(f"[info] device={device}, dtype={torch_dtype}, device_map={device_map}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    if args.interactive:
        interactive_shell(model, tokenizer, args, device, device_map)
        return

    messages = build_messages([], args.prompt, args.system_prompt)
    run_generation(
        model=model,
        tokenizer=tokenizer,
        args=args,
        device=device,
        device_map=device_map,
        messages=messages,
    )
    print("[done]")


if __name__ == "__main__":
    main()
