# SGLite

SGLite is a lightweight LLM inference framework focused on readability and practical serving performance. It is a compact re-implementation of core ideas from [SGLang](https://github.com/sgl-project/sglang), intended both as a usable engine and as a codebase that is small enough to study.

The current implementation exposes:

- An OpenAI-compatible server endpoint at `/v1/chat/completions`
- A simple streaming endpoint at `/generate`
- An interactive terminal chat mode
- Multi-GPU tensor parallel serving
- KV cache management with `radix` and `naive` modes
- Chunked prefill, CUDA graph capture, and mixed attention backends

## Supported Models

The current implementation supports:

- Dense `Qwen3` models
- Dense `Llama` models
- AWQ-quantized variants of supported model families

For AWQ models, SGLite detects quantization metadata from the Hugging Face config or local quantization config files. On compatible SM80+ GPUs, it will try to use AWQ Marlin automatically and fall back to AWQ Triton when needed.

## Status

This repository is still early-stage. The public surface is usable, but some areas are intentionally narrow:

- Supported model families in the current codebase: `Qwen3` and `Llama`
- Quantized inference support includes AWQ models
- The OpenAI-style API only implements a subset of request fields
- The chat endpoint currently returns a streaming response even if `stream=false`

For the implemented flags and API fields, see [docs/parameters.md](./docs/parameters.md).

## Requirements

- Linux
- Python 3.10+
- NVIDIA GPU with CUDA available
- CUDA toolkit installed for JIT-compiled kernels

This project is not intended for native Windows or macOS execution. If needed, use WSL2 or a Linux container with GPU passthrough.

## Installation

`uv` is the expected package manager in this repository.

```bash
git clone https://github.com/sgl-project/sglite.git
cd sglite

uv venv --python=3.12
source .venv/bin/activate

uv pip install -e .
```

Install dev tools if you want to run tests and local checks:

```bash
uv pip install -e ".[dev]"
```

## Quick Start

### Run the API server

Single GPU:

```bash
python -m sglite --model-path "Qwen/Qwen3-0.6B"
```

Single GPU with an AWQ model:

```bash
python -m sglite --model-path "Qwen/Qwen3-4B-AWQ"
```

Tensor parallel across 4 GPUs:

```bash
python -m sglite \
  --model-path "meta-llama/Llama-3.1-70B-Instruct" \
  --tp-size 4 \
  --port 30000
```

Useful options:

- `--dtype auto|float16|bfloat16|float32`
- `--cache-type radix|naive`
- `--attention-backend auto|fi|fa|fa,fi`
- `--max-seq-len-override <int>`
- `--memory-ratio <float>`

Inspect all implemented flags with:

```bash
python -m sglite --help
```

### Run interactive CLI mode

```bash
python -m sglite --model-path "Qwen/Qwen3-0.6B" --cli
```

CLI mode with an AWQ model:

```bash
python -m sglite --model-path "Qwen/Qwen3-4B-AWQ" --cli
```

CLI benchmark display:

```bash
python -m sglite --model-path "Qwen/Qwen3-8B" --cli-bench
```

CLI commands:

- `/clear` clears the current conversation history
- `/exit` exits the shell

### Example API calls

Health check:

```bash
curl http://127.0.0.1:1919/v1
```

List the loaded model:

```bash
curl http://127.0.0.1:1919/v1/models
```

OpenAI-compatible chat completion:

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Write a haiku about CUDA graphs."}
    ],
    "max_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 20
  }'
```

Simple generate endpoint:

```bash
curl http://127.0.0.1:1919/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain radix cache in one paragraph.",
    "max_tokens": 128,
    "ignore_eos": false
  }'
```

## Benchmarks

Offline benchmark:

```bash
python benchmark/offline/bench.py
```

Online benchmark against a running SGLite server:

```bash
python benchmark/online/bench_sglite.py
```

Comparison script for a vLLM server:

```bash
python benchmark/online/bench_vllm.py
```

For overlap-scheduling ablation, explicitly disable it:

```bash
SGLITE_DISABLE_OVERLAP_SCHEDULING=1 python benchmark/offline/bench.py
```

## Development

Run tests:

```bash
uv run pytest
```

The repository layout is documented here:

- [docs/features.md](./docs/features.md)
- [docs/parameters.md](./docs/parameters.md)
- [docs/structures.md](./docs/structures.md)

## Notes on Current Behavior

- The server binds to `127.0.0.1:1919` by default.
- The distributed control address uses `port + 1`.
- In CLI mode, SGLite forces single-request execution and disables normal server-side output noise.
- The effective runtime context length is capped by both model config and KV cache capacity.
- `SGLITE_DISABLE_OVERLAP_SCHEDULING` is currently opt-out by default in the implementation unless you explicitly set it to `0`, `false`, or `no`.

## License

MIT. See [LICENSE](./LICENSE).
