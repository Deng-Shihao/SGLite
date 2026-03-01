# SGLite Parameters

This document summarizes the parameters implemented in the current codebase. It covers:

- Command-line startup arguments
- Runtime environment variables
- API request parameters
- Effective context length rules

Note: some examples in `README.md` still use older flag names such as `--model`, `--shell`, `--tp`, and `--cache`. The current implementation uses `--model-path`, `--shell-mode`, `--tp-size`, and `--cache-type`.

## Command-Line Arguments

These arguments are defined in `python/sglite/server/args.py`.

| Argument | Alias | Default | Description |
|---|---|---|---|
| `--model-path` | none | required | Model path. Supports a local directory or a Hugging Face repo ID. |
| `--dtype` | none | `auto` | Weight and activation dtype. Choices: `auto`, `float16`, `bfloat16`, `float32`. `auto` infers from the HF config. |
| `--tensor-parallel-size` | `--tp-size` | `1` | Tensor parallel size. |
| `--max-running-requests` | none | `256` | Maximum number of concurrently running requests. |
| `--max-seq-len-override` | none | `None` | Overrides the model max sequence length. The effective value is still capped by KV cache capacity. |
| `--memory-ratio` | none | `0.9` | Fraction of GPU memory reserved for KV cache. |
| `--dummy-weight` | none | `False` | Uses dummy weights for testing. |
| `--disable-pynccl` | none | `False` | Disables PyNCCL. PyNCCL is enabled by default. |
| `--host` | none | `127.0.0.1` | Server bind address. |
| `--port` | none | `1919` | Server port. The distributed address uses `port + 1`. |
| `--cuda-graph-max-bs` | `--graph` | `None` | Maximum batch size captured by CUDA Graph. `None` means automatic selection. The current implementation auto-selects `256` on SM90/H200-class GPUs and `160` otherwise. |
| `--num-tokenizer` | `--tokenizer-count` | `0` | Number of tokenizer worker processes. `0` means the tokenizer is shared with the detokenizer. |
| `--max-prefill-length` | `--max-extend-length` | `8192` | Maximum chunk size, in tokens, for chunked prefill. |
| `--num-pages` | `--num-tokens` | `None` | Overrides the maximum number of KV cache pages. With the current `page_size=1`, this is close to a token-capacity limit. |
| `--attention-backend` | `--attn` | `auto` | Attention backend. Supported values are `fi`, `fa`, or a hybrid pair such as `fa,fi`. With `auto`, the current implementation selects `fi` on Blackwell and pre-Hopper GPUs, and `fa,fi` on Hopper. |
| `--cache-type` | none | `radix` | KV cache management policy. Supported values are `radix` and `naive`. |
| `--shell-mode` | none | `False` | Runs the server in interactive shell mode. |
| `--shell-bench` | none | `False` | Shows shell-side TTFT and generation speed. Enabling it also enables shell mode. |

## Shell Mode Behavior

When shell mode is enabled, the launcher forces:

| Setting | Forced Value |
|---|---|
| `cuda_graph_max_bs` | `1` |
| `max_running_req` | `1` |
| `silent_output` | `True` |

The shell also stores chat history in memory and sends the whole conversation back on each turn. Use `/reset` to clear that history.

`--shell-bench` also enables shell mode automatically.

## Environment Variables

These variables are defined in `python/sglite/env.py`. All use the `SGLITE_` prefix.

Boolean environment variables are parsed from strings. The values `1`, `true`, and `yes` are treated as `True`; any other provided value is treated as `False`.

| Environment Variable | Default | Description |
|---|---|---|
| `SGLITE_SHELL_MAX_TOKENS` | `2048` | Maximum generated tokens per shell reply. |
| `SGLITE_SHELL_TOP_K` | `-1` | Shell sampling `top_k`. `-1` behaves like no top-k limit. |
| `SGLITE_SHELL_TOP_P` | `1.0` | Shell sampling `top_p`. |
| `SGLITE_SHELL_TEMPERATURE` | `0.6` | Shell sampling temperature. |
| `SGLITE_FLASHINFER_USE_TENSOR_CORES` | `None` | Optional FlashInfer Tensor Core toggle. If unset, backend defaults apply. |
| `SGLITE_DISABLE_OVERLAP_SCHEDULING` | `True` | Disables overlap scheduling. This is currently off by default unless you explicitly set `SGLITE_DISABLE_OVERLAP_SCHEDULING=0`/`false`/`no`. |
| `SGLITE_OVERLAP_EXTRA_SYNC` | `False` | Enables extra synchronization in overlap scheduling code paths. |
| `SGLITE_PYNCCL_MAX_BUFFER_SIZE` | `1G` | Maximum PyNCCL buffer size. Supports `K`, `M`, and `G` suffixes. |

## API Request Parameters

### `/generate`

| Field | Default | Description |
|---|---|---|
| `prompt` | required | Input text. |
| `max_tokens` | required | Maximum generated tokens. |
| `ignore_eos` | `False` | Ignores EOS if set. |

### `/v1/chat/completions`

Only part of the OpenAI-style schema is currently wired into inference. The endpoint always returns a streaming `text/event-stream` response regardless of the incoming `stream` flag.

| Field | Default | Used by current implementation | Description |
|---|---|---|---|
| `model` | required | no | Compatibility field. The current server does not use it for model selection. |
| `prompt` | `None` | yes | String prompt for non-chat style usage. |
| `messages` | `None` | yes | Chat messages. |
| `max_tokens` | `16` | yes | Maximum generated tokens. |
| `temperature` | `1.0` | yes | Sampling temperature. |
| `top_k` | `-1` | yes | Top-k sampling. |
| `top_p` | `1.0` | yes | Top-p sampling. |
| `n` | `1` | no | Multi-candidate generation is not implemented. |
| `stream` | `False` | no | Parsed for compatibility, but the current endpoint always streams. |
| `stop` | `[]` | no | Stop sequence handling is not implemented. |
| `presence_penalty` | `0.0` | no | Not implemented. |
| `frequency_penalty` | `0.0` | no | Not implemented. |
| `ignore_eos` | `False` | yes | Ignores EOS if set. |

## Core Sampling Parameters

The internal `SamplingParams` object uses these defaults before API or shell values override them:

| Field | Default | Description |
|---|---|---|
| `temperature` | `0.0` | Greedy-like when `<= 0.0`. |
| `top_k` | `-1` | No top-k restriction. |
| `top_p` | `1.0` | No top-p restriction. |
| `ignore_eos` | `False` | Whether EOS is ignored. |
| `max_tokens` | `1024` | Maximum generated tokens. |

## Effective Context Length

The project distinguishes between configured max sequence length and effective runtime context length.

### Configured max sequence length

- If `--max-seq-len-override` is set, that value is used.
- Otherwise, SGLite uses `max_position_embeddings` from the Hugging Face model config.
- The current implementation does not use `tokenizer.model_max_length` or `model_max_length` as the runtime context limit.

### Effective runtime context length

The actual runtime limit is:

```text
align_up_32(min(config.max_seq_len, num_pages))
```

This means the usable context length is bounded by both:

- the configured model sequence length
- the number of KV cache pages available

With the current implementation, requests are validated against this effective limit. If the prompt is too long, the request is dropped. If the prompt fits but leaves too little room for output, `max_tokens` is reduced automatically.

## Examples

Run a server:

```bash
python -m sglite --model-path "Qwen/Qwen3-0.6B"
```

Run shell mode with a custom context override:

```bash
python -m sglite --model-path "Qwen/Qwen3-0.6B" --shell-mode --max-seq-len-override 32768
```

Run shell mode with custom generation length:

```bash
SGLITE_SHELL_MAX_TOKENS=1024 python -m sglite --model-path "Qwen/Qwen3-0.6B" --shell-mode
```

Enable overlap scheduling explicitly:

```bash
SGLITE_DISABLE_OVERLAP_SCHEDULING=0 python -m sglite --model-path "Qwen/Qwen3-0.6B"
```
