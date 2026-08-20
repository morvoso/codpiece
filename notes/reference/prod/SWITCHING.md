# Engine switching — llm-host (192.168.134.60)

Two engines serve **the same port 8020** and **the same model id
`qwen3.8-27b`**, so no client config (qwen CLI / pi / omp) ever changes.
Only one can run at a time: each needs ~46 GiB of the 48 GiB of VRAM.

    cd /home/morvoso/llm
    ./switch.sh vllm       # fast      (currently deployed)
    ./switch.sh llamacpp   # accurate  (previous config, preserved exactly)
    ./switch.sh status     # what is running, plus GPU state

## The two configs

|                     | vLLM (deployed)         | llama.cpp (fallback)        |
|---------------------|-------------------------|-----------------------------|
| file                | docker-compose.vllm.yml | docker-compose.llamacpp.yml |
| weights             | Qwen/Qwen3.8-27B-FP8    | Unsloth UD-Q8_K_XL GGUF     |
| KV cache            | fp8                     | **f16 (higher fidelity)**   |
| context             | 196608                  | 196608                      |
| prose / code decode | **72 / 95 tok/s**       | 51 / 67 tok/s               |
| prefill (27K)       | **1612 tok/s**          | 1381 tok/s                  |
| 4-way concurrent    | **260 tok/s aggregate** | 73 tok/s                    |
| vision / tools      | yes / yes               | yes / yes                   |
| thinking field      | `reasoning`             | `reasoning_content`         |

Both measured 2026-08-16 with the same client-side harness, thinking off,
GPU0 power cap removed.

## When to switch back to llama.cpp

* You want **f16 KV** rather than fp8. That is the only remaining fidelity
  difference between the two — everything else (weights at 8-bit, context,
  vision, tools, reasoning) is equivalent.
* A client turns up that only understands `reasoning_content`. vLLM emits
  `reasoning`, and vLLM's own source calls `reasoning_content` deprecated.
  Verified against vLLM: pi 0.84.2 works, omp 17.3.4 works. The qwen CLI was
  NOT verified against vLLM — test it before relying on it there.
* vLLM misbehaves after an image update. llama.cpp is pinned to
  `server-cuda-b10423`; vLLM tracks `vllm/vllm-openai:latest`, which moves.
  Pin it with `VLLM_IMAGE=vllm/vllm-openai:v0.27.1` if that ever bites.

`./switch.sh llamacpp` restores the llama.cpp config **byte-for-byte** —
switching never edits the compose files, so nothing drifts over time.

## Do not lose these flags

Both stacks need `ipc: host` and `NCCL_P2P_DISABLE=1`. GeForce cards have
peer-to-peer DMA disabled in the driver, and Docker's default 64 MiB
`/dev/shm` starves NCCL's shared-memory transport — without both,
tensor-parallel aborts at startup with a generic CUDA error.

vLLM additionally needs `--max-num-seqs 4`. Its default of 256 allocates
about 256 Gated-DeltaNet recurrent-state slots, roughly 9.7 GiB per GPU on
this architecture, and OOMs before the KV cache is even allocated.

## Benchmarking gotcha

`scratchpad/engbench.py` measures both engines. Count real **tokens**, never
SSE chunks: llama.cpp emits exactly 1 token per stream event, but vLLM packs
2.5–3.1 (MTP emits all accepted tokens in one event). Counting chunks
understates vLLM by its acceptance length and produced a completely wrong
verdict once already.
