# SAFETY.md — hardware & production safety protocol for llm-host

llm-host (192.168.134.60) is a **production box**: it serves Qwen3.8-27B to
the user's daemons 24/7 (`llama-qwen` container, port 8020) plus searxng,
questboard, cloudflared. Both 3090s sit at ~95% VRAM while prod runs.
These rules are absolute for all tandem work. They exist because every one of
them was paid for in downtime (ENGINE.md §8).

## Hardware protection

1. **Never raise power limits.** GPU0 is capped at 260 W by
   `gpu-powerlimit.service` because its fan sits against GPU1's backplate.
   tandem never calls `nvidia-smi -pl`, never edits that service. (Raising to
   280–300 W is a *user-only* decision with temps watched.)
2. **Temperature watchdog during any GPU work we initiate.** Poll
   `nvidia-smi --query-gpu=temperature.gpu,power.draw` every 10 s; abort the
   workload at ≥ 83 °C GPU temp on either card. (3090 GDDR6X runs hot; the
   card self-throttles ~90 °C — we stay well under.)
3. Inference/compile workloads cannot electrically damage these cards under
   stock limits; the real risks are thermal dwell (rule 2) and operator error
   (everything below).

## Production protection

4. **Zero GPU allocations while prod serves.** Free VRAM is ~1.6 GiB total;
   even a bare CUDA context can OOM prod's flash-attn VMM pool mid-prefill.
   All GPU execution happens inside a **locked bench window**:
   - take `flock /tmp/llama-sweep.lock` (same lock the sweep harness uses)
   - check prod is idle via `curl :8020/slots` (refuse if busy)
   - `docker compose -f ~/llm/docker-compose.llamacpp.yml stop` prod
   - run, tee ALL output to a log file (never `/dev/null`, never `--rm`)
   - restore prod, verify `/health` + a 1-token completion, release lock
5. **CPU-only work (builds, parsing, tokenizers) is allowed alongside prod**
   but nice'd: `nice -n 19`, build jobs ≤ 8, and never within a benchmark rep.
6. **RAM budget:** prod's host prompt-cache wants up to 40 GiB. Keep tandem's
   build+test RSS under 8 GiB while prod runs; check `free -h` first.
7. **Kill by PID, never by pattern.** `pkill -f` has matched its own wrapper
   4 times on this box. Filter docker by container ID, never `ancestor=`.
8. **Disk:** check `df -h` before writing anything sized (models, builds).
   Never delete or re-download models without an etag/sha check first.
9. **Sudo:** the user granted sudo for this project. Use it only when a step
   cannot work otherwise, log every use in `notes/`, prefer rootless paths
   (docker group already works). Never edit system services, network config,
   or the power-limit service.
10. **No prod config edits.** tandem gets its own compose file / port. The
    switch between prod engines stays `~/llm/switch.sh`'s job until M7 adds a
    tandem profile *as a new file*.

## Benchmark discipline (for honest numbers, which is also safety)

11. Never benchmark while prod is up or any other GPU user exists — a
    concurrent prefill silently halves decode.
12. ≥ 2 reps minimum; MTP-acceptance noise is ±10% rep-to-rep. Single-rep
    wins are luck, not results.
13. One sweep at a time (the flock enforces it); overlapping sweeps once
    produced a bogus table and OOM'd every cell.
14. Server-class processes always tee stderr to a file; a crash at arg-parse
    with output on `/dev/null` is indistinguishable from a hang.

## Correctness discipline (learned the hard way, 2026-08-19)

17. **The CPU backend cannot validate GPU paths.** Two bugs passed every CPU
    check and only appeared on CUDA: all-f32 KV caches (undertested kernels)
    and an uninitialized device-side mask (CPU's sync fallback hid it).
    Any change touching buffers, layouts, or upload paths must run
    `tandem selftest --gpu` in a bench window before it is believed.
18. **Compare path-matched.** tandem-FA vs oracle `-fa on`; tandem non-FA vs
    `-fa off`. At fp16 both are correct and round differently; an unmatched
    comparison produces a fake regression.
19. **Device memory is not zero.** Compute buffers hold whatever the
    allocator last left there. Anything read before being fully written in
    the same graph must be explicitly initialized.

## Standing invariants

15. Priority order: **ACCURACY > CONTEXT > SPEED** (user's standing rule).
16. Prod stays llama.cpp b10423 + Q8_K_XL untouched until an M7 gate passes
    and the user explicitly flips the switch.
