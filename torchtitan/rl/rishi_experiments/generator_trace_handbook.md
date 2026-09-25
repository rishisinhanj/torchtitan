# Handbook: capturing stock vs. ATO generator Kineto traces (Qwen3-14B, MI355X)

Produces two sets of Perfetto-loadable chrome-trace JSON files (one stock, one
ATO) from vLLM's generator, with nothing else in the loop -- no trainer, no
Monarch actors. Verified working end-to-end 2026-09-25.

## Prerequisites

- Access to the Crusoe node (`crusoe-node` SSH alias; the LB alias `crusoe`
  fails host-key verification -- use `crusoe-node` directly) and a live/held
  Slurm allocation via `spur`:
  ```bash
  export SPUR_CONTROLLER_ADDR=10.245.149.59:6817
  spur exec <jobid> bash   # or: squeue -u $USER to find your jobid
  ```
- Checkout at `~/qwen3_kernel_study` (git worktree of `rishisinhanj/torchtitan`,
  branch `qwen3-kernel-study`), which already contains the patched
  `torchtitan/rl/generate.py`. If it doesn't exist yet:
  ```bash
  cd ~/rissinha_torchtitan_fork/torchtitan  # or wherever you have the fork cloned
  git worktree add ~/qwen3_kernel_study/torchtitan -b qwen3-kernel-study origin/main
  ```
- `~/AMD-TorchTitan-Ops` present (rsync'd, not `git clone` -- this cluster's
  account hits an Enterprise IP allowlist, so `gh`/`git clone` don't work here).
- Checkpoint present at
  `~/qwen3_kernel_study/torchtitan/torchtitan/rl/example_checkpoint/Qwen3-14B`
  (18 files). If missing:
  ```bash
  python3 scripts/download_hf_assets.py --repo_id Qwen/Qwen3-14B \
    --local_dir torchtitan/rl/example_checkpoint --all
  ```
- Docker image `titanrl:mi355-rocm` built (see below).

## 1. Build the docker image (skip if already built)

```bash
cd ~/qwen3_kernel_study
docker build -t titanrl:mi355-rocm -f Dockerfile.titanrl-vllm-rocm-mi355 .
```

Takes ~15-20 min (vLLM + AITER built from source). Only rebuild when a
*dependency* changes -- source edits are bind-mounted in at run time instead.

## 2. What's already patched into `generate.py` (nothing to do, just know it's there)

`~/qwen3_kernel_study/torchtitan/torchtitan/rl/generate.py` has, on top of
upstream:
- `_vllm_attention_backend()` -- routes ROCm to `ROCM_AITER_FA` instead of
  `CUSTOM` (ports pytorch/torchtitan#4866; `CUSTOM` asserts on a CUDA-only
  field and would otherwise crash at engine construction).
- `os.path.abspath(config.hf_assets_path)` -- `CheckpointManager.Config`
  rejects a relative path.
- CLI flags: `--override-imports`, `--warmup`, `--profile`, `--profile-dir`
  (default `/tmp/generate_traces`), `--profile-tag`.
- `_profile_one_pass()` -- a plain `torch.profiler.profile(activities=[CPU,
  CUDA])` context manager around one generation pass, `export_chrome_trace()`
  at the end. This is what actually writes the trace files.

If any of these are missing, `git diff origin/main -- torchtitan/rl/generate.py`
on this worktree to see the exact patch.

## 3. Capture the stock trace

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 128g \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --network=host \
  -v ~/qwen3_kernel_study/torchtitan:/app/torchtitan/torchtitan \
  -v ~/qwen3_kernel_study/outputs:/app/torchtitan/outputs \
  -w /app/torchtitan \
  titanrl:mi355-rocm \
  torchrun --nproc_per_node=8 -m torchtitan.rl.generate \
    --module alphabet_sort --config rl_grpo_qwen3_14b \
    --profile --profile-dir outputs/traces/stock --profile-tag stock
```

## 4. Capture the ATO trace

Same command, plus the ATO mount/env and `--override-imports`:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 128g \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --network=host \
  -v ~/qwen3_kernel_study/torchtitan:/app/torchtitan/torchtitan \
  -v ~/qwen3_kernel_study/outputs:/app/torchtitan/outputs \
  -v ~/AMD-TorchTitan-Ops:/ato:ro -e PYTHONPATH=/ato \
  -w /app/torchtitan \
  titanrl:mi355-rocm \
  torchrun --nproc_per_node=8 -m torchtitan.rl.generate \
    --module alphabet_sort --config rl_grpo_qwen3_14b \
    --override-imports amd_titan.ops.attention.qk_norm_rope \
    --profile --profile-dir outputs/traces/ato --profile-tag ato
```

`qk_norm_rope` is the only override that reaches the generator at all (vLLM
substitutes `inner_attention` before overrides run, so `attention.varlen`
never touches this side -- 0/28 nodes). Combining `qk_norm_rope` with
unscoped `norm.rmsnorm` / `rope.neox` raises a `ValueError` at
`apply_overrides()` time (ancestor/descendant conflict) -- don't add them.

## 5. Confirm it worked

```bash
ls -la ~/qwen3_kernel_study/outputs/traces/stock/
ls -la ~/qwen3_kernel_study/outputs/traces/ato/
```

Expect `stock_rank{0-7}.json` (~24-25MB each) and `ato_rank{0-7}.json`
(~20MB each). Sanity check the run actually generated real text (not a
silent hang) by grepping the container's stdout for the alphabetical-sort
output and a clean `profiler_stop` per rank -- container exit code 0 alone is
**not** sufficient evidence of success on this stack (Monarch/vLLM can mask a
crash as a clean-looking exit in other contexts; for this standalone script
specifically, check the printed generation output).

## Gotchas that will burn you if skipped

1. **Bind-mount depth.** The source mount target is
   `/app/torchtitan/torchtitan` (one level *below* the workdir), not
   `/app/torchtitan`. Mounting at `/app/torchtitan` shadows the package
   directory the image already has baked in at that exact path, and
   `import torchtitan` fails with `ModuleNotFoundError` even though the repo
   is right there.
2. **Don't forget the `outputs` bind-mount.** Without
   `-v ~/qwen3_kernel_study/outputs:/app/torchtitan/outputs`, everything
   writes inside the disposable `--rm` container and is gone the moment it
   exits -- this cost a full completed run earlier in this investigation.
3. **`--nproc_per_node=8`, not 4.** This script has no trainer, so all 8 GPUs
   go to the generator (`rl_grpo_qwen3_14b`'s own `tensor_parallel_degree`
   default). Launching with 4 (the split used for the combined trainer+
   generator run) causes a torchrun elastic failure.
4. **`amd_titan` must be bind-mounted, not baked in.** No image build so far
   has installed ATO's Python package; `PYTHONPATH=/ato` is required for the
   ATO leg every single time.
5. **AITER JIT is not cached across separate container runs.** Expect a fresh
   `module_fused_qk_norm_rope_cache_quant_shuffle` build (a couple minutes)
   on every ATO-leg invocation in a fresh `--rm` container.

## Reading the result

Pull `stock_rank0.json` and `ato_rank0.json` to a local machine and open each
in https://ui.perfetto.dev (drag-and-drop). For an automated kernel-level
diff instead of eyeballing two timelines:

```bash
python3 ~/AMD-TorchTitan-Ops/scripts/compare_kineto_traces.py \
  outputs/traces/stock/stock_rank0.json outputs/traces/ato/ato_rank0.json \
  --threshold 0.03
```

Expect the ATO leg to collapse the unfused RoPE math (`mul`/`add`/`neg`/
bf16-copy, ~36ms across those ops) into one `fused_rope_rms_1way_kernel`
call (~3.8ms) -- roughly a 9.6x reduction on that specific kernel, consistent
with the standalone microbench's measured 7.8x-18.8x range for this op. The
script's naive total-time delta is **not** reliable on its own -- it's
dominated by unrelated communication-kernel noise between two independent
single-sample runs (e.g. allreduce and copy-buffer kernels neither leg's
override touches). Trust the per-named-kernel deltas, not the aggregate.
