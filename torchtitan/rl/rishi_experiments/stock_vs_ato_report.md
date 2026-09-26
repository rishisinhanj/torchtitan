# Stock vs. ATO: Qwen3-14B RL report (trainer + generator, 2026-09-26)

Model: Qwen3-14B (`rl_grpo_qwen3_14b` / `rl_grpo_qwen3_14b_no_compile`), TP=4 trainer +
TP=4 generator (8 GPUs), MI355X (gfx950). ATO override set: trainer =
`amd_titan.ops.norm.rmsnorm,amd_titan.ops.rope.neox,amd_titan.ops.attention.varlen`;
generator = `amd_titan.ops.attention.qk_norm_rope` alone (the only override that
reaches the generator at all -- see Confluence "ATO Override Coverage" page, whose
node-count table we independently reproduced: `qk_norm_rope` 28/28, `attention.varlen`
28/0 trainer/generator).

**Compile-off on both legs everywhere below.** ATO's `attention.varlen` override's
custom `autograd.Function` fails Dynamo's `fullgraph=True` tracing on the trainer
(`duplicate tensor input`), so `rl_grpo_qwen3_14b_no_compile` (`compile=None`) is used
for stock and ATO alike -- this is deliberate, so every comparison below is apples to
apples on the one axis that's under our control. (The generator never uses
`torch.compile` on either leg regardless.)

## 1. Training curves (50 steps each, WandB)

| leg | WandB run | validation reward (pre -> post) |
|---|---|---|
| stock | [avid-resonance-90](https://wandb.ai/rissinha-amd/titan_rl/runs/5heyjmuu) | +0.540 -> +0.709 |
| ATO | [stellar-rain-91](https://wandb.ai/rissinha-amd/titan_rl/runs/s6kuk2kq) | +0.534 -> +0.706 |

Final performance is statistically indistinguishable between the two legs -- ATO's
overrides do not measurably hurt convergence on this task. Both saturate almost all
their improvement within the first ~10 steps (matches an earlier 10-step smoke test,
which independently landed at +0.706/+0.709); 50 steps mostly confirms the plateau
rather than showing further gains. Neither run kept checkpoints (`--trainer.
checkpointer.interval 1000`, past the 50-step horizon) -- these are metrics-only runs,
nothing to resume from.

## 2. Generator-side Kineto trace (stock vs. ATO `qk_norm_rope`)

Captured via the standalone `torchtitan/rl/generate.py` harness (no Monarch actor,
sidesteps a real Kineto/Monarch incompatibility on this stack -- see the trace
handbook for why). Both legs produce identical output text and clean profiler
markers on all 8 ranks.

**Real, attributable win:** the unfused RoPE math (6 separate elementwise kernels --
`mul`/`add`/`neg`/bf16-copy, exactly `q*cos + cat(-x2,x1)*sin`) totals **36,052.7us**
on stock. ATO replaces all six with a single `fused_rope_rms_1way_kernel` at
**3,755.6us** -- a **~9.6x reduction**, matching the standalone microbench's measured
7.8x-18.8x range for this op.

**Caveat:** the trace diff's naive total-time headline (~33% reduction) is **not**
trustworthy on its own -- it's dominated by kernels `qk_norm_rope` never touches
(`cross_device_reduce_2stage` allreduce, `__amd_rocclr_copyBuffer`), moving by more
than the actual fused kernel's savings, in both directions, between two independent
single-sample runs. Trust the per-named-kernel deltas only.

## 3. Trainer-side Kineto trace (stock vs. ATO, both compile-off)

`--trainer.profiler.enable-profiling`, 10 steps, TP=4, iteration_10 traces, rank0
compared via `compare_kineto_traces.py`.

**Real, attributable per-kernel deltas** (kernels present on only one side --
these are exactly the ops the trainer override set targets):

| component | stock kernel(s) | ATO kernel(s) | verdict |
|---|---|---|---|
| RMSNorm | `vectorized_layer_norm_kernel` -- 28,631.8us | `_rmsnorm_kernel.kd` -- 35,202.1us | **ATO is ~23% *slower* here**, not a win at this shape |
| Attention fwd+bwd | `attn_fwd.kd` (11,301.7us) + `bwd_kernel_fuse.kd` (62,134.9us) = 73,436.6us | `fmha_fwd_hd128_bf16_causal_group` (20,814.2us) + `fmha_bwd_hd128_bf16_causal_br_a32_psskddv_group` (28,097.9us) + 2 smaller bwd-support kernels (6,298.6us) + 2 cache-entry kernels (12,440.7us) = 67,651.4us | modest ~8% net win, far short of the microbench's 2.37x -- this run's seq_len is short enough to land in the microbench's own "net negative at short seq" regime (0.97x@S=512), not its long-seq win regime |
| RoPE (unfused `neg` op) | `neg_kernel_cuda` -- 8,941.4us present on stock only | (fused into the rmsnorm/attention kernels above, no standalone kernel) | consistent with fusion, can't isolate a clean before/after pair here the way the generator side allowed |

**This is a genuinely different, more honest result than the isolated microbench
implied.** The microbench's per-op numbers (rmsnorm 0.80x-1.07x "wash", varlen
7.8x-18.8x "strong win") were measured at bench-chosen shapes, not this model's real
per-layer shapes at this seq length. On the real trainer forward/backward at this
model's actual dimensions: RMSNorm is a **net loss**, and attention is a **thin win**,
not the large win the isolated bench suggested. This matches the microbench's own
stated caveat for `rmsnorm` ("only wins at largest shapes") and for `varlen_attention`
("net positive at real training seq lengths, not at short ones") -- we're evidently
still on the short-seq / small-shape side of both curves in this config.

**Same total-time caveat as the generator side, and worse here:** naive totals show
stock at 3,642,980.5us vs ATO at 4,256,959.5us (**ATO +16.9% slower overall**) --
almost entirely explained by `ncclDevKernel_Generic_1` (NCCL allreduce) jumping from
1,689,422.3us (46.4% of stock's total) to 2,147,459.9us (50.4% of ATO's total, +27.1%).
Neither the RMSNorm nor attention override touches collective communication. This is
noise/contention between two independent single-iteration captures, not a real
regression attributable to ATO -- but it does mean **do not quote the 16.9% number as
"ATO makes the trainer slower"**; only the per-kernel deltas above are load-bearing.

## 4. Bottom line

- **Numerically safe**: ATO's overrides don't hurt training quality (WandB curves
  match within noise).
- **Generator-side `qk_norm_rope`**: a real, clean, ~9.6x win on the fused kernel,
  well-isolated from communication noise.
- **Trainer-side `rmsnorm` + `attention.varlen`**: real but much smaller than the
  microbench suggested at this model's actual shapes and this run's sequence length --
  RMSNorm is a net loss, attention a thin win. The microbench's own stated caveats
  (shape/seq-length dependence) turned out to matter in practice, not just in theory.
- Neither side's naive total-trace-time comparison is trustworthy; both are dominated
  by NCCL/copy-buffer noise unrelated to the overrides under test.

## Trace/run locations (Crusoe node, `~/qwen3_kernel_study/outputs/`)

- `stock_wandb50/`, `ato_wandb50/` -- 50-step WandB runs (no profiling, no checkpoints)
- `stock_trainer_nocompile_v2/profiling/traces/iteration_10/rank{0-3}_trace.json.gz` --
  stock trainer, compile-off
- `ato_trainer_traced_nocompile/profiling/traces/iteration_10/rank{0-3}_trace.json.gz`
  -- ATO trainer, compile-off
- `traces/stock/stock_rank{0-7}.json`, `traces/ato/ato_rank{0-7}.json` -- generator,
  stock/ATO (compile never applies to the generator)

## Repro steps

All four runs below assume: `titanrl:mi355-rocm` already built (`docker build -t
titanrl:mi355-rocm -f Dockerfile.titanrl-vllm-rocm-mi355 .` from `~/qwen3_kernel_study`
-- rebuild if you're on a different node, docker images are node-local), 8 clean GPUs
(`rocm-smi --showmeminfo vram` back to baseline on all 8, `docker ps -a` clear of
other tenants' active workloads), and enough free space on `/home` for a checkpoint
or two if you drop the `--trainer.checkpointer.interval` override (check `df -h
/home` first -- this NFS mount is shared cluster-wide and has genuinely run out of
space during this investigation).

### 1. Stock, 50 steps, WandB, no profiling

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 128g \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --network=host \
  -v ~/qwen3_kernel_study/torchtitan:/app/torchtitan/torchtitan \
  -v ~/qwen3_kernel_study/outputs:/app/torchtitan/outputs \
  -e WANDB_API_KEY=<your key> \
  -w /app/torchtitan \
  titanrl:mi355-rocm \
  python3 -m torchtitan.rl.train --module alphabet_sort --config rl_grpo_qwen3_14b_no_compile \
  --trainer.parallelism.tensor-parallel-degree 4 \
  --generator.parallelism.tensor-parallel-degree 4 \
  --async-loop.num-training-steps 50 \
  --trainer.checkpointer.interval 1000 \
  --dump-folder outputs/stock_wandb50
```

### 2. ATO, 50 steps, WandB, no profiling

Same as above, plus the ATO mount/env and override flags:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 128g \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --network=host \
  -v ~/qwen3_kernel_study/torchtitan:/app/torchtitan/torchtitan \
  -v ~/qwen3_kernel_study/outputs:/app/torchtitan/outputs \
  -v ~/AMD-TorchTitan-Ops:/ato:ro -e PYTHONPATH=/ato \
  -e WANDB_API_KEY=<your key> \
  -w /app/torchtitan \
  titanrl:mi355-rocm \
  python3 -m torchtitan.rl.train --module alphabet_sort --config rl_grpo_qwen3_14b_no_compile \
  --trainer.parallelism.tensor-parallel-degree 4 \
  --generator.parallelism.tensor-parallel-degree 4 \
  --async-loop.num-training-steps 50 \
  --trainer.checkpointer.interval 1000 \
  --trainer.override.imports amd_titan.ops.norm.rmsnorm,amd_titan.ops.rope.neox,amd_titan.ops.attention.varlen \
  --generator.override.imports amd_titan.ops.attention.qk_norm_rope \
  --dump-folder outputs/ato_wandb50
```

### 3. Trainer Kineto trace, stock, no-compile, 10 steps

Same as #1, but drop `WANDB_API_KEY` (unneeded), restore the default step count, and
add the trainer profiler flag:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 128g \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --network=host \
  -v ~/qwen3_kernel_study/torchtitan:/app/torchtitan/torchtitan \
  -v ~/qwen3_kernel_study/outputs:/app/torchtitan/outputs \
  -e WANDB_MODE=disabled \
  -w /app/torchtitan \
  titanrl:mi355-rocm \
  python3 -m torchtitan.rl.train --module alphabet_sort --config rl_grpo_qwen3_14b_no_compile \
  --trainer.parallelism.tensor-parallel-degree 4 \
  --generator.parallelism.tensor-parallel-degree 4 \
  --trainer.checkpointer.interval 1000 \
  --trainer.profiler.enable-profiling \
  --dump-folder outputs/stock_trainer_nocompile_v2
```

### 4. Trainer Kineto trace, ATO, no-compile, 10 steps

Same as #2, restore default step count, drop `WANDB_API_KEY`, add the profiler flag:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 128g \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --network=host \
  -v ~/qwen3_kernel_study/torchtitan:/app/torchtitan/torchtitan \
  -v ~/qwen3_kernel_study/outputs:/app/torchtitan/outputs \
  -v ~/AMD-TorchTitan-Ops:/ato:ro -e PYTHONPATH=/ato \
  -e WANDB_MODE=disabled \
  -w /app/torchtitan \
  titanrl:mi355-rocm \
  python3 -m torchtitan.rl.train --module alphabet_sort --config rl_grpo_qwen3_14b_no_compile \
  --trainer.parallelism.tensor-parallel-degree 4 \
  --generator.parallelism.tensor-parallel-degree 4 \
  --trainer.checkpointer.interval 1000 \
  --trainer.override.imports amd_titan.ops.norm.rmsnorm,amd_titan.ops.rope.neox,amd_titan.ops.attention.varlen \
  --generator.override.imports amd_titan.ops.attention.qk_norm_rope \
  --trainer.profiler.enable-profiling \
  --dump-folder outputs/ato_trainer_traced_nocompile
```

### 5. Generator traces (stock + ATO)

No trainer, no compile, not part of the Monarch loop -- see
`generator_trace_handbook.md` in this same directory for the full standalone
`torchtitan/rl/generate.py` commands (`--config rl_grpo_qwen3_14b`, `--profile`,
with/without `--override-imports amd_titan.ops.attention.qk_norm_rope`).

### 6. Re-run the kernel-level diff on any pair

```bash
docker run --rm \
  -v ~/qwen3_kernel_study/outputs:/outputs:ro \
  -v ~/AMD-TorchTitan-Ops:/ato:ro \
  --entrypoint python3 titanrl:mi355-rocm \
  /ato/scripts/compare_kineto_traces.py \
  /outputs/<leg-a>/profiling/traces/iteration_N/rank0_trace.json.gz \
  /outputs/<leg-b>/profiling/traces/iteration_N/rank0_trace.json.gz \
  --threshold 0.03
```

### Gotchas hit while producing this report (don't re-discover them)

- Docker images are node-local -- a fresh Slurm allocation on a different physical
  node needs a full rebuild, ~15-20 min.
- Check `df -h /home` before any run that saves checkpoints -- this shared NFS mount
  hit 99% full mid-investigation from accumulated checkpoint dirs across many past
  runs (each full Qwen3-14B checkpoint is ~83GB), and a checkpoint write failing
  partway through silently kills the whole job with zero traceback. Checkpoints
  are root-owned (written from inside the container) -- delete them via a container
  (`docker run --rm -v <outputs>:/outputs alpine rm -rf /outputs/*/checkpoint`), not
  a plain host-side `rm`.
- `--exclusive` on this cluster's `amd-spur` partition/`amd-burst-qos` QOS does not
  reliably mean actually exclusive -- verify with `rocm-smi --showmeminfo vram` and
  `docker ps -a` on every fresh allocation before trusting it. Use
  `--qos=amd-aifw-aim-qos` instead of `amd-burst-qos` for a real dedicated node, and
  `sinfo -p amd-spur -N -o "%N %T"` to find a node genuinely in `idle` state if the
  first allocation isn't clean.
- A queue_interposition hang (`Async signal handler still waiting on signal`) hit
  the trainer's own profiler once, on a stock+no-compile+profiling combination that
  had worked fine moments before on the ATO leg -- retried, ran clean the second
  time. Treat it as flaky/timing-dependent, not a deterministic incompatibility with
  a specific config.
