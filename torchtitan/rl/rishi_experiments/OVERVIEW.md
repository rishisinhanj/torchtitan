# ATO kernel study on TitanRL -- overview

Branch `qwen3-kernel-study` on `rishisinhanj/torchtitan`.

**What this branch is for:** measuring whether AMD-TorchTitan-Ops (ATO) kernel
overrides are worth adopting in the TitanRL loop on MI355X (gfx950), on
**Qwen3-14B**, for both the trainer and the vLLM generator. It contains the
code changes needed to capture Kineto traces from both sides, plus the written
results.

This file is the index. Start here.

---

## 1. Results (read these first)

| where | what |
| --- | --- |
| [Confluence: TitanRL ATO analysis with Generator](https://amd.atlassian.net/wiki/spaces/AIGAIM/pages/2000848804/TitanRL+ATO+analysis+with+Generator) | **The results page.** Numbers only: what we overrode and why, WandB deltas, kernel deltas, logprob agreement, caveats |
| [Confluence: ATO Override Coverage and Model Sharing](https://amd.atlassian.net/wiki/spaces/~712020101ec3cf0d5b432b9174a7eec5246c15/pages/1975423938/TorchTitan+RL+ATO+Override+Coverage+and+Model+Sharing+in+Trainer+and+Generator) | Prior page (teammate's) explaining *why* an override reaches the trainer but not the generator. Our 40/0 counts confirm its model |
| `torchtitan/rl/rishi_experiments/stock_vs_ato_report.md` | Long-form report: results + full repro commands + infra gotchas |

### WandB

| leg | run | link |
| --- | --- | --- |
| stock | `avid-resonance-90` | https://wandb.ai/rissinha-amd/titan_rl/runs/5heyjmuu |
| ATO | `stellar-rain-91` | https://wandb.ai/rissinha-amd/titan_rl/runs/s6kuk2kq |
| project | | https://wandb.ai/rissinha-amd/titan_rl |

Both are 50-step Qwen3-14B runs, compile off, trainer TP=4 + generator TP=4.
Graphs for every metric quoted in the Confluence page live here.

---

## 2. Documents in this repo

All under `torchtitan/rl/rishi_experiments/`:

| file | contents |
| --- | --- |
| `stock_vs_ato_report.md` | Results, all four `docker run` repro commands, kernel diff invocation, infra gotchas |
| `generator_trace_handbook.md` | Standalone runbook for capturing generator traces (stock + ATO). Hand this to someone who just needs traces |
| `mi355_ato_kernel_bench.md` | Running investigation log: ATO setup, Phase 1 microbench (`python -m bench`) results and kernel-selection reasoning, the generator-profiling bring-up saga |
| `structured_logs_to_timeline.py` | Converts a run's `structured_logs/*.jsonl` into one Perfetto trace showing trainer / rollout / generation as parallel tracks |

---

## 3. Code changes on this branch

| file | change |
| --- | --- |
| `torchtitan/rl/attention_backend.py` | **New.** `vllm_attention_backend()` -- routes ROCm to `ROCM_AITER_FA` instead of `CUSTOM` (which asserts on `vllm_flash_attn_version`, unset on ROCm). Ports upstream PR #4866. Shared by `generate.py` and `generator.py` |
| `torchtitan/rl/generate.py` | Standalone generator profiling: `--profile`, `--profile-dir`, `--profile-tag`, `--override-imports`; `_profile_one_pass()`; absolute `hf_assets_path` |
| `torchtitan/rl/generator.py` | Uses the shared attention-backend helper |
| `torchtitan/observability/profiler.py` | Adds `with_stack` config field (trainer traces) |
| `torchtitan/rl/examples/alphabet_sort/config_registry.py` | Adds `rl_grpo_qwen3_14b_no_compile` (`compile=None`) |
| `Dockerfile.titanrl-vllm-rocm-mi355`, `.dockerignore` | Image used for every run here (gfx950) |

**Why a `no_compile` config exists:** ATO's `attention.varlen` override wraps a custom
`autograd.Function` that Dynamo cannot trace under the trainer's per-block
`fullgraph=True` compile (`duplicate tensor input`). `rl_grpo_qwen3_14b` hardcodes
`compile=CompileConfig(...)` with no `enable` field to negate, so a config variant was
the only way to get an apples-to-apples pair. **Both** legs use it.

**Generator profiling is deliberately a standalone script, not part of the RL loop.**
Driving vLLM's profiler through a Monarch actor endpoint fails two ways on this stack:
`SIGSEGV` when backgrounded via `asyncio.to_thread` (Kineto's CUDA/HIP activity backend
is thread-bound), and an unrecoverable hang when called inline on the actor's event loop
(rocprofiler-sdk queue-interposition defect). Three variants were tried; the in-loop
plumbing was reverted in `7b0d7d0e3` rather than left in looking usable. Do not
re-add it without an upstream fix.

---

## 4. Trace files -- NOT in this repo

Traces total ~763 MB, so they live only on the Crusoe shared NFS home, under
`~/qwen3_kernel_study/outputs/`.

> **Warning:** `/home` on that cluster has repeatedly sat at ~100% full (22 GB free of
> 10 TB at last check). These traces exist in exactly one place. A full mount already
> killed one run silently in this work. Copy anything you care about off before
> relying on it.

### Trainer traces (`.json.gz`, 4 ranks, TP=4)

| leg | path |
| --- | --- |
| **stock, compile-off** | `stock_trainer_nocompile_v2/profiling/traces/iteration_10/rank{0-3}_trace.json.gz` |
| **ATO, compile-off** | `ato_trainer_traced_nocompile/profiling/traces/iteration_10/rank{0-3}_trace.json.gz` |
| stock, compile-**on** (superseded) | `stock_trainer_traced/profiling/traces/iteration_4/` |
| post-cleanup verify run | `verify_rl_loop/profiling/traces/iteration_10/` |

### Generator traces (`.json`, 8 ranks, TP=8)

| leg | path |
| --- | --- |
| **stock** | `traces/stock/stock_rank{0-7}.json` |
| **ATO** | `traces/ato/ato_rank{0-7}.json` |
| post-cleanup verify run | `traces/verify_stock/` |

**The two comparable pairs are the bolded ones.** `stock_trainer_traced` is
compile-**on** and therefore does *not* match the ATO trainer leg; the `verify_*`
sets are smoke tests from after a cleanup, not part of the reported comparison.

### Other run artifacts

| path | contents |
| --- | --- |
| `stock_wandb50/`, `ato_wandb50/` | The 50-step runs: `structured_logs/`, `rollout_samples.jsonl`, local `wandb/` dirs. No checkpoints (suppressed deliberately) |
| `<run>/structured_logs/*.jsonl` | Per-actor event logs with epoch-microsecond timestamps -- input to `structured_logs_to_timeline.py` |
| `~/stock_wandb50_timeline.json` | Generated Perfetto timeline (~10 MB, not committed) |

Viewing any of these: drag into https://ui.perfetto.dev.

---

## 5. External dependencies

| what | where | note |
| --- | --- | --- |
| ATO (`amd_titan`) | `~/AMD-TorchTitan-Ops` on the node | **Never baked into the docker image.** Every ATO run bind-mounts it: `-v ~/AMD-TorchTitan-Ops:/ato:ro -e PYTHONPATH=/ato` |
| `compare_kineto_traces.py` | `~/AMD-TorchTitan-Ops/scripts/` | Kernel-level trace diff, pure stdlib |
| Qwen3-14B checkpoint | `torchtitan/rl/example_checkpoint/Qwen3-14B` | ~28 GB, `.dockerignore`d -- bind-mount, never `COPY` |

---

## 6. Gotchas that cost time here

- **Docker images are node-local.** A new Slurm allocation needs a full rebuild (~15-20 min).
- **The source bind-mount target is `/app/torchtitan/torchtitan`**, one level *below* the
  workdir. Mounting at `/app/torchtitan` shadows the package and `import torchtitan` fails.
- **Always bind-mount `outputs/`.** Without it results are written inside the `--rm`
  container and vanish on exit. This cost a full completed run.
- **`--exclusive` has not reliably meant exclusive** on `amd-spur`. Verify with
  `rocm-smi --showmeminfo vram` + `docker ps -a` before trusting a fresh node. Use
  `--qos=amd-aifw-aim-qos`, and `sinfo -p amd-spur -N -o "%N %T" | grep -w idle` to pick
  a genuinely idle host.
- **Checkpoints are root-owned** (written from inside the container). Delete with
  `docker run --rm -v <outputs>:/outputs alpine rm -rf /outputs/*/checkpoint`, not a
  host-side `rm`. Each Qwen3-14B checkpoint is ~83 GB; suppress with
  `--trainer.checkpointer.interval <past horizon>` when you do not need them.
- **`generate.py` takes `--config` only** -- there is no `--module` flag on it (that is
  `train.py`). And it needs `--nproc_per_node=8`, matching the config's own
  `tensor_parallel_degree`, since there is no trainer sharing the node.
- **n=1 trace totals are not a performance measurement** on this stack. Whole-trace
  deltas are dominated by NCCL/copy-buffer noise that no override touches. Trust
  per-named-kernel deltas and the WandB throughput metrics.
