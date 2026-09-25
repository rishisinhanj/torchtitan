# MI355X ATO Kernel Replacement Study

**Goal:** find which TitanRL kernels are worth replacing with AMD-TorchTitan-Ops (ATO)
overrides on MI355X, then trace stock vs. ATO on both trainer and generator. Full plan:
`~/.claude/plans/this-is-the-plan-swirling-cook.md`.

## TL;DR

- **Trainer override set (verified safe):** `--trainer.override.imports
  amd_titan.ops.norm.rmsnorm,amd_titan.ops.rope.neox,amd_titan.ops.attention.varlen`
- **Generator override set (corrected -- see below):** `--generator.override.imports
  amd_titan.ops.attention.qk_norm_rope` **alone**. Originally planned to add
  `norm.rmsnorm`+`rope.neox` too, but that combination raises `ValueError` at
  `apply_overrides()` time -- `qk_norm_rope` is an umbrella over the whole attention
  module, and the other two independently match its own child nodes (neither is
  FQN-scoped), which torchtitan's override system mechanically forbids
  (ancestor/descendant conflict check). Cost: the generator's non-attention RMSNorm
  calls run stock, not ATO-accelerated -- no FQN-scoped `rmsnorm` variant exists to
  exclude just the qk_norm submodule.
- **Target model:** Qwen3-14B (`rl_grpo_qwen3_14b`) -- confirmed compatible with every
  override above (head_dim=128, RMSNorm+CosSinRoPE qk_norm, `VarlenInnerAttention`,
  bf16 -- exactly ATO's own stated validated domain for `qk_norm_rope`).
- **Compile-off decision-log harvest needs a code change, not a flag.** `rl_grpo_qwen3_14b`
  hardcodes `compile=CompileConfig(backend="aot_eager")`; the field is `CompileConfig | None`
  with no `enable` sub-field, so no `--compile...` CLI flag can null it out (the
  `--compile.no-enable` some docs reference is stale against this torchtitan pin). Need
  a small config variant setting `compile=None` for the compile-off diagnostic pass.
- **GPU topology (decided):** override to trainer TP=4 + generator TP=4 (8 GPUs total,
  one node) instead of the registered TP=8/TP=8 default (16 GPUs -- more than this node
  has; trainer and generator run as separate concurrent meshes, not time-shared).
- **Isolated workspace (decided):** job `174449`/`crsuse2-m2m-091` is a **separate agent**
  doing real Muse Glimmer enablement work on `~/rissinha_torchtitan_fork/torchtitan`
  (branch `ato_titanRL_integration`) -- not ours to touch. Our own work now lives in a
  fresh git worktree, `~/qwen3_kernel_study` (new branch `qwen3-kernel-study` off
  `main`, same commit `0dd5f78` as the Muse Glimmer branch, so our already-verified
  5-file profiling patch applied there with a clean 3-way merge, zero conflicts).
  Reverted our earlier direct edits to the shared checkout before switching.
  **Their `ENABLEMENT.md` is worth reading in full** -- directly relevant findings:
  Path A (alphabet_sort, which `rl_grpo_qwen3_14b` also uses) already completes
  training steps on this same image/torchtitan commit; the `trunc_normal_` segfault
  they hit doesn't affect Qwen3 (uses `normal_`, not `trunc_normal_`); and critically,
  **container exit code 0 does not mean success** -- Monarch intercepts segfaults and
  reports a proc-agent timeout instead, so `grep`-ing for `SIGSEGV`/`Segfault` finds
  nothing on a real crash; check for `"The process likely crashed"` + native stack
  frames instead. Applying this to our own stock/ATO verification.
- **Status:** Phase 1 done. Qwen3-14B compatibility verified (safe to run with the
  corrected override set + TP=4/TP=4). TitanRL profiling-code agent done (5 files,
  all 4 known bugs avoided, zero new lint findings -- see below). Docker image needs
  rebuilding again in the new isolated worktree (the one built earlier was against the
  shared checkout, now abandoned).

### TitanRL profiling-code agent -- what it built

Real vLLM v0.29.0 source traced (not guessed) to confirm the correct profiler-enable
API: `ProfilerConfig(profiler="torch", torch_profiler_dir=...)` passed into
`EngineArgs` at construction, not the `VLLM_TORCH_PROFILER_DIR` env var (confirmed via
the actual chain `EngineArgs.profiler_config -> VllmConfig -> GPUWorker.profile()`).
`start_profiling`/`stop_profiling` wrapped in `asyncio.to_thread`; trace dir keyed by
`generator_name`, not rank; `close()`'s endpoint decoration untouched (diffed to
confirm byte-for-byte). `controller.py`'s `profile_at_step` brackets one step in a
`try/finally`, existing generator-handle-retention pattern reused as-is (checked: no
actual handle-discarding bug exists in this codebase's `InterGeneratorRouter`, unlike
the prior local diff). Verified: `ast.parse` + repo's own pinned `ufmt`/`flake8`
clean on all 5 files, zero new findings. Not verifiable without a live GPU/vLLM:
whether `profiler_config` actually threads through on the real ROCm build, and
whether the `to_thread`-wrapped calls race the concurrent `_engine_loop` task.

Trace-diff script: `/home/rissinha/AMD-TorchTitan-Ops/scripts/compare_kineto_traces.py`
(91 lines) -- loads a gzipped chrome-trace JSON, sums duration by kernel name
(`cat in {kernel, gpu_memcpy, gpu_memset}`, excludes host-side ops), diffs two traces.
Confirmed no built-in `torch.profiler` loader exists for a saved trace file, so this
is the minimum necessary custom code, not a choice to reinvent something.

### Still needed before the run
- No CLI flag disables `torch.compile` for `rl_grpo_qwen3_14b` (`CompileConfig` has no
  `enable` field; `--compile.no-enable` some docs reference is stale against this
  torchtitan pin). Fixed: added `rl_grpo_qwen3_14b_no_compile()` config variant
  (identical, `compile=None`) to `alphabet_sort/config_registry.py`, for the
  decision-log harvest pass only (ATO goes silent under `torch.compiler.is_compiling()`).
- Ported PR #4866 (`_vllm_attention_backend`, routes ROCm generator to
  `ROCM_AITER_FA` instead of `CUSTOM`) into `torchtitan/rl/generator.py` -- still
  unmerged upstream as of this session, and `rl_grpo_qwen3_14b` uses `attn_backend=
  "varlen"`, which hits exactly the bug this PR fixes (`CUSTOM` asserts on
  `vllm_flash_attn_version`, None on ROCm). **Confirmed working**: the actual stock
  run got past attention-backend init entirely with no assertion.
- `hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-14B"` is a **local
  directory**, not an HF Hub auto-download. Populated via `scripts/download_hf_assets.py
  --repo_id Qwen/Qwen3-14B --local_dir torchtitan/rl/example_checkpoint --all` (18
  files, confirmed present). **Added this path to `.dockerignore`** -- without it, the
  ~28GB checkpoint gets baked into every image rebuild via `COPY .`; bind-mount it at
  `docker run` time instead (`-v .../Qwen3-14B:/assets:ro`).

### Cluster gotcha (important, will recur): `--exclusive` does not guarantee GPU isolation
First real stock-run attempt failed with `ValueError: Free memory on device cuda:2
(14.54/287.98 GiB)...` -- looked like a trainer/generator GPU-partitioning bug at
first (verified the partitioning code itself, `PerHostProvisioner` in `rl/train.py`,
is correct: sequential disjoint `CUDA_VISIBLE_DEVICES` ranges). Real cause: `rocm-smi`
showed **all 8 GPUs ~94% full** even after our container had already exited, and
`docker ps -a` revealed **other teams' unrelated containers already running on this
"exclusive" node** (`ds4pro8`, `geak_evok`, both `Up`, images `atom-dsv4-mlafix` /
`sglang-rocm` -- nothing to do with us). Stopping them (with explicit confirmation,
since this terminates other teams' active work on a shared cluster) freed the node
back to ~297MB/GPU baseline. **Lesson: `sbatch --exclusive` on this cluster does not
guarantee GPU memory isolation from other users' long-running containers. Check
`rocm-smi --showmeminfo vram` + `docker ps -a` on any freshly-allocated node before
trusting it's actually clean.**

## MILESTONE: stock Qwen3-14B TitanRL confirmed working end-to-end (2026-09-25)

3-step smoke test, TP=4 trainer + TP=4 generator, clean GPU node. **Genuine success,
verified by content not exit code**: all 3 steps logged real metrics (loss/mean,
rollout_reward, grad_norm, entropy, lr -- not placeholder zeros), validation ran with
real pre/post reward (+0.530 -> +0.709), clean teardown ("tearing down actors" ->
"Shutting down vLLM renderer" x4 -> exit 0), zero crash signatures anywhere in the
log. GPU partition confirmed correct: trainer on 0-3 (~45GB/GPU), generator on 4-7
(~282GB/GPU, vLLM's default KV-cache preallocation). This directly updates ATO's own
`run_titanrl.sh` pessimism ("no training step has ever completed") for our specific
config+environment -- it now reliably does.

Next: wire in trace capture (the `with_stack` field + generator profiling code
already merged into this same `qwen3_kernel_study` worktree by the TitanRL agent)
and run the actual stock-vs-ATO comparison -- the one stated goal.

## Generator in-process profiling: real, unresolved native bug (2026-09-25)

Trainer-side profiling (`--trainer.profiler.enable-profiling`, upstream's own
already-proven Kineto wrapper + our tiny `with_stack` addition) carries no new risk
and is not implicated in anything below.

Generator-side profiling (the subagent's new `start_profiling`/`stop_profiling` via
vLLM's `ProfilerConfig`) hit two different real failures, both empirically confirmed,
both at the exact point `--profile-at-step` engages (training itself is unaffected --
steps before that point always complete cleanly with real metrics):

1. **`asyncio.to_thread`-wrapped calls (as the subagent wrote it): SIGSEGV**
   (`Killed(sig=11, core)`) in the generator actor. Consistent with calling
   CUDA/HIP-touching profiler APIs from a thread that never bound that GPU's device
   context (context is thread-local; `asyncio.to_thread` uses a fresh executor
   thread).
2. **Direct/synchronous call (removing `to_thread`): hang, not a crash.** Confirmed
   via `ps`/log evidence, not exit code: a climbing "Async signal handler still
   waiting on signal" loop in ROCm's own `queue_interposition.cpp` (its
   kernel-launch-interception layer, underlying the CUDA/HIP profiler activity
   backend), never resolving. Killed manually; left GPUs 5-7 holding stale memory
   afterward, cleaned up with `docker rm -f` + recheck.

Net: neither of the two natural implementations of "toggle vLLM's profiler from
Monarch's actor RPC layer" works cleanly on this stack (torch nightly + vLLM 0.28 +
Monarch + ROCm gfx950). This looks like a real, deep interaction between Monarch's
async actor runtime and ROCm's profiler signal handling, not something to keep
guessing at via trial and error with the two obvious threading choices already
exhausted.

**Superseded by a better option, found via 2 parallel research agents (2026-09-25):**
`rocprofv3` was ruled out too (`bench/rocprof.py`'s own finding: SIGABRTs on our
runtime skew; `rocprofv2` refuses gfx950 outright). More importantly, a **proven,
already-working, actor-free profiling harness already exists**:
`torchtitan/experiments/rl/generate.py` on branch `upstream/inference-ablation`
(`_profile_one_pass()`) -- a plain `torch.profiler.profile()` context manager,
`torchrun`-launched, on the script's own main thread, no Monarch actor/RPC/thread
hand-off at all. ~10 commits already cite real per-kernel numbers pulled from it.
A second agent's research on Monarch's threading/signal internals *independently*
confirmed why our two attempts failed and why this structurally can't hit the same
bugs: the actor Python runs on a dedicated stable thread (`monarch-actor-event-loop`,
not a pool thread) with Monarch installing no fatal-signal handlers in OSS builds --
the hang is a **known rocprofiler-sdk inline-queue-interposition defect**, and the
segfault is Kineto's thread-binding requirement violated by `asyncio.to_thread`.
Neither is fixable by picking a third threading variant.

## MILESTONE: generator Kineto trace captured successfully (2026-09-25)

Ported the proven `_profile_one_pass` pattern into the newer-layout
`torchtitan/rl/generate.py` (242 lines, already existed, already threads through
ATO's real override mechanism via `config.generator.override` -- did not need to
port the whole 857-line ablation script). Also added the same ROCm attention-backend
fix (PR #4866) this script was missing, and a `--override-imports` CLI flag.

Three real bugs found and fixed getting this to run (in order): the leftover broken
docstring escaping in `config_registry.py` needed its bind-mount restored (image's
baked-in copy is stale); `CheckpointManager.Config` requires an absolute
`hf_assets_path` (script had it relative) -- fixed with `os.path.abspath()`;
`--nproc_per_node` must match the config's own `tensor_parallel_degree` (8 for
`rl_grpo_qwen3_14b`'s generator, not the 4 we'd been using for the trainer+generator
split -- irrelevant here since this script has no trainer at all).

**Stock leg: confirmed clean.** Correct generated output twice (warmup + profiled
pass: `<alphabetical_sorted>Alice, Bob, Charlie</alphabetical_sorted>`), clean
`profiler_stop` USDT markers on all 8 ranks, trace files written and substantial
(24-25MB each, `outputs/traces/stock/stock_rank{0-7}.json`).

**ATO leg (`--override-imports amd_titan.ops.attention.qk_norm_rope`): also
confirmed clean.** Same signature as stock (correct output twice, clean
`profiler_stop` all 8 ranks). Needed two more real fixes en route: `amd_titan`
wasn't installed in the image at all (all prior ATO work happened in the separate
bench venv, never inside this container) -- fixed by bind-mounting
`~/AMD-TorchTitan-Ops` at `/ato` + `PYTHONPATH=/ato`; and a fresh AITER JIT compile
of `module_fmha_v3_varlen_fwd` in the disposable `--rm` container (expected, not a
bug -- JIT cache doesn't survive across separate container runs).

## Stock vs ATO comparison result (2026-09-25) -- goal achieved, with an honest caveat

Ran `scripts/compare_kineto_traces.py` on both rank0 traces (24.6MB stock, 20.5MB
ato). **Real, clean, attributable win, cross-validated against the microbench:**

- Stock: 6 separate elementwise kernels (`mul`/`add`/`neg`/bf16-copy -- exactly the
  unfused `q*cos + cat(-x2,x1)*sin` RoPE math from the bench's own reference
  implementation), summing to **36,052.7us**.
- ATO: replaced by **one** kernel, `fused_rope_rms_1way_kernel` (same kernel name
  as the microbench's `aiter_fused` candidate), at **3,755.6us**.
- **~9.6x reduction, ~32,300us saved** -- closely matches the microbench's measured
  7.8x-18.8x range for this op. This is the real signal.

**Caveat, stated plainly rather than glossed over:** the trace diff's naive
"232,886us -> 155,772us total" (~33% reduction) headline is **not** a reliable ATO
win -- it's dominated by kernels `qk_norm_rope` never touches: `cross_device_reduce_2stage`
(an inter-GPU allreduce, -98.3%, ~59K us) and `__amd_rocclr_copyBuffer` (+257.9%,
~30.7K us) move by far more than the actual fused kernel's savings, in both
directions. This is noise between two independent single-sample (n=1) process
launches, not a code-path difference -- the two traces are **not** "identical
except for the replaced kernel" at the raw-total level, only at the specific
fused-region level. A repeated-pass (n>1) comparison would be needed to make the
aggregate total trustworthy; the per-kernel attributable finding above does not
need that caveat, since it's the same named kernel in both traces changing for an
architecturally-explained reason (fusion), not noise.

## Setup (do this once per fresh node)

Node is burst-QoS -- **expect preemption**; already happened once (job `174023`
CANCELLED after 6h). `~/AMD-TorchTitan-Ops` lives on shared NFS so it survives; docker
images and anything else node-local do not.

1. `rsync` (not `git clone`) `~/AMD-TorchTitan-Ops` + its `third_party/aiter` (incl.
   nested `composable_kernel`) from a machine with GitHub access -- this cluster's
   account hits `amd-eng-emu`'s Enterprise IP allow list, so `gh`/`git clone` don't
   work here at all.
2. `uv pip install --no-build-isolation ./third_party/aiter` into the `make setup` venv.
3. `uv pip install -e third_party/torchtitan` (full deps, not `--no-deps` -- ATO's own
   op modules import `torchtitan.config` directly; `make setup`'s README claiming
   torchtitan isn't needed is stale). Confirmed torchtitan doesn't pin `torch` itself,
   so this doesn't disturb the `torch==2.14.0+rocm7.2` pin.
4. `uv pip uninstall nvidia-cutlass-dsl` (+ its `-libs-*` siblings) -- a CUDA-only
   package pulled in transitively (via torchao) whose MLIR/nanobind bindings **abort
   the process** (not a catchable exception) colliding with FlyDSL's, because `bench`
   eagerly imports every op family regardless of which one you're testing. Same bug
   *class* as the documented aiter/flydsl collision, different package. `torchao`
   still imports fine without it.
5. Use `spur exec <jobid> bash -c '...'` for everything -- `srun --overlap` was found
   wedged specifically for one job/node (isolated by testing the same command against
   a different held job, which was instant). Cause unconfirmed; `spur exec` just works.

Two harmless recurring warnings (ignore): torchao's `_C_cutlass_90a.abi3.so` (CUDA
SM90-only) and a `cp310`-tagged `.so` (ABI mismatch vs our `cp311` venv) both fail to
load and torchao gracefully degrades those paths to unavailable.

**Known ATO bug, not ours to fix, not blocking:** `bench/e2e/mla_attention.py` pins
`backend="aiter_flash_attn"`, a name ADR 0042's rename (`6eab3e1`) changed to `"aiter"`
everywhere else -- raises `ForcedBackendError` at HEAD.

## Phase 1a -- microbench (`python -m bench <op> --check`, isolated op, real GPU tensors)

| op | SQNR | speedup | verdict |
|---|---|---|---|
| `rmsnorm` | 55.6dB, matches stock exactly | 0.80x-1.07x | Wash; free but only wins at largest shapes |
| `rope` | 55.6dB, matches stock exactly | **3.0x-10.5x**, grows with seq len | Clean win, numerically free -- best result |
| `qk_norm_rope` | **50.8dB vs stock's 52.6dB** -- real ~1.8dB cost, confirmed apples-to-apples (both return full-precision q/k, not a quantized side-buffer) | **7.8x-18.8x** | Strong win with a real (small) accuracy cost, thin margin above the 50dB bar. Reaches both trainer+generator, but trainer separately blocked by a Confluence-documented compile issue (`torch.autograd.grad` inside `backward`, untraceable under `fullgraph=True`) -- unrelated to this bench, which is forward-only |
| `varlen_attention` | matches stock within 0.1dB | **0.97x at S=512** (slightly slower) -> 2.37x at S=4096 | Net positive at real training seq lengths, not at short ones. This is the inner-kernel override -- reaches trainer only (0/28 on generator, vLLM substitutes attention first) |
| `mxfp8_linear` | -- | -- | **Fails `--check` (rc=1), environment gap not a bug.** Champion candidate `primus_turbo_gemm_fp8` needs a from-source HIP build we haven't done; `make setup`'s venv doesn't include it. No MXFP8 comparison data available without that build. `rl_grpo_qwen3_0_6b_varlen` is bf16 with no MXFP8 converter anyway, so may not matter. |

## Phase 1b -- `bench.e2e` (trainer-only, full training loop)

**Not treated as a source of truth for the kernel decision.** `bench.e2e`'s built-in
experiments are hardcoded to Llama3/Flux shapes (confirmed: `varlen_attention`'s
route_key showed `head_dim=16`, a tiny debugmodel proxy value, nothing like Qwen3's
real `head_dim=128`), so a result here doesn't transfer to Qwen3. Kept running only to
confirm ATO's dispatch/decision-log mechanics work end-to-end in a real training loop
(they do -- clean `reason=forced`, zero baseline contamination); real Qwen3 numbers
have to come from the actual TitanRL run.

## Phase 2 -- kicked off (2026-09-24, in progress)

Two subagents dispatched in parallel (pure code/research, no GPU needed):

- **TitanRL agent**: add `with_stack` field to upstream `Profiler.Config` (trainer);
  write generator-side profiling from scratch (doesn't exist upstream at all) --
  briefed on 4 known bugs from a prior attempt to avoid (endpoint decoration lost on
  `close()`, wrong vLLM profiler env var vs. real `ProfilerConfig`-at-construction API,
  all-generators-collide-on-rank-0 trace paths, synchronous `start_profile()` blocking
  the actor event loop). Also writing the smallest possible stock-vs-ATO Kineto trace
  diff script (prefer `torch.profiler`'s own APIs over porting the heavier alola
  analysis scripts).
- **ATO verification agent**: confirm the override set is actually compatible with
  Qwen3-14B (rope convention match, `qk_norm_rope`'s head_dim/norm-type gate, compile
  default, exact CLI override-import syntax) before we spend a real 8-GPU run on it.

Meanwhile (orchestrator): reserve the node against further preemption, rebuild
`titanrl:mi355-rocm` (lost when `174023` was preempted), finish Phase 1's remaining
`bench.e2e` runs.

### Known risk carried into Phase 2

ATO's own `scripts/run_titanrl.sh` states, as of 2026-09-18, that **no TitanRL training
step has ever completed** on this path -- needs `attn_backend="flex"` and
`spmd_backend="partial_dtensor"` specifically. Getting a stock run green is a real,
separate bring-up task before any A/B comparison means anything.
