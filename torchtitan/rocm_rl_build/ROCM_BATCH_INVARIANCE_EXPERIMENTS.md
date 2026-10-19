# ROCm Batch-Invariance Experiments

This document records the experiments performed while investigating
batch-invariant TitanRL with compiled FlexAttention on ROCm.

## Goal

Enable the TorchTitan trainer and vLLM generator to produce bitwise-identical
log probabilities on ROCm for plain-attention models, ultimately requiring:

```text
bit_wise/logprob_diff/max == 0
```

The intended upstream deliverable is a small TorchTitan change comparable to
the existing CUDA batch-invariant path. Docker changes are validation
infrastructure, not the primary deliverable.

## Hardware and software

The end-to-end experiments were run on gfx950 (`AMD Instinct MI350X`) on
`gbt350-odcdh2-b05-1`.

The validated ROCm 10 environment used:

- Torch `2.16.0.dev20261018+rocm10.0`
- Triton ROCm `3.9.0+gitfc8d31a5`
- vLLM `0.32.0rc1.dev369+g1b2a66639.rocm100`
- Qwen3-0.6B
- Two GPUs with tensor parallelism degree 2

The vLLM native extensions were rebuilt from commit `1b2a66639` against the
Torch 2.16 nightly.

## Compiled FlexAttention determinism

The original gfx950 backward nondeterminism reproducer was tested on the same
node and shapes across three toolchains.

| Toolchain | Result |
|---|---|
| Torch `2.15.0.dev20260827+rocm7.2`, Triton 3.8 | Nondeterministic within 10 iterations |
| Torch `2.16.0.dev20261012+rocm10.0`, Triton 3.9 | 300 iterations bitwise exact |
| Torch `2.16.0.dev20261012+rocm10.1`, Triton 3.9 | 300 iterations bitwise exact |

The ROCm 10 tests included:

- Original `B=8, H=16, S=2048, D=16, bf16` reproducer.
- Batch-invariant emission with `D=128`, `BLOCK_M=16`, `BLOCK_N=16`, and
  `BACKEND=TRITON`.
- Higher occupancy at `B=16`.
- Sequence length 4096.
- fp16.

`TORCH_LOGS=output_code` confirmed that the deterministic ROCm 10 runs still
emitted the compiled Triton FlexAttention forward and backward kernels. The
result was not caused by switching to another backend.

The exact component that fixed the race has not been bisected. Torch, Triton,
and ROCm userspace all changed together.

## Batch-invariant operator coverage

Initial testing of `batch_invariant_ops` on gfx950 found:

- `mm`: batch invariant and stable across 30 runs.
- `bmm`: failed at batch size 2 when only the upstream package was installed.

TorchTitan now supplements the upstream package with vLLM's
`bmm_batch_invariant` implementation and fixed-schedule `sum.dim_IntList`.
The earlier lack of a `bmm` override is therefore no longer a current
TorchTitan limitation.

## Trainer compiled-Flex change

`set_determinism()` previously forced eager FlexAttention for every ROCm run.
The experimental branch preserves eager Flex for ordinary deterministic ROCm
runs but uses compiled Flex when `debug.batch_invariant=True`.

The focused CPU tests verify:

- ROCm plus batch invariance selects compiled FlexAttention.
- ROCm deterministic mode without batch invariance retains eager FlexAttention.
- Max autotune and coordinate-descent tuning remain disabled.

Result:

```text
7 passed
```

## Docker and packaging findings

The moving ROCm vLLM base image required several compatibility fixes before an
end-to-end test could start.

### Torch installation

An ordinary editable TorchTitan installation replaced the ROCm nightly Torch
with a stable CUDA build. The image now installs TorchTitan with the ROCm 10
nightly index available during dependency resolution.

### Overlapping Triton packages

The base image contained both:

- `triton 3.8.0+git669b31ac`
- `triton-rocm 3.9.0+gitfc8d31a5`

Both distributions install files under the same `triton` package. Removing
both and reinstalling the exact `triton-rocm` requirement declared by Torch
fixed the mixed installation.

### vLLM import segmentation fault

vLLM promotes `libtorch_cpu.so` with `RTLD_GLOBAL` before importing Torch to
enable ROCm profiler symbol discovery. With the Torch 2.16 ROCm nightly, this
changed native symbol resolution and caused a segmentation fault while loading
Triton's `libtriton`.

The failure was isolated with:

1. A clean matching Triton installation.
2. Direct `torch._inductor.lowering` import, which succeeded.
3. Early `RTLD_GLOBAL` loading of `libtorch_cpu.so`, after which the same
   Inductor import segfaulted.

Disabling only vLLM's optional ROCm symbol-promotion call fixed the import while
preserving vLLM's Inductor fallback-list patch.

### Native vLLM ABI

The base vLLM native extensions referenced a Torch symbol absent from the
Torch 2.16 nightly:

```text
c10::impl::cow::materialize_cow_storage
```

Rebuilding vLLM against the installed nightly fixed `_C` and `_rocm_C`
loading.

### Torchvision ABI

The base `torchvision 0.27.0` did not match Torch 2.16 and failed because
`torchvision::nms` was not registered. Upgrading to
`torchvision 0.30.0.dev20261018+rocm10.0` fixed this.

### Moving TorchTitan/vLLM APIs

Two compatibility differences were encountered between this TorchTitan
checkout and the tested vLLM source:

- `PretrainedConfig` was not re-exported from
  `vllm.transformers_utils.config`.
- vLLM's `PassConfig` no longer accepted `enable_sp` and
  `sp_min_token_num`.

Disposable runtime shims were used for the experiments. They are packaging/API
compatibility issues, not batch-invariance fixes.

## Full Flex parity experiments

The standard `TestBitwiseParityFlex` suite exercises:

1. Trainer packed-prefill batch invariance.
2. Trainer prefill versus vLLM prefill.
3. vLLM decode versus a second vLLM prefill.

### Baseline: vLLM cache block size 16

| Test | Result |
|---|---|
| Trainer batch invariance | Pass: max delta 0, zero differing tokens |
| Trainer versus vLLM prefill | Fail: max delta `9.45e-2`, 130/149 tokens differ |
| vLLM decode versus prefill | Fail: max delta `4.62e-3`, 33/50 tokens differ |

This is the first direct proof that compiled trainer FlexAttention is
batch-invariant on the tested ROCm stack.

### vLLM batch-invariant flag

`VLLM_BATCH_INVARIANT=1` was enabled and verified. vLLM selected 16x16 final
Flex kernel tiles as intended.

Result: identical failures and deltas to the baseline.

Conclusion: failure to enable the vLLM environment flag was not the parity
cause.

### `num_stages=2`

`num_stages=2` was pinned experimentally on both the trainer and vLLM Flex
kernels.

Result: identical failures and deltas to the baseline.

Conclusion: `num_stages=2` is not required and should not be added to the
shared converter.

### vLLM cache block size 256

This matches the production TorchTitan ROCm generator setting.

| Test | Result |
|---|---|
| Trainer batch invariance | Pass |
| Trainer versus vLLM prefill | Fail: max delta `1.14e-1`, 138/149 tokens differ |
| vLLM decode versus prefill | Pass for all five sequences, bitwise exact |

Conclusion: the production cache block size fixes generator self-parity but
does not fix trainer-to-generator parity.

### Cache and logical block size 128

vLLM's cache block, logical query block, and logical KV block were aligned with
the trainer's 128-token Flex block size. Kernel tiles remained 16x16.

Result:

- Generator self-parity was exact when the cache block covered the tested
  sequence.
- Trainer-to-vLLM parity remained unchanged at max delta `1.14e-1`.

### Logical block size 16 on both sides

The trainer Flex block size was changed to 16 to match vLLM's default paged
path.

Result: identical failures and deltas to the original 16-token vLLM baseline.

Conclusion: logical block-size alignment is not sufficient.

## Intermediate localization

An opt-in, single-sequence diagnostic synchronized trainer weights into vLLM,
disabled vLLM graph replay, and captured every Qwen3 layer on both TP ranks.
With the default vLLM Flex configuration:

- Embeddings, layer inputs, norms, and post-RoPE Q/K/V were bitwise equal
  through layer 2.
- Reconstructing layer 2 K/V from vLLM's block table and packed paged cache
  produced tensors bitwise equal to vLLM's logical K/V inputs on both ranks.
- Replaying those Q/K/V through contiguous compiled Flex was bitwise equal to
  the trainer's padded contiguous Flex output.
- The first unequal value was vLLM's paged Flex output on TP rank 0 at
  `(token=86, local_head=6, dim=91)`: one of 153,600 elements differed by
  `2.441406e-4`.
- TP rank 1's attention output was still equal. Its first drift appeared in
  `wo`, after the TP reduction incorporated rank 0's unequal value.

This rules out weight transfer, embeddings, RoPE, Q/K/V projection, KV-cache
contents, and the trainer's block padding as the first cause. The remaining
failure is specifically the paged Flex computation/metadata path. Aligning the
vLLM cache and logical blocks to 128 moved the first error to layer 0 rather
than fixing it.

### Exact paged replay and fix

The diagnostic now saves rank-local Q/K/V, attention outputs, the used cache
pages, and vLLM's actual `BlockMask` tensors. The saved replay reproduces both
outputs exactly.

The layer-2 mismatch was one BF16 ULP:

```text
index:    token 86, local query head 6, value dimension 91
trainer:  -0.0468750000  (BF16 0xbd40)
vLLM:     -0.0466308594  (BF16 0xbd3f)
FP64 ref: -0.0467443086  (rounds to vLLM's 0xbd3f)
```

The trigger was vLLM's request-dependent two-dimensional lookup inside its
paged mask:

```text
physical_to_logical[query_request, physical_kv_block]
```

Flattening that same lookup still reproduced vLLM's one-ULP result. Replacing
it with logically equivalent global per-page metadata reproduced the trainer
exactly. The fix computes a global physical-page-to-logical-block vector and a
64-bit request-ownership mask before Flex compilation. Shared prefix pages can
belong to multiple requests without reintroducing the request-indexed table.

With that patch, the complete ROCm Flex parity class passed:

- Trainer batch invariance: zero differences.
- Every captured intermediate on both TP ranks: zero differences.
- Trainer versus vLLM prefill for five mixed-length prompts: zero differences.
- vLLM decode versus second prefill for five prompts: zero differences.

```text
4 passed
```

### Two-step TitanRL result

The saved `scripts/run_rocm_flex_bi_two_step.sh` recipe ran two optimizer steps
with two trainer and two generator GPUs. vLLM CUDA-graph replay was disabled
after its first rollout caused a ROCm HSA memory fault; compiled FlexAttention
remained enabled.

```text
Train | Step: 1 ... bit_wise/logprob_diff/max: 0
Train | Step: 2 ... bit_wise/logprob_diff/max: 0
```

Step 2 includes an optimizer update and trainer-to-generator weight transfer,
so this validates more than initial-checkpoint parity.

Final logs from the reviewed implementation:

- Full Flex parity:
  `outputs/rocm_flex_parity_global_map_int64/diagnostic.log`
- Two-step TitanRL:
  `outputs/rocm_flex_bi_two_step_final/two_step.log`

## Current conclusions

1. Compiled trainer FlexAttention is deterministic and batch-invariant on the
   tested ROCm 10 gfx950 stack.
2. The paged Flex drift was caused by request-indexed mask metadata changing
   the ROCm kernel's accumulation, not by incorrect Q/K/V or cache contents.
3. Global page mapping restores complete trainer-vLLM and decode-prefill
   parity for the tested plain-attention model.
4. Two TitanRL steps both report `bit_wise/logprob_diff/max: 0`.
5. ROCm batch invariance can be enabled for FlexAttention while retaining the
   controller rejection for unsupported ROCm attention backends.
6. vLLM CUDA-graph replay remains a separate ROCm runtime issue; the successful
   validation used eager graph execution with compiled FlexAttention.
7. The ownership representation supports at most 64 concurrent requests per
   generator. ROCm Flex batch-invariant startup rejects a larger
   `max_num_seqs` instead of silently returning to the drifting lookup.

## Current upstream patch boundary

The evidence-backed TorchTitan change is:

- Use compiled FlexAttention for ROCm batch-invariant trainer determinism.
- Replace vLLM's request-indexed paged-mask lookup with global page metadata
  only for ROCm Flex batch-invariant generation.
- Preserve the existing eager fallback for ordinary deterministic ROCm runs.
- Narrow the controller's ROCm rejection to non-Flex attention.
- Keep CUDA behavior unchanged.

The Dockerfile and local vLLM wheel are validation infrastructure and should
not be presented as the upstream batch-invariance fix.
