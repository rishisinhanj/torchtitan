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

## Current conclusions

1. Compiled trainer FlexAttention is deterministic and batch-invariant on the
   tested ROCm 10 gfx950 stack.
2. vLLM generator decode and prefill can be bitwise identical on ROCm when
   using the production 256-token cache block.
3. Trainer-to-vLLM prefill parity is still not bitwise exact.
4. The remaining drift is unaffected by:
   - vLLM's batch-invariant environment flag;
   - 16x16 kernel-tile alignment;
   - `num_stages=2`;
   - matching logical Flex block sizes;
   - matching cache and logical block sizes.
5. A full two-step TitanRL run cannot satisfy zero logprob difference while
   the simpler trainer-versus-vLLM prefill test already fails.
6. The blanket ROCm controller rejection should remain until the remaining
   trainer-to-vLLM drift is localized and fixed.

## Current upstream patch boundary

The smallest evidence-backed TorchTitan change is:

- Use compiled FlexAttention for ROCm batch-invariant trainer determinism.
- Preserve the existing eager fallback for ordinary deterministic ROCm runs.
- Add unit coverage for that dispatch decision.
- Keep the controller's ROCm rejection, with an error message describing the
  current trainer-to-vLLM parity limitation.

The Dockerfile and local vLLM wheel are validation infrastructure and should
not be presented as the upstream batch-invariance fix.

## Next diagnostic

Run one matched single-sequence prefill and capture the first differing
intermediate between trainer and vLLM:

1. Embeddings and position tensors.
2. First-layer Q, K, and V after RoPE.
3. Attention output before the output projection.
4. Output projection, residual, and normalization tensors.
5. Final logits and logprobs.

If Q/K/V are bitwise equal but the attention output differs, replay identical
captured Q/K/V through:

- the trainer's contiguous compiled Flex path; and
- vLLM's paged-KV compiled Flex path.

That experiment will distinguish kernel math from KV-cache write/read layout
without further speculative tile or stage changes.
