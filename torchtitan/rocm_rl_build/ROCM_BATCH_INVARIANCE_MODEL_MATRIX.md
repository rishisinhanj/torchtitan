# ROCm TitanRL Batch-Invariance Model Matrix

## Result

TitanRL produced bitwise-identical trainer and vLLM generator log probabilities
for 25 optimizer steps on Qwen3-0.6B, 1.7B, 4B, and 8B:

```text
bit_wise/logprob_diff/max == 0
```

Every completed step in every run, including the two partial runs, reported
zero maximum log-probability difference. No run had a parity failure, HSA
fault, or out-of-memory failure.

## Environment

- Host: `gbt350-odcdh2-b05-1`
- GPU: gfx950 (`AMD Instinct MI350X`, 251.98 GiB per GPU)
- Torch: `2.16.0.dev20261018+rocm10.0`
- Triton: `3.9.0+gitfc8d31a5`
- ROCm userspace: `10.0.0.0-9999-6b0e43f3`
- vLLM: `0.32.0rc1.dev369+g1b2a66639.rocm100`
- Attention: plain compiled FlexAttention
- Trainer tensor parallelism: 2
- Generator tensor parallelism: 2
- vLLM graph mode: `NONE` (`enforce_eager=True`)
- W&B project: `rocm_batch_invariance`

The validation recipe is `scripts/rocm_flex_bi_long_run.py`; its Docker runner
is `scripts/run_rocm_flex_bi_long_run.sh`.

## Successful 25-step runs

### Qwen3-0.6B

- Sampling: 4 samples per prompt, temperature 0.8
- Result: 25/25 steps; all 25 maximum log-probability differences were zero
- W&B: [mk6j4qvu](https://forge.coreweave.com/wandb/rissinha-amd/rocm_batch_invariance/runs/mk6j4qvu)
- Log: `outputs/rocm_flex_bi_qwen3_0.6b_25step/long_run.log`

### Qwen3-1.7B

- Sampling: 8 samples per prompt, temperature 1.5
- Result: 25/25 steps; all 25 maximum log-probability differences were zero
- W&B: [dj122pxg](https://forge.coreweave.com/wandb/rissinha-amd/rocm_batch_invariance/runs/dj122pxg)
- Log: `outputs/rocm_flex_bi_qwen3_1.7b_25step_diverse/long_run.log`

### Qwen3-4B

- Sampling: 16 samples per prompt, temperature 2.0
- Result: 25/25 steps; all 25 maximum log-probability differences were zero
- W&B: [gsbwh33o](https://forge.coreweave.com/wandb/rissinha-amd/rocm_batch_invariance/runs/gsbwh33o)
- Log: `outputs/rocm_flex_bi_qwen3_4b_25step_diverse/long_run.log`

### Qwen3-8B

- Sampling: 16 samples per prompt, temperature 2.0
- Result: 25/25 steps; all 25 maximum log-probability differences were zero
- W&B: [wbzall1v](https://forge.coreweave.com/wandb/rissinha-amd/rocm_batch_invariance/runs/wbzall1v)
- Log: `outputs/rocm_flex_bi_qwen3_8b_25step_diverse/long_run.log`

## Partial and infrastructure-failed attempts

### Qwen3-1.7B initial attempt

- Result: failed before model startup because copying downloaded assets into
  the Docker writable layer exhausted disk space
- W&B: [0m9grbhw](https://forge.coreweave.com/wandb/rissinha-amd/rocm_batch_invariance/runs/0m9grbhw)
- Log: `outputs/rocm_flex_bi_qwen3_1.7b_25step/long_run.log`
- Resolution: the runner now excludes `assets` and `outputs` from its source
  archive and bind-mounts the selected assets

### Qwen3-1.7B low-diversity attempt

- Sampling: 4 samples per prompt, temperature 0.8
- Result: 6/25 steps; all six maximum log-probability differences were zero
- Stop reason: ten consecutive untrainable batches with no reward diversity
- W&B: [6hezjxdg](https://forge.coreweave.com/wandb/rissinha-amd/rocm_batch_invariance/runs/6hezjxdg)
- Log: `outputs/rocm_flex_bi_qwen3_1.7b_25step_retry/long_run.log`

### Qwen3-4B low-diversity attempt

- Sampling: 8 samples per prompt, temperature 1.5
- Result: 22/25 steps; all 22 maximum log-probability differences were zero
- Stop reason: ten consecutive untrainable batches with no reward diversity
- W&B: [nc63td94](https://forge.coreweave.com/wandb/rissinha-amd/rocm_batch_invariance/runs/nc63td94)
- Log: `outputs/rocm_flex_bi_qwen3_4b_25step/long_run.log`

The two workload aborts are reward-diversity failures in the alphabet-sort
training task. They are not numerical parity failures. Increasing sample count
and temperature allowed both model sizes to complete.

## What the runtime exercised

The runs exercised the full TitanRL loop:

1. vLLM generated rollouts with paged FlexAttention.
2. The generator returned token log probabilities.
3. The trainer ran continuous compiled FlexAttention on the same tokens.
4. TitanRL compared trainer and generator token log probabilities.
5. The trainer performed backward, optimizer update, and weight transfer.
6. vLLM reset its KV cache and generated with the updated weights.

Strict on-policy operation (`target_offpolicy_steps=0`) ensured that each
comparison used matching weights. The vLLM path installed the shared token
log-probability implementation and the batch-invariant global page mapping.
Runtime logs confirmed both patches were active.

## Operational notes and limits

- Graph replay was disabled deliberately. These results do not establish
  parity with vLLM CUDA/HIP graph replay enabled.
- The paged-attention owner map currently supports at most 64 simultaneous
  requests. The successful runs used 8, 16, or 32 requests.
- Missing `numactl` emitted nonfatal tracebacks; execution explicitly continued.
- W&B warned that a post-validation metric at step 25 followed training step
  26. Training parity metrics were still synchronized.
- Periodic full optimizer checkpoints initially consumed excessive disk.
  The validation recipe now suppresses intermediate checkpoints and saves only
  a final model-only checkpoint.
- Generated checkpoints were removed after validation; logs and W&B records
  were retained.

## Conclusion

On this ROCm 10.0 gfx950 environment, the branch's compiled-Flex ROCm
batch-invariance changes preserve zero trainer-vLLM log-probability drift across
25 optimizer and weight-transfer cycles for four plain-attention Qwen3 sizes
from 0.6B through 8B.

This conclusion is limited to the tested toolchain, plain-attention Qwen3
models, tensor parallelism degree 2, eager vLLM execution, and the tested
request counts. It does not cover ROCm 7.2, GDN models, graph replay, or more
than 64 simultaneous requests.
