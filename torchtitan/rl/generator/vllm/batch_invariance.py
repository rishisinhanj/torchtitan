# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime helper for bitwise trainer/generator numerics parity."""

import logging

import torch

logger = logging.getLogger(__name__)


def force_flex_page_mapping_for_batch_invariance() -> None:
    """Use request-independent page lookup in vLLM's paged Flex mask.

    vLLM normally indexes ``physical_to_logical[request, physical_block]``
    inside the FlexAttention mask. On ROCm that request-dependent 2-D lookup
    changes the compiled kernel's floating-point accumulation enough to differ
    from contiguous trainer FlexAttention by one BF16 ULP, even when Q/K/V and
    the reconstructed cache are bitwise identical.

    Collapse the table to a global per-page logical-index vector and a request
    ownership bitmask before compiling the mask. The ownership mask preserves
    shared prefix-cache pages. The resulting mask is logically equivalent but
    emits the same attention values as the trainer's contiguous Flex path.
    """
    from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

    if getattr(FlexAttentionMetadata, "_torchtitan_global_page_map", False):
        return

    def get_paged_mask_mod(self):
        if self.num_reqs > 64:
            raise ValueError(
                "ROCm FlexAttention batch invariance supports at most 64 "
                f"concurrent requests, got {self.num_reqs}"
            )

        active_mapping = self.physical_to_logical[: self.num_reqs]
        valid_mapping = active_mapping >= 0
        physical_to_logical = torch.where(
            valid_mapping,
            active_mapping,
            -1,
        ).amax(dim=0)
        request_indices = torch.arange(
            self.num_reqs,
            device=active_mapping.device,
            dtype=torch.int64,
        )[:, None]
        request_bits = torch.ones_like(request_indices) << request_indices
        physical_request_mask = torch.where(
            valid_mapping,
            request_bits,
            0,
        ).sum(dim=0)
        doc_ids = self.doc_ids
        query_start_loc = self.query_start_loc
        decode_offset = self.decode_offset
        logical_mask_mod = self.logical_mask_mod
        block_size = self.block_size

        def global_page_mask_mod(
            batch_index,
            head_index,
            query_index,
            physical_kv_index,
        ):
            query_request = doc_ids[query_index]
            query_request_i64 = query_request.to(torch.int64)
            physical_kv_block = physical_kv_index // block_size
            physical_kv_offset = physical_kv_index % block_size
            logical_block = physical_to_logical[physical_kv_block]
            logical_kv_index = logical_block * block_size + physical_kv_offset
            logical_query_index = (
                query_index
                - query_start_loc[query_request]
                + decode_offset[query_request]
            )
            request_bit = (
                torch.ones_like(query_request_i64) << query_request_i64
            )
            request_owns_page = (
                physical_request_mask[physical_kv_block] & request_bit
            ) != 0
            return (
                (logical_block >= 0)
                & request_owns_page
                & logical_mask_mod(
                    batch_index,
                    head_index,
                    logical_query_index,
                    logical_kv_index,
                )
            )

        return global_page_mask_mod

    FlexAttentionMetadata.get_paged_mask_mod = get_paged_mask_mod
    FlexAttentionMetadata._torchtitan_global_page_map = True
    logger.info(
        "Patched vLLM paged FlexAttention to use batch-invariant global page mapping"
    )


def force_logprobs_fn_for_batch_invariance() -> None:
    """Make vLLM's v2 logprob path dispatch.

    The v2 GPU sampler computes per-token logprobs with a fused Triton kernel
    (``compute_token_logprobs`` -> ``_topk_log_softmax_kernel``) that inlines
    ``log(softmax(logits))`` and never calls PyTorch ops.

    Swapping the kernel routes the generator and the trainer through
    the same set of ops, so logprobs match bit-for-bit. The wrapper supplies
    full logits to vLLM, so the wrapper explicitly selects the replicated path.
    """
    import vllm.v1.worker.gpu.sample.logprob as vllm_logprob

    from torchtitan.components.loss import compute_logprobs

    def generator_compute_token_logprobs(
        logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Per-token logprobs for vLLM's v2 sampler (replaces its fused kernel).

        Args:
            logits: Full-vocabulary ``[N, V]`` next-token logits for N sampled
                positions, already gathered by the vLLM model wrapper.
            token_ids: ``[N, K]`` the K token ids to score per position (vLLM
                passes the sampled token's logprob plus any top-k logprobs it requested).

        Returns:
            ``[N, K]`` logprob of each of the K token ids at each position.
        """
        # vLLM gives token_ids [N, K]. SamplingParams(logprobs=0) makes real
        # requests K=1, but we can't assert that: vLLM's kernel warmup probes
        # this patched fn with K>1 (e.g. K=6), so we must handle any K. Reuse
        # the trainer's compute_logprobs once per column.
        #
        # NOTE: each element of token_ids is scored independently, after the
        # whole generated sequence is marterialized. We iterate column-by-column
        # purely because torchtitan's compute_logprobs takes one token id per
        # position; it is not a cross-column/cross-position dependency.
        token_ids = token_ids.to(torch.int64)
        per_column = [
            compute_logprobs(
                logits,
                token_ids[:, k],
                vocab_parallel_group=None,
            )
            for k in range(token_ids.shape[1])
        ]
        return torch.stack(per_column, dim=-1)  # [N, K]

    vllm_logprob.compute_token_logprobs = generator_compute_token_logprobs
    logger.info(
        "Patched vLLM compute_token_logprobs with trainer's implementation "
        "so generator and trainer share one logprob code path"
    )
