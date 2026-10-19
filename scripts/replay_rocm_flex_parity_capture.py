#!/usr/bin/env python3
"""Sweep contiguous FlexAttention options against a saved trainer/vLLM capture."""

import argparse
import math
from pathlib import Path

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--skip-paged",
        action="store_true",
        help="skip the memory-intensive exact paged-cache replay",
    )
    return parser.parse_args()


def compare(name: str, output: torch.Tensor, reference: torch.Tensor) -> None:
    output = output.squeeze(0).transpose(0, 1).cpu()
    reference = reference[: output.shape[0]]
    unequal = output != reference
    max_delta = (output.float() - reference.float()).abs().max().item()
    print(
        f"{name}: equal={torch.equal(output, reference)}, "
        f"num_diff={unequal.sum().item()}/{output.numel()}, "
        f"max_delta={max_delta:.9e}"
    )
    if unequal.any():
        first_flat = unequal.flatten().nonzero()[0]
        first_index = torch.unravel_index(first_flat, output.shape)
        print(
            f"  first_diff={tuple(index.item() for index in first_index)}, "
            f"output={output[first_index].item():.9e}, "
            f"reference={reference[first_index].item():.9e}"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("This replay requires a ROCm GPU")

    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    num_tokens = capture["prompt"].numel()
    device = torch.device("cuda")
    q = capture["generator_q"][:num_tokens].to(device).transpose(0, 1).unsqueeze(0)
    k = capture["generator_k"][:num_tokens].to(device).transpose(0, 1).unsqueeze(0)
    v = capture["generator_v"][:num_tokens].to(device).transpose(0, 1).unsqueeze(0)
    trainer_reference = capture["trainer_output"]
    generator_reference = capture["generator_output"]

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch._dynamo.config.recompile_limit = 32

    def causal_mask(_batch, _head, query_index, kv_index):
        return kv_index <= query_index

    compiled_flex = torch.compile(flex_attention, fullgraph=True)
    masks = {
        block_size: create_block_mask(
            causal_mask,
            1,
            None,
            num_tokens,
            num_tokens,
            device=device,
            BLOCK_SIZE=(block_size, block_size),
            separate_full_blocks=False,
        )
        for block_size in (16, 128)
    }
    option_sets = {
        "trainer": {
            "BACKEND": "TRITON",
            "BLOCK_M": 16,
            "BLOCK_N": 16,
        },
        "trainer_is_divisible_false": {
            "BACKEND": "TRITON",
            "BLOCK_M": 16,
            "BLOCK_N": 16,
            "IS_DIVISIBLE": False,
        },
        "vllm": {
            "FORCE_USE_FLEX_ATTENTION": True,
            "BLOCK_M": 16,
            "BLOCK_N": 16,
            "IS_DIVISIBLE": False,
        },
    }

    for block_size, block_mask in masks.items():
        for option_name, kernel_options in option_sets.items():
            name = f"block={block_size}/{option_name}"
            output = compiled_flex(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=float(capture["scale"]),
                enable_gqa=bool(capture["enable_gqa"]),
                kernel_options=kernel_options,
            )
            compare(f"{name} vs trainer", output, trainer_reference)
            compare(f"{name} vs vLLM", output, generator_reference)

    if args.skip_paged:
        return

    cache_block_size = int(capture["cache_block_size"])
    physical_blocks = capture["physical_blocks"].to(device).long()
    cache_tokens = int(capture["cache_num_blocks"]) * cache_block_size
    required_blocks = math.ceil(num_tokens / cache_block_size)
    physical_blocks = physical_blocks[:required_blocks]
    paged_k = torch.zeros(
        1,
        k.shape[1],
        cache_tokens,
        k.shape[-1],
        dtype=k.dtype,
        device=device,
    )
    paged_v = torch.zeros_like(paged_k)

    packed_cache_blocks = capture["packed_cache_blocks"].to(device)
    packed_cache_blocks = packed_cache_blocks.transpose(1, 2)
    cached_k, cached_v = packed_cache_blocks.split(k.shape[-1], dim=-1)
    for block_index, physical_block in enumerate(physical_blocks.tolist()):
        physical_start = physical_block * cache_block_size
        physical_end = physical_start + cache_block_size
        paged_k[:, :, physical_start:physical_end] = cached_k[block_index].permute(
            1, 0, 2
        )
        paged_v[:, :, physical_start:physical_end] = cached_v[block_index].permute(
            1, 0, 2
        )

    vllm_physical_to_logical = capture["physical_to_logical"].to(device)
    doc_ids = capture["doc_ids"].to(device)
    seq_lens = capture["seq_lens"].to(device)
    query_start_loc = capture["query_start_loc"].to(device)
    decode_offset = capture["decode_offset"].to(device)

    def paged_causal_mask(_batch, _head, query_index, physical_kv_index):
        query_request = doc_ids[query_index]
        physical_kv_block = physical_kv_index // cache_block_size
        physical_kv_offset = physical_kv_index % cache_block_size
        logical_block = vllm_physical_to_logical[
            query_request, physical_kv_block
        ]
        logical_kv_index = logical_block * cache_block_size + physical_kv_offset
        valid = (
            (logical_block >= 0)
            & (logical_kv_index >= 0)
            & (logical_kv_index < seq_lens[query_request])
        )
        logical_query_index = (
            query_index
            - query_start_loc[query_request]
            + decode_offset[query_request]
        )
        return valid & (logical_kv_index <= logical_query_index)

    kv_indices = capture["blockmask_kv_indices"].to(device)
    kv_num_blocks = capture["blockmask_kv_num_blocks"].to(device)
    block_size = tuple(int(item) for item in capture["blockmask_block_size"])
    logical_by_physical_token = torch.full(
        (cache_tokens,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    for physical_block in physical_blocks.tolist():
        physical_start = physical_block * cache_block_size
        physical_end = physical_start + cache_block_size
        logical_block = vllm_physical_to_logical[0, physical_block]
        logical_by_physical_token[physical_start:physical_end] = (
            logical_block * cache_block_size
            + torch.arange(cache_block_size, device=device)
        )

    def token_lookup_mask(_batch, _head, query_index, physical_kv_index):
        logical_kv_index = logical_by_physical_token[physical_kv_index]
        return (logical_kv_index >= 0) & (logical_kv_index <= query_index)

    def block_lookup_mask(_batch, _head, query_index, physical_kv_index):
        physical_kv_block = physical_kv_index // cache_block_size
        physical_kv_offset = physical_kv_index % cache_block_size
        logical_block = vllm_physical_to_logical[0, physical_kv_block]
        logical_kv_index = logical_block * cache_block_size + physical_kv_offset
        return (logical_block >= 0) & (logical_kv_index <= query_index)

    def request_lookup_mask(_batch, _head, query_index, physical_kv_index):
        query_request = doc_ids[query_index]
        physical_kv_block = physical_kv_index // cache_block_size
        physical_kv_offset = physical_kv_index % cache_block_size
        logical_block = vllm_physical_to_logical[
            query_request, physical_kv_block
        ]
        logical_kv_index = logical_block * cache_block_size + physical_kv_offset
        return (logical_block >= 0) & (logical_kv_index <= query_index)

    # This capture has one request; unused persistent rows contain garbage.
    global_physical_to_logical = vllm_physical_to_logical[0]

    def request_with_global_map(_batch, _head, query_index, physical_kv_index):
        query_request = doc_ids[query_index]
        physical_kv_block = physical_kv_index // cache_block_size
        physical_kv_offset = physical_kv_index % cache_block_size
        logical_block = global_physical_to_logical[physical_kv_block]
        logical_kv_index = logical_block * cache_block_size + physical_kv_offset
        logical_query_index = (
            query_index
            - query_start_loc[query_request]
            + decode_offset[query_request]
        )
        return (logical_block >= 0) & (logical_kv_index <= logical_query_index)

    valid_mapping = vllm_physical_to_logical[:1] >= 0
    request_bits = torch.ones(
        (1, 1),
        dtype=torch.int64,
        device=device,
    ) << torch.arange(1, dtype=torch.int64, device=device)[:, None]
    physical_request_mask = torch.where(
        valid_mapping,
        request_bits,
        0,
    ).sum(dim=0)

    def request_bitmask_global_map(_batch, _head, query_index, physical_kv_index):
        query_request = doc_ids[query_index]
        physical_kv_block = physical_kv_index // cache_block_size
        physical_kv_offset = physical_kv_index % cache_block_size
        logical_block = global_physical_to_logical[physical_kv_block]
        logical_kv_index = logical_block * cache_block_size + physical_kv_offset
        logical_query_index = (
            query_index
            - query_start_loc[query_request]
            + decode_offset[query_request]
        )
        request_bit = torch.ones_like(query_request) << query_request
        request_owns_page = (
            physical_request_mask[physical_kv_block] & request_bit
        ) != 0
        return (
            (logical_block >= 0)
            & request_owns_page
            & (logical_kv_index <= logical_query_index)
        )

    flattened_physical_to_logical = vllm_physical_to_logical.flatten()
    physical_mapping_width = vllm_physical_to_logical.shape[1]

    def flattened_request_lookup(_batch, _head, query_index, physical_kv_index):
        query_request = doc_ids[query_index]
        physical_kv_block = physical_kv_index // cache_block_size
        physical_kv_offset = physical_kv_index % cache_block_size
        flattened_index = (
            query_request * physical_mapping_width + physical_kv_block
        )
        logical_block = flattened_physical_to_logical[flattened_index]
        logical_kv_index = logical_block * cache_block_size + physical_kv_offset
        return (logical_block >= 0) & (logical_kv_index <= query_index)

    for mask_name, mask_mod in (
        ("token_lookup", token_lookup_mask),
        ("block_lookup", block_lookup_mask),
        ("request_with_global_map", request_with_global_map),
        ("request_bitmask_global_map", request_bitmask_global_map),
        ("flattened_request_lookup", flattened_request_lookup),
        ("request_lookup", request_lookup_mask),
        ("exact_vllm", paged_causal_mask),
    ):
        paged_mask = BlockMask.from_kv_blocks(
            seq_lengths=(num_tokens, cache_tokens),
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=None,
            full_kv_indices=None,
            BLOCK_SIZE=block_size,
            mask_mod=mask_mod,
            compute_q_blocks=False,
        )
        paged_output = compiled_flex(
            q,
            paged_k,
            paged_v,
            block_mask=paged_mask,
            scale=float(capture["scale"]),
            enable_gqa=bool(capture["enable_gqa"]),
            kernel_options=option_sets["vllm"],
        )
        compare(
            f"paged {mask_name} vs trainer",
            paged_output,
            trainer_reference,
        )
        compare(
            f"paged {mask_name} vs vLLM",
            paged_output,
            generator_reference,
        )


if __name__ == "__main__":
    main()
