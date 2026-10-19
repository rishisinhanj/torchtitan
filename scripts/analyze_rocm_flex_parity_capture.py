#!/usr/bin/env python3
"""Explain the exact element-level drift in a saved Flex parity capture."""

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "capture",
        type=Path,
        help="flex_parity_rankN.pt written by TestBitwiseParityFlex",
    )
    parser.add_argument(
        "--top-contributors",
        type=int,
        default=10,
        help="number of KV-token contributions to print per differing element",
    )
    return parser.parse_args()


def bf16_bits(value: torch.Tensor) -> int:
    return value.to(torch.bfloat16).view(torch.uint16).item()


def main() -> None:
    args = parse_args()
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    num_tokens = capture["prompt"].numel()

    for name in ("q", "k", "v"):
        trainer = capture[f"trainer_{name}"][:num_tokens]
        generator = capture[f"generator_{name}"][:num_tokens]
        reconstructed = capture.get(f"reconstructed_paged_{name}")
        print(f"{name.upper()} trainer-vLLM equal: {torch.equal(trainer, generator)}")
        if reconstructed is not None:
            print(
                f"{name.upper()} vLLM-cache equal: "
                f"{torch.equal(generator, reconstructed[:num_tokens])}"
            )

    trainer_output = capture["trainer_output"][:num_tokens]
    generator_output = capture["generator_output"][:num_tokens]
    contiguous_replay = capture["contiguous_replay"][:num_tokens]
    unequal = trainer_output != generator_output
    differing_indices = unequal.nonzero()
    max_delta = (
        (trainer_output.float() - generator_output.float()).abs().max().item()
    )
    print(
        "contiguous replay equals trainer: "
        f"{torch.equal(contiguous_replay, trainer_output)}"
    )
    print(
        f"attention differences: {differing_indices.shape[0]}/"
        f"{trainer_output.numel()}, max_delta={max_delta:.9e}"
    )
    print(f"physical cache blocks: {capture['physical_blocks'].tolist()}")
    print(f"cache block size: {capture['cache_block_size']}")

    q = capture["generator_q"][:num_tokens]
    k = capture["generator_k"][:num_tokens]
    v = capture["generator_v"][:num_tokens]
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    queries_per_kv = q_heads // kv_heads
    scale = float(capture["scale"])

    for index in differing_indices:
        token, query_head, value_dim = (int(item) for item in index)
        kv_head = query_head // queries_per_kv
        trainer_value = trainer_output[token, query_head, value_dim]
        generator_value = generator_output[token, query_head, value_dim]

        query = q[token, query_head].double()
        keys = k[: token + 1, kv_head].double()
        values = v[: token + 1, kv_head, value_dim].double()
        scores = torch.mv(keys, query) * scale
        probabilities = torch.softmax(scores, dim=0)
        contributions = probabilities * values
        fp64_reference = contributions.sum()
        fp32_reference = (
            torch.softmax(
                torch.mv(
                    k[: token + 1, kv_head].float(),
                    q[token, query_head].float(),
                )
                * scale,
                dim=0,
            )
            * v[: token + 1, kv_head, value_dim].float()
        ).sum()
        rounded_reference = fp64_reference.to(torch.bfloat16)

        print()
        print(
            f"index=(token={token}, query_head={query_head}, "
            f"kv_head={kv_head}, value_dim={value_dim})"
        )
        print(
            f"trainer={trainer_value.item():.9e} "
            f"(bf16_bits=0x{bf16_bits(trainer_value):04x})"
        )
        print(
            f"vLLM={generator_value.item():.9e} "
            f"(bf16_bits=0x{bf16_bits(generator_value):04x})"
        )
        ulp_distance = abs(
            bf16_bits(trainer_value) - bf16_bits(generator_value)
        )
        print(f"bf16 ULP distance: {ulp_distance}")
        print(f"fp64 reference={fp64_reference.item():.12e}")
        print(f"fp32 reference={fp32_reference.item():.12e}")
        print(
            f"fp64 rounded to bf16={rounded_reference.item():.9e} "
            f"(bf16_bits=0x{bf16_bits(rounded_reference):04x})"
        )

        top_count = min(args.top_contributors, contributions.numel())
        top = contributions.abs().topk(top_count).indices
        print("largest |softmax(score) * V| contributions:")
        for key_token in top.tolist():
            print(
                f"  kv_token={key_token:3d} "
                f"score={scores[key_token].item(): .8e} "
                f"prob={probabilities[key_token].item(): .8e} "
                f"value={values[key_token].item(): .8e} "
                f"contribution={contributions[key_token].item(): .8e}"
            )


if __name__ == "__main__":
    main()
