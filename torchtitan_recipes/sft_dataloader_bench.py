# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only dataloader comparison for the multi-turn vs single-turn SFT study.

The GPU trace cannot answer how much the multi-turn renderer path costs:
rendering runs on the Grain prefetch thread, it issues no torch ops, and the
profiler is built with ``with_stack=False``, so Kineto records nothing for it.
This harness builds each recipe's dataloader with no model, drains a fixed
number of microbatches, and reports producer throughput alongside the packing
statistics that the turn count actually moves.

Run from the repository root::

    python -m torchtitan_recipes.sft_dataloader_bench
"""

import time
from dataclasses import dataclass

import numpy as np

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.trainer import Trainer

from torchtitan_recipes.sft_multiturn_profile import (
    sft_qwen3_8b_multiturn_profiled,
    sft_qwen3_8b_singleturn_profiled,
)


_NUM_WARMUP_BATCHES = 5
_NUM_MEASURED_BATCHES = 50


@dataclass(kw_only=True, slots=True)
class BenchResult:
    """Producer timing and packing statistics for one recipe."""

    batch_seconds: np.ndarray
    tokens_per_row: int
    real_tokens_per_row: np.ndarray
    supervised_tokens_per_row: np.ndarray
    documents_per_row: np.ndarray
    document_lengths: np.ndarray


def _document_lengths(positions: np.ndarray, num_real_tokens: int) -> np.ndarray:
    """Split a packed row into per-document lengths.

    Packing concatenates documents at the front of the row and pads the tail,
    and each document restarts ``positions`` at zero, so the zeros within the
    real prefix are the document boundaries.
    """
    starts = np.flatnonzero(positions[:num_real_tokens] == 0)
    return np.diff(np.append(starts, num_real_tokens))


def measure(config: Trainer.Config) -> BenchResult:
    """Drain microbatches from a recipe's dataloader and collect statistics."""
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    dataloader = config.dataloader.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        max_context_length=config.training.max_context_length,
        num_tokens_per_microbatch=(
            config.training.num_tokens_per_microbatch_per_dp_rank
        ),
    )

    batch_seconds: list[float] = []
    real_tokens: list[int] = []
    supervised_tokens: list[int] = []
    documents: list[int] = []
    lengths: list[np.ndarray] = []
    tokens_per_row = 0

    iterator = iter(dataloader)
    for index in range(_NUM_WARMUP_BATCHES + _NUM_MEASURED_BATCHES):
        start = time.perf_counter()
        batch = next(iterator)
        elapsed = time.perf_counter() - start
        if index < _NUM_WARMUP_BATCHES:
            continue

        padding_mask = batch.padding_mask.numpy()
        num_real = int((~padding_mask).sum())
        row_lengths = _document_lengths(batch.positions.numpy(), num_real)

        batch_seconds.append(elapsed)
        tokens_per_row = padding_mask.size
        real_tokens.append(num_real)
        supervised_tokens.append(int((batch.labels.numpy() != IGNORE_INDEX).sum()))
        documents.append(len(row_lengths))
        lengths.append(row_lengths)

    return BenchResult(
        batch_seconds=np.asarray(batch_seconds),
        tokens_per_row=tokens_per_row,
        real_tokens_per_row=np.asarray(real_tokens),
        supervised_tokens_per_row=np.asarray(supervised_tokens),
        documents_per_row=np.asarray(documents),
        document_lengths=np.concatenate(lengths),
    )


def _summarize(result: BenchResult) -> dict[str, float]:
    seconds = result.batch_seconds
    real = result.real_tokens_per_row
    lengths = result.document_lengths
    return {
        "rows/s": len(seconds) / seconds.sum(),
        "batch ms mean": 1e3 * seconds.mean(),
        "batch ms p50": 1e3 * float(np.percentile(seconds, 50)),
        "batch ms p99": 1e3 * float(np.percentile(seconds, 99)),
        "tokens/row": result.tokens_per_row,
        "real tokens/row": real.mean(),
        "padding fraction": 1.0 - real.mean() / result.tokens_per_row,
        "supervised fraction": (
            result.supervised_tokens_per_row.mean() / result.tokens_per_row
        ),
        "documents/row": result.documents_per_row.mean(),
        "doc len mean": lengths.mean(),
        "doc len std": lengths.std(),
        "doc len p50": float(np.percentile(lengths, 50)),
        "doc len p99": float(np.percentile(lengths, 99)),
    }


def main() -> None:
    """Benchmark the multi-turn subject against the single-turn control."""
    summaries = {
        "multiturn": _summarize(measure(sft_qwen3_8b_multiturn_profiled())),
        "singleturn": _summarize(measure(sft_qwen3_8b_singleturn_profiled())),
    }
    print(f"{'metric':<22}{'multiturn':>14}{'singleturn':>14}{'ratio':>10}")
    for metric in summaries["multiturn"]:
        subject = summaries["multiturn"][metric]
        control = summaries["singleturn"][metric]
        ratio = subject / control if control else float("nan")
        print(f"{metric:<22}{subject:>14.3f}{control:>14.3f}{ratio:>10.2f}")


if __name__ == "__main__":
    main()
