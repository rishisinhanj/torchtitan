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
    python -m torchtitan_recipes.sft_dataloader_bench --arms matched control
"""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.trainer import Trainer

from torchtitan_recipes.sft_multiturn_profile import (
    sft_qwen3_8b_multiturn_profiled,
    sft_qwen3_8b_multiturn_token_matched_profiled,
    sft_qwen3_8b_singleturn_profiled,
)


_NUM_WARMUP_BATCHES = 5
_NUM_MEASURED_BATCHES = 200

_CONTROL_ARM = "control"
_ARMS: dict[str, Callable[[], Trainer.Config]] = {
    "multiturn": sft_qwen3_8b_multiturn_profiled,
    "matched": sft_qwen3_8b_multiturn_token_matched_profiled,
    _CONTROL_ARM: sft_qwen3_8b_singleturn_profiled,
}


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


def measure(
    config: Trainer.Config,
    *,
    num_batches: int = _NUM_MEASURED_BATCHES,
    num_warmup_batches: int = _NUM_WARMUP_BATCHES,
) -> BenchResult:
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
    try:
        for index in range(num_warmup_batches + num_batches):
            start = time.perf_counter()
            batch = next(iterator)
            elapsed = time.perf_counter() - start
            if index < num_warmup_batches:
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
    finally:
        # The prefetch thread keeps rendering under the GIL otherwise, biasing
        # whichever arm is measured next.
        dataloader.close()

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
        # A fixed token budget per row means longer documents pack fewer per
        # row, so rows/s answers a capacity question while ms/document
        # isolates per-sample cost. wall s is a tripwire: once it drifts from
        # the loop's own elapsed time the producer is no longer the bottleneck
        # and these timings stop meaning anything.
        "rows/s": len(seconds) / seconds.sum(),
        "ms/document": 1e3 * seconds.mean() / result.documents_per_row.mean(),
        "us/real token": 1e6 * seconds.mean() / real.mean(),
        "wall s": seconds.sum(),
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


def _report(summaries: dict[str, dict[str, float]]) -> None:
    """Print one column per arm, plus a ratio column against the control."""
    arms = list(summaries)
    others = [arm for arm in arms if arm != _CONTROL_ARM]
    ratio_arms = others if _CONTROL_ARM in summaries else []

    ratio_labels = {arm: f"{arm}/ctl" for arm in ratio_arms}
    width = max((len(label) for label in ratio_labels.values()), default=0) + 2

    header = f"{'metric':<22}" + "".join(f"{arm:>13}" for arm in arms)
    header += "".join(f"{ratio_labels[arm]:>{width}}" for arm in ratio_arms)
    print(header)
    for metric in summaries[arms[0]]:
        row = f"{metric:<22}" + "".join(
            f"{summaries[arm][metric]:>13.3f}" for arm in arms
        )
        for arm in ratio_arms:
            base = summaries[_CONTROL_ARM][metric]
            ratio = summaries[arm][metric] / base if base else float("nan")
            row += f"{ratio:>{width}.2f}"
        print(row)


def main() -> None:
    """Benchmark the selected recipes' dataloaders and print a comparison.

    Read ``rows/s`` across {multiturn, control} for capacity and
    ``ms/document`` across {matched, control} for attribution; the other
    pairings mix a length effect with a structure effect. Timing columns are
    noisy run to run, so lean on the structural ones.
    """
    parser = argparse.ArgumentParser(
        description="Compare SFT dataloader throughput and packing across arms."
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list(_ARMS),
        default=list(_ARMS),
        help="Recipes to measure; the control enables the ratio columns.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=_NUM_MEASURED_BATCHES,
        help="Microbatches timed per arm after warmup.",
    )
    parser.add_argument(
        "--num-warmup-batches",
        type=int,
        default=_NUM_WARMUP_BATCHES,
        help="Microbatches drained before timing starts.",
    )
    args = parser.parse_args()

    _report(
        {
            arm: _summarize(
                measure(
                    _ARMS[arm](),
                    num_batches=args.num_batches,
                    num_warmup_batches=args.num_warmup_batches,
                )
            )
            for arm in args.arms
        }
    )


if __name__ == "__main__":
    main()
