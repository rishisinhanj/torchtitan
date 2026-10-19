# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Profiled Qwen3-8B GSM8K SFT configurations for a multi-turn vs single-turn study.

Both configurations derive from ``sft_qwen3_8b_math`` and differ only in how the
sample is turned into a conversation: the multi-turn one installs a chat
renderer (which produces its own loss mask over an arbitrary number of turns)
and a ``messages_fn`` that expands each GSM8K question/answer pair into two
user/assistant round trips, while the single-turn one keeps the base recipe's
renderer-less single-turn path. Step count and profiler settings are identical
so the two sets of traces are directly comparable.
"""

from typing import cast

from renderers import Message, Qwen3RendererConfig

from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import ChatProcessor
from torchtitan.components.renderer import from_renderers
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.qwen3 import sft_qwen3_8b_math


_STEPS = 20


def _profiler_config() -> Profiler.Config:
    """Build a fresh profiler configuration shared by both recipes.

    Returns a new instance per call so the two configurations never alias the
    same mutable dataclass.
    """
    return Profiler.Config(
        enable_profiling=True,
        profile_freq=10,
        profiler_warmup=3,
        profiler_active=1,
        enable_memory_snapshot=True,
    )


def sft_qwen3_8b_multiturn_profiled() -> Trainer.Config:
    """Qwen3-8B SFT on GSM8K rendered as a two-round-trip conversation."""

    def messages_fn(sample) -> list[Message]:
        answer = sample["answer"]
        reasoning, final_answer = answer.rsplit("####", 1)
        reasoning = reasoning.strip()
        final_answer = final_answer.strip()
        return [
            {"role": "user", "content": sample["question"]},
            {
                "role": "assistant",
                # Only the renderer path consumes reasoning_content.
                "reasoning_content": reasoning,
                "content": final_answer,
            },
            {"role": "user", "content": "Repeat your answer."},
            {"role": "assistant", "content": final_answer},
        ]

    config = sft_qwen3_8b_math()
    dataloader = cast(GrainDataLoader.Config, config.dataloader)
    packing = cast(FirstFitPackingConfig, dataloader.dataset)
    dataset = cast(SingleDatasetConfig, packing.dataset)
    processor = cast(ChatProcessor.Config, dataset.processor)
    processor.messages_fn = messages_fn
    processor.renderer = from_renderers(Qwen3RendererConfig())
    config.training.steps = _STEPS
    config.profiler = _profiler_config()
    return config


def sft_qwen3_8b_singleturn_profiled() -> Trainer.Config:
    """Single-turn control for ``sft_qwen3_8b_multiturn_profiled``."""
    config = sft_qwen3_8b_math()
    config.training.steps = _STEPS
    config.profiler = _profiler_config()
    return config
