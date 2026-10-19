# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Profiled Qwen3-8B GSM8K SFT configurations for a multi-turn vs single-turn study.

Both configurations derive from ``sft_qwen3_8b_math``, use the same Qwen3
renderer, and differ only in the number of messages each GSM8K sample becomes:
the subject expands a sample into two user/assistant round trips, the control
keeps the base recipe's single round trip. Routing both through the renderer
keeps the tokenization path identical, so turn count is the only variable.

``reasoning_content`` has to sit on the FINAL assistant turn. Qwen3's renderer
emits a thinking block only for messages after the last user turn
(``renderers/qwen3.py``, ``emit_in_template_window``), so reasoning attached to
an earlier turn is dropped from the rendered text entirely. The subject
therefore carries the first turn's derivation inline in its ``content``.

Sequence length is pinned to 4096 so the token budget, rather than
``max_num_documents``, decides how many samples land in a packed row; at the
base recipe's 40960 every row hits the 32-document cap and bin packing cannot
respond to the length difference under test.
"""

from typing import cast

from renderers import Message, Qwen3RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
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
_SEQ_LEN = 4096


def _chat_processor(config: Trainer.Config) -> ChatProcessor.Config:
    """Return the chat processor buried in a recipe's packed Grain dataset."""
    dataloader = cast(GrainDataLoader.Config, config.dataloader)
    packing = cast(FirstFitPackingConfig, dataloader.dataset)
    dataset = cast(SingleDatasetConfig, packing.dataset)
    return cast(ChatProcessor.Config, dataset.processor)


def _profiled_base() -> Trainer.Config:
    """Build the profiled GSM8K recipe shared by the subject and the control.

    Returns a new config tree on every call. ``sft_qwen3_8b_math`` constructs
    its nested dataclasses fresh, so the two recipes never alias a processor.
    """
    config = sft_qwen3_8b_math(seq_len=_SEQ_LEN)
    config.training.steps = _STEPS
    config.profiler = Profiler.Config(
        enable_profiling=True,
        profile_freq=10,
        profiler_warmup=3,
        profiler_active=1,
    )
    # The trainer saves inside the profiled step span, so an 8B fp32 write
    # would land in the step-20 trace.
    cast(CheckpointManager.Config, config.checkpointer).load_only = True
    _chat_processor(config).renderer = from_renderers(Qwen3RendererConfig())
    return config


def sft_qwen3_8b_multiturn_profiled() -> Trainer.Config:
    """Qwen3-8B SFT on GSM8K expanded into two user/assistant round trips."""

    def messages_fn(sample) -> list[Message]:
        answer = sample["answer"]
        reasoning, final_answer = answer.rsplit("####", 1)
        reasoning = reasoning.strip()
        final_answer = final_answer.strip()
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": reasoning},
            {
                "role": "user",
                "content": "Restate your derivation and give the final answer.",
            },
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": final_answer,
            },
        ]

    config = _profiled_base()
    _chat_processor(config).messages_fn = messages_fn
    return config


def sft_qwen3_8b_singleturn_profiled() -> Trainer.Config:
    """Single-turn control for ``sft_qwen3_8b_multiturn_profiled``."""
    return _profiled_base()
