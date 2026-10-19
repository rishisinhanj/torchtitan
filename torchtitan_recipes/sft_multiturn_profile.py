# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Profiled Qwen3-8B GSM8K SFT configurations for a multi-turn vs single-turn study.

All three configurations derive from ``sft_qwen3_8b_math`` and use the same
Qwen3 renderer, so the tokenization path is identical and only the messages
differ. The control keeps the base recipe's single round trip. The two
multi-turn recipes answer different questions: ``multiturn`` doubles the
content across two round trips, which measures whether a longer corpus pushes
the data producer onto the critical path, while ``multiturn_token_matched``
splits the same content across two round trips, which isolates what the
four-message structure costs per sample.

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


def sft_qwen3_8b_multiturn_token_matched_profiled() -> Trainer.Config:
    """Qwen3-8B SFT on GSM8K split across two round trips at matched length.

    Carries the same supervised content as the control at roughly the same
    token count, so a per-document comparison against the control attributes
    cost to the four-message structure rather than to length.
    ``sft_qwen3_8b_multiturn_profiled`` instead doubles the content, which
    measures capacity rather than per-sample structure.
    """

    def messages_fn(sample) -> list[Message]:
        answer = sample["answer"]
        reasoning, final_answer = answer.rsplit("####", 1)
        steps = reasoning.strip().split("\n")
        half = max(1, len(steps) // 2)
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": "\n".join(steps[:half])},
            {"role": "user", "content": "Continue."},
            {
                "role": "assistant",
                "reasoning_content": "\n".join(steps[half:]),
                "content": final_answer.strip(),
            },
        ]

    config = _profiled_base()
    _chat_processor(config).messages_fn = messages_fn
    return config


def sft_qwen3_8b_singleturn_profiled() -> Trainer.Config:
    """Single-turn control for both multi-turn recipes."""
    return _profiled_base()
