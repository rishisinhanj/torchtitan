"""Small two-step TitanRL recipe for ROCm Flex batch-invariance validation."""

import dataclasses
import os

from torchtitan_recipes.rl.alphabet_sort import (
    rl_grpo_qwen3_0_6b_flex_batch_invariant,
)


def rocm_flex_bi_two_step():
    config = rl_grpo_qwen3_0_6b_flex_batch_invariant()
    config.hf_assets_path = os.environ.get(
        "HF_ASSETS_PATH",
        config.hf_assets_path,
    )
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=2,
        num_prompts_per_train_step=2,
        num_samples_per_prompt=4,
    )
    config.generator = dataclasses.replace(
        config.generator,
        cuda_graph=dataclasses.replace(
            config.generator.cuda_graph,
            mode="NONE",
        ),
        sampling=dataclasses.replace(
            config.generator.sampling,
            max_tokens=32,
        ),
    )
    config.metrics = dataclasses.replace(
        config.metrics,
        enable_wandb=False,
    )
    return config
