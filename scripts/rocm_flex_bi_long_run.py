"""Configurable TitanRL ROCm Flex batch-invariance validation recipe."""

import dataclasses
import os

from torchtitan.config.transform import BatchInvariantFlexConverter
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan_recipes.rl.alphabet_sort import (
    _qwen3_rl_model_configs,
    rl_grpo_qwen3_0_6b_flex_batch_invariant,
)


def rocm_flex_bi_long_run():
    flavor = os.environ.get("ROCM_BI_MODEL_FLAVOR", "0.6B")
    num_steps = int(os.environ.get("ROCM_BI_NUM_STEPS", "25"))
    num_samples = int(os.environ.get("ROCM_BI_NUM_SAMPLES", "8"))
    temperature = float(os.environ.get("ROCM_BI_TEMPERATURE", "0.8"))
    num_prompts = 2
    if num_prompts * num_samples > 64:
        raise ValueError(
            "Paged Flex batch invariance supports at most 64 simultaneous "
            f"requests, got {num_prompts * num_samples}"
        )

    config = rl_grpo_qwen3_0_6b_flex_batch_invariant()
    config.hf_assets_path = os.environ.get(
        "HF_ASSETS_PATH",
        config.hf_assets_path,
    )
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=num_steps,
        num_prompts_per_train_step=num_prompts,
        num_samples_per_prompt=num_samples,
    )
    config.generator = dataclasses.replace(
        config.generator,
        cuda_graph=dataclasses.replace(
            config.generator.cuda_graph,
            mode="NONE",
        ),
        sampling=dataclasses.replace(
            config.generator.sampling,
            temperature=temperature,
            max_tokens=32,
        ),
    )
    config.trainer = dataclasses.replace(
        config.trainer,
        checkpointer=dataclasses.replace(
            config.trainer.checkpointer,
            # Avoid full optimizer checkpoints during this parity-only matrix.
            interval=num_steps + 1,
            last_save_model_only=True,
        ),
    )
    config.metrics = MetricsProcessor.Config(
        enable_wandb=True,
        wandb_project=os.environ.get(
            "WANDB_PROJECT",
            "rocm_batch_invariance",
        ),
    )

    if flavor != "0.6B":
        config.trainer.model, config.generator.model = _qwen3_rl_model_configs(
            flavor,
            seq_len=config.trainer.training.max_context_length,
            attn_backend="flex",
            trainer_parallelism=config.trainer.parallelism,
            generator_parallelism=config.generator.parallelism,
            converters=[BatchInvariantFlexConverter.Config()],
        )
        for model_config in (config.trainer.model, config.generator.model):
            model_config.local_compile_regions = []

    return config
