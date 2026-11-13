# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5-9B-Base on TMax with ``rl_grpo_qwen35_9b_terminal_bench``'s setup, on one 8-GPU host.

The model, renderer, sampling, optimizer, loss and agent timeouts are the 16-GPU
recipe's. What differs:

- 4 trainer GPUs (FSDP=4) and 4 one-GPU generators instead of 8 and 8.
- The shape of the 100-step run reported in the recipe's PR (#4942): 16 tasks per
  step and a 30-turn agent cap (step-0 held-out reward 0.334, training reward
  0.344 over steps 1-50 and 0.393 over 51-100). The PR doesn't give the samples
  per task; 8 matches its held-out evaluation.
- One step off-policy: 256 rollouts in flight, sandboxes included.
- No Terminal-Bench 2.1 validation: the training reward of the first steps is the
  base model's TMax reward.

``rl_grpo_qwen35_9b_base_tb_8gpu`` loads the base weights and never saves.
``rl_grpo_qwen35_9b_base_tb_8gpu_ckpt20`` also saves a full checkpoint at step 1
and every 20 steps (a crashed run resumes from the latest) and a bfloat16
model-only export at the last step. ``rl_grpo_qwen35_9b_base_tb_8gpu_ckpt10``
saves every 10 steps, for continuing a run from one of its full checkpoints
(``--resume-step``) on an allocation too short for 20 more steps, and
``rl_grpo_qwen35_9b_base_tb_8gpu_ckpt5`` every 5 and ``rl_grpo_qwen35_9b_base_tb_8gpu_ckpt2``
every 2, so that a time limit that is not a multiple of 10 steps loses fewer of them
(the time-limit SIGINT saves nothing; saves land on multiples of the interval).
"""

from renderers import Qwen35RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.observability.metrics import TensorBoardLogger, WandBLogger
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.engine import VLLMEngine
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.recorder import (
    KeepExtremeRewardsFilter,
    RolloutSampleRecorder,
)
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.trainer.losses import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer

from torchtitan_recipes.rl.model_configs import build_rl_model_configs
from torchtitan_recipes.rl.verifiers_terminal_bench import (
    _terminal_bench_rollouter_config,
)

_MODELS_DIR = "/mnt/m2m_nobackup/rissinha/mt-trace/models"
_RUNS_DIR = "/mnt/m2m_nobackup/rissinha/mt-trace/runs"


def rl_grpo_qwen35_9b_base_tb_8gpu() -> Controller.Config:
    """Qwen3.5-9B-Base: train on TMax (``local/tmax@v1``), 8 GPUs, 30 agent turns."""
    max_context_length = 131072
    max_output_tokens = 16384
    async_loop = AsyncLoopConfig(
        num_training_steps=100,
        num_prompts_per_train_step=16,
        num_samples_per_prompt=8,
        target_offpolicy_steps=1,
        validation=ValidationConfig(steps=0),
    )
    trainer_parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=4,
        tensor_parallel_degree=1,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=1,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "9B",
        seq_len=max_context_length,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path=f"{_MODELS_DIR}/Qwen3.5-9B-Base",
        dump_folder=f"{_RUNS_DIR}/qwen35_9b_base_tb_8gpu",
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(keep_errors=True)
        ),
        async_loop=async_loop,
        rollouter=_terminal_bench_rollouter_config(
            train_dataset="local/tmax@v1",
            validation_dataset="terminal-bench/terminal-bench-2-1",
            max_context_length=max_context_length,
            max_turns=30,
            max_concurrent_rollouts=async_loop.max_active_rollout_groups
            * async_loop.num_samples_per_prompt,
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(
                enable_thinking=True,
                thinking_retention="all",
            )
        ),
        num_generators=4,
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(
            loggers=[WandBLogger.Config(), TensorBoardLogger.Config()],
        ),
        precision=PrecisionConfig(dtype="float32"),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[
                        AdamW.Config(
                            pattern=r".*",
                            lr=1e-6,
                            betas=(0.9, 0.999),
                            weight_decay=0.0,
                        )
                    ]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0,
                    min_lr_factor=1.0,
                ),
            ),
            training=TrainingConfig(
                disable_accelerator_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_context_length,
            ),
            parallelism=trainer_parallelism,
            activation_remat=FullActivationRematConfig(),
            # Loads the base weights and never saves.
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True, load_only=True
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=32,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            engine=VLLMEngine.Config(
                cuda_graph=VLLMCudaGraphConfig(mode="FULL_DECODE_ONLY")
            ),
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_output_tokens,
            ),
        ),
    )


def rl_grpo_qwen35_9b_base_tb_8gpu_ckpt20() -> Controller.Config:
    """``rl_grpo_qwen35_9b_base_tb_8gpu``, saving a full checkpoint every 20 steps.

    The step-1 save exercises the save path before the first interval save.
    """
    config = rl_grpo_qwen35_9b_base_tb_8gpu()
    config.trainer.checkpointer = CheckpointManager.Config(
        initial_load_in_hf=True,
        enable_first_step_checkpoint=True,
        interval=20,
        last_save_model_only=True,
        export_dtype="bfloat16",
    )
    return config


def rl_grpo_qwen35_9b_base_tb_8gpu_ckpt10() -> Controller.Config:
    """``rl_grpo_qwen35_9b_base_tb_8gpu``, saving a full checkpoint every 10 steps."""
    config = rl_grpo_qwen35_9b_base_tb_8gpu()
    config.trainer.checkpointer = CheckpointManager.Config(
        initial_load_in_hf=True,
        interval=10,
        last_save_model_only=True,
        export_dtype="bfloat16",
    )
    return config


def rl_grpo_qwen35_9b_base_tb_8gpu_ckpt5() -> Controller.Config:
    """``rl_grpo_qwen35_9b_base_tb_8gpu``, saving a full checkpoint every 5 steps."""
    config = rl_grpo_qwen35_9b_base_tb_8gpu()
    config.trainer.checkpointer = CheckpointManager.Config(
        initial_load_in_hf=True,
        interval=5,
        last_save_model_only=True,
        export_dtype="bfloat16",
    )
    return config


def rl_grpo_qwen35_9b_base_tb_8gpu_ckpt2() -> Controller.Config:
    """``rl_grpo_qwen35_9b_base_tb_8gpu``, saving a full checkpoint every 2 steps."""
    config = rl_grpo_qwen35_9b_base_tb_8gpu()
    config.trainer.checkpointer = CheckpointManager.Config(
        initial_load_in_hf=True,
        interval=2,
        last_save_model_only=True,
        export_dtype="bfloat16",
    )
    return config
