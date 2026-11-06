# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Terminal-Bench recipes that trace where an RL step's time goes, one 8-GPU host each.

Qwen3 base models train on TMax (``local/tmax@v1``) and validate on Terminal-Bench
2.1 with the Verifiers setup of ``verifiers_terminal_bench.py``. Each model has
three arms:

- ``shakedown``: 3 steps, 2 samples per prompt, 4 agent turns, no profiling; the
  rollout recorder keeps every rollout.
- ``mt``: multi-turn trace, 12 steps of 4 prompts x 8 samples, 24 agent turns; the
  trainer profiler traces step 8, and the generator profiler the first 256 engine
  steps from the start of step 8.
- ``st``: ``mt`` with one agent turn, the single-turn control.

Qwen3-30B-A3B-Base also runs every arm with four one-GPU generators (``ep1``)
instead of one expert-parallel generator, whose all-to-all decode step took
about 150 ms per token, and ``mt``/``st`` on them with decode CUDA graphs
(``ep1_graphs``).

Base models score about 0 on these tasks, so zero-variance reward groups are
trained on instead of dropped; dropping them ends a run after 10 untrainable
batches.
"""

from typing import Literal

from renderers import Qwen3RendererConfig

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
from torchtitan.models.qwen3 import build_model_config
from torchtitan.observability.metrics import TensorBoardLogger, WandBLogger
from torchtitan.observability.profiler import Profiler
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.examples.verifiers.rollouter import VerifiersRollouter
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig, VLLMProfilerConfig
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
from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.losses import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder
from verifiers.v1.configs.agent import TimeoutConfig as AgentTimeoutConfig

from torchtitan_recipes.rl.model_configs import build_rl_model_configs
from torchtitan_recipes.rl.verifiers_terminal_bench import (
    _terminal_bench_rollouter_config,
)

_Arm = Literal["shakedown", "mt", "st"]

_MODELS_DIR = "/mnt/m2m_nobackup/rissinha/mt-trace/models"
_RUNS_DIR = "/mnt/m2m_nobackup/rissinha/mt-trace/runs"
# 24 agent turns of about 1.1K tokens (completion plus terminal output) fit.
_MAX_CONTEXT_LENGTH = 32768
_MAX_OUTPUT_TOKENS = 4096


def _trace_rollouter_config(
    *, max_turns: int, max_concurrent_rollouts: int
) -> VerifiersRollouter.Config:
    rollouter = _terminal_bench_rollouter_config(
        train_dataset="local/tmax@v1",
        validation_dataset="terminal-bench/terminal-bench-2-1",
        max_context_length=_MAX_CONTEXT_LENGTH,
        max_turns=max_turns,
        max_concurrent_rollouts=max_concurrent_rollouts,
    )
    environment = rollouter.verifiers_env_server.environment
    # A rollout stuck at upstream's limits (7200 s agent, 12000 s scoring) would
    # hold its group, and the step after it, for longer than a 12-step run takes.
    timeout = AgentTimeoutConfig(setup=600, rollout=1800, scoring=1800)
    rollouter.verifiers_env_server.environment = environment.model_copy(
        update={"agent": environment.agent.model_copy(update={"timeout": timeout})}
    )
    return rollouter


def _qwen3_base_terminal_bench_trace(
    flavor: str,
    *,
    arm: _Arm,
    hf_assets_path: str,
    dump_folder: str,
    trainer_parallelism: ParallelismConfig,
    generator_parallelism: InferenceParallelismConfig,
    num_generators: int,
    generator_cuda_graph_mode: Literal["NONE", "FULL_DECODE_ONLY"],
) -> Controller.Config:
    """The arm's loop, rollouter and profiling around one model and GPU layout.

    Optimizer, schedule, remat, precision, loss, routing and validation logging
    follow ``rl_grpo_qwen35_9b_terminal_bench``.
    """
    is_shakedown = arm == "shakedown"
    async_loop = AsyncLoopConfig(
        num_training_steps=3 if is_shakedown else 12,
        num_prompts_per_train_step=4,
        num_samples_per_prompt=2 if is_shakedown else 8,
        target_offpolicy_steps=2,
        batcher=Batcher.Config(
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False
            )
        ),
        validation=ValidationConfig(steps=2 if is_shakedown else 8),
    )
    max_turns = {"shakedown": 4, "mt": 24, "st": 1}[arm]
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        flavor,
        seq_len=_MAX_CONTEXT_LENGTH,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path=hf_assets_path,
        dump_folder=dump_folder,
        rollout_recorder=RolloutSampleRecorder.Config(
            # The shakedown records every rollout: a group of at most 2k is kept whole.
            filter=KeepExtremeRewardsFilter.Config(
                k=async_loop.num_samples_per_prompt if is_shakedown else 1,
                keep_errors=True,
            )
        ),
        async_loop=async_loop,
        rollouter=_trace_rollouter_config(
            max_turns=max_turns,
            max_concurrent_rollouts=async_loop.max_active_rollout_groups
            * async_loop.num_samples_per_prompt,
        ),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        num_generators=num_generators,
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(
            loggers=[WandBLogger.Config(), TensorBoardLogger.Config()],
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
                "timing/validate",
            ],
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
                num_tokens_per_microbatch_per_dp_rank=_MAX_CONTEXT_LENGTH,
            ),
            parallelism=trainer_parallelism,
            activation_remat=FullActivationRematConfig(),
            # Loads the base weights and never saves.
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True, load_only=True
            ),
            # Traces step 8: steps 1-4 wait, 5-7 warm up.
            profiler=Profiler.Config(
                enable_profiling=not is_shakedown,
                profile_freq=8,
                profiler_warmup=3,
                profiler_active=1,
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
                cuda_graph=VLLMCudaGraphConfig(mode=generator_cuda_graph_mode),
                # Starts with step 8, the step the trainer profiler traces.
                profiler=None if is_shakedown else VLLMProfilerConfig(),
            ),
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=_MAX_OUTPUT_TOKENS,
            ),
        ),
    )


def _qwen3_8b_base(arm: _Arm, *, name: str) -> Controller.Config:
    """Qwen3-8B-Base: 4 trainer GPUs (FSDP=4) and 4 one-GPU generators."""
    return _qwen3_base_terminal_bench_trace(
        "8B",
        arm=arm,
        hf_assets_path=f"{_MODELS_DIR}/Qwen3-8B-Base",
        dump_folder=f"{_RUNS_DIR}/{name}",
        trainer_parallelism=ParallelismConfig(data_parallel_shard_degree=4),
        generator_parallelism=InferenceParallelismConfig(),
        num_generators=4,
        generator_cuda_graph_mode="FULL_DECODE_ONLY",
    )


def _qwen3_30b_a3b_base(arm: _Arm, *, name: str) -> Controller.Config:
    """Qwen3-30B-A3B-Base: 4 trainer GPUs (FSDP=2, TP=2, EP=4) and one 4-GPU
    generator (DP=2, TP=2, EP=4), the layout of ``rl_grpo_qwen3_30b_a3b_varlen``.

    Generator CUDA graphs are off: the standard all-to-all MoE token dispatcher
    copies split sizes to the host, which graph capture does not allow.
    """
    config = _qwen3_base_terminal_bench_trace(
        "30B-A3B",
        arm=arm,
        hf_assets_path=f"{_MODELS_DIR}/Qwen3-30B-A3B-Base",
        dump_folder=f"{_RUNS_DIR}/{name}",
        trainer_parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            expert_parallel_degree=4,
        ),
        generator_parallelism=InferenceParallelismConfig(
            data_parallel_degree=2,
            tensor_parallel_degree=2,
            expert_parallel_degree=4,
        ),
        num_generators=1,
        generator_cuda_graph_mode="NONE",
    )
    # Weight-sync receive buffers live outside vLLM's budget; at 0.9 the first
    # forward after a model state dict load ran out of memory.
    config.generator.gpu_memory_limit = 0.8
    return config


def _qwen3_30b_a3b_base_ep1(
    arm: _Arm, *, name: str, cuda_graphs: bool = False
) -> Controller.Config:
    """Qwen3-30B-A3B-Base: the trainer of ``_qwen3_30b_a3b_base`` and four one-GPU
    generators (TP=1, EP=1).

    Without expert parallelism a decode step has no all-to-all, no split sizes
    copied to the host and no TP all-reduce. ``torch._grouped_mm``'s ROCm fallback
    copies its group offsets to the host, which CUDA graph capture does not allow;
    with ``cuda_graphs`` the generators capture decode graphs, which needs the
    CK-tile grouped GEMM the vLLM generator enables on ROCm.

    An eager decode step records ~150K CPU ops in the generator profiler, and writing 32
    steps of them stalls each engine for ~8 minutes, so its window is 16 engine steps
    instead of 256. With decode graphs only prefill steps run eagerly: 64 steps.
    """
    config = _qwen3_base_terminal_bench_trace(
        "30B-A3B",
        arm=arm,
        hf_assets_path=f"{_MODELS_DIR}/Qwen3-30B-A3B-Base",
        dump_folder=f"{_RUNS_DIR}/{name}",
        trainer_parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            expert_parallel_degree=4,
        ),
        generator_parallelism=InferenceParallelismConfig(),
        num_generators=4,
        generator_cuda_graph_mode="FULL_DECODE_ONLY" if cuda_graphs else "NONE",
    )
    config.generator.gpu_memory_limit = 0.8
    if config.generator.engine.profiler is not None:
        config.generator.engine.profiler = VLLMProfilerConfig(
            max_iterations=64 if cuda_graphs else 16
        )
    return config


def rl_grpo_qwen3_8b_base_tb_shakedown() -> Controller.Config:
    """Qwen3-8B-Base shakedown: 3 steps, 2 samples per prompt, 4 agent turns."""
    return _qwen3_8b_base("shakedown", name="rl_grpo_qwen3_8b_base_tb_shakedown")


def rl_grpo_qwen3_8b_base_tb_mt() -> Controller.Config:
    """Qwen3-8B-Base multi-turn trace: 12 steps, 4 prompts x 8 samples, 24 turns."""
    return _qwen3_8b_base("mt", name="rl_grpo_qwen3_8b_base_tb_mt")


def rl_grpo_qwen3_8b_base_tb_st() -> Controller.Config:
    """Qwen3-8B-Base single-turn control: the multi-turn trace with one agent turn."""
    return _qwen3_8b_base("st", name="rl_grpo_qwen3_8b_base_tb_st")


def rl_grpo_qwen3_30b_a3b_base_tb_shakedown() -> Controller.Config:
    """Qwen3-30B-A3B-Base shakedown: 3 steps, 2 samples per prompt, 4 agent turns."""
    return _qwen3_30b_a3b_base(
        "shakedown", name="rl_grpo_qwen3_30b_a3b_base_tb_shakedown"
    )


def rl_grpo_qwen3_30b_a3b_base_tb_mt() -> Controller.Config:
    """Qwen3-30B-A3B-Base multi-turn trace: 12 steps, 4 prompts x 8 samples, 24 turns."""
    return _qwen3_30b_a3b_base("mt", name="rl_grpo_qwen3_30b_a3b_base_tb_mt")


def rl_grpo_qwen3_30b_a3b_base_tb_st() -> Controller.Config:
    """Qwen3-30B-A3B-Base single-turn control: the multi-turn trace with one agent turn."""
    return _qwen3_30b_a3b_base("st", name="rl_grpo_qwen3_30b_a3b_base_tb_st")


def rl_grpo_qwen3_30b_a3b_base_ep1_tb_shakedown() -> Controller.Config:
    """Qwen3-30B-A3B-Base shakedown on four one-GPU generators."""
    return _qwen3_30b_a3b_base_ep1(
        "shakedown", name="rl_grpo_qwen3_30b_a3b_base_ep1_tb_shakedown"
    )


def rl_grpo_qwen3_30b_a3b_base_ep1_tb_mt() -> Controller.Config:
    """Qwen3-30B-A3B-Base multi-turn trace on four one-GPU generators."""
    return _qwen3_30b_a3b_base_ep1("mt", name="rl_grpo_qwen3_30b_a3b_base_ep1_tb_mt")


def rl_grpo_qwen3_30b_a3b_base_ep1_tb_st() -> Controller.Config:
    """Qwen3-30B-A3B-Base single-turn control on four one-GPU generators."""
    return _qwen3_30b_a3b_base_ep1("st", name="rl_grpo_qwen3_30b_a3b_base_ep1_tb_st")


def rl_grpo_qwen3_30b_a3b_base_ep1_graphs_tb_mt() -> Controller.Config:
    """Qwen3-30B-A3B-Base multi-turn trace on four one-GPU generators with decode graphs."""
    return _qwen3_30b_a3b_base_ep1(
        "mt", name="rl_grpo_qwen3_30b_a3b_base_ep1_graphs_tb_mt", cuda_graphs=True
    )


def rl_grpo_qwen3_30b_a3b_base_ep1_graphs_tb_st() -> Controller.Config:
    """Qwen3-30B-A3B-Base single-turn control on four one-GPU generators with decode graphs."""
    return _qwen3_30b_a3b_base_ep1(
        "st", name="rl_grpo_qwen3_30b_a3b_base_ep1_graphs_tb_st", cuda_graphs=True
    )
