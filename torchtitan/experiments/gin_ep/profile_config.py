import os

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
)
from torchtitan.config import ParallelismConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.deepseek_v3 import model_registry
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_16b


def _deepseek_16b_profile(comm_backend: str):
    world_size = int(os.environ.get("GIN_PROFILE_WORLD_SIZE", "16"))
    steps = int(os.environ.get("GIN_PROFILE_STEPS", "4"))
    tokens = int(os.environ.get("GIN_PROFILE_TOKENS", "512"))
    enable_profiling = os.environ.get("GIN_PROFILE_ENABLE", "1") == "1"
    config = deepseek_v3_16b()
    config.model_spec = model_registry(
        "16B",
        attn_backend="flex",
        moe_comm_backend=comm_backend,
    )
    if comm_backend == "gin":
        capacity_factor = float(
            os.environ.get("GIN_PROFILE_CAPACITY_FACTOR", "1.0")
        )
        cta_count = int(os.environ.get("GIN_PROFILE_CTA_COUNT", "64"))
        static_balanced_routing = (
            os.environ.get("GIN_PROFILE_STATIC_BALANCED", "0") == "1"
        )
        steady_state_barrier_elision = (
            os.environ.get("GIN_PROFILE_BARRIER_ELISION", "0") == "1"
        )
        phase_timing = (
            os.environ.get("GIN_PROFILE_PHASE_TIMING", "0") == "1"
        )
        phase_timing_capacity = int(
            os.environ.get("GIN_PROFILE_PHASE_TIMING_CAPACITY", "2048")
        )
        for layer in config.model_spec.model.layers:
            moe = getattr(layer, "moe", None)
            if moe is None:
                continue
            dispatcher = moe.routed_experts.token_dispatcher
            dispatcher.capacity_factor = capacity_factor
            dispatcher.cta_count = cta_count
            dispatcher.static_balanced_routing = static_balanced_routing
            dispatcher.steady_state_barrier_elision = (
                steady_state_barrier_elision
            )
            dispatcher.phase_timing = phase_timing
            dispatcher.phase_timing_capacity = phase_timing_capacity
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )

    config.training.num_tokens_per_microbatch_per_dp_rank = tokens
    config.training.max_context_length = tokens
    config.training.steps = steps
    config.training.disable_cuda_graphs = True
    config.training.dtype = "bfloat16"

    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=world_size,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=world_size,
        enable_sequence_parallel=False,
        spmd_backend="partial_dtensor",
    )

    config.activation_checkpoint = None
    config.compile.enable = False
    config.checkpoint.enable = False
    config.debug.moe_force_load_balance = True
    config.debug.enable_structured_logging = False
    config.metrics.log_freq = 1

    config.profiler.enable_profiling = enable_profiling
    config.profiler.profile_freq = 3
    config.profiler.profiler_warmup = 1
    config.profiler.profiler_active = 1
    config.profiler.profiler_repeat = 1
    config.profiler.with_stack = True
    return config


def deepseek_16b_standard_profile():
    return _deepseek_16b_profile("standard")


def deepseek_16b_gin_profile():
    return _deepseek_16b_profile("gin")
