import os

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
)
from torchtitan.config import ParallelismConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.deepseek_v3 import model_registry
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_16b, deepseek_v3_671b


def _deepseek_16b_profile(comm_backend: str):
    world_size = int(os.environ.get("GIN_PROFILE_WORLD_SIZE", "16"))
    steps = int(os.environ.get("GIN_PROFILE_STEPS", "4"))
    # Normal torchtitan knobs: sequence length and (local) batch size, set
    # independently rather than conflated into one "tokens" value. Total
    # per-rank token budget is their product (this fork packs sequences up
    # to seq_len to fill that budget -- see TrainingConfig in configs.py).
    seq_len = int(os.environ.get("GIN_PROFILE_SEQ_LEN", "512"))
    batch_size = int(os.environ.get("GIN_PROFILE_BATCH_SIZE", "1"))
    tokens = seq_len * batch_size
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
    config.training.max_context_length = seq_len
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


def _deepseek_671b_profile(comm_backend: str, num_layers: int = 8):
    """671B model truncated to num_layers, for a larger per-exchange
    message size than 16B (bigger hidden_dim / more experts) at the same
    seq_len/LBS. Requested layer count is taken as-is (not tuned for MoE-
    layer coverage -- 671B's first 8 layers are known to contain fewer MoE
    layers than 16B's 26, see torchtitan-gin-vs-rccl-20260902/README.md's
    "Workload decision" section)."""
    world_size = int(os.environ.get("GIN_PROFILE_WORLD_SIZE", "16"))
    steps = int(os.environ.get("GIN_PROFILE_STEPS", "4"))
    seq_len = int(os.environ.get("GIN_PROFILE_SEQ_LEN", "512"))
    batch_size = int(os.environ.get("GIN_PROFILE_BATCH_SIZE", "1"))
    tokens = seq_len * batch_size
    enable_profiling = os.environ.get("GIN_PROFILE_ENABLE", "1") == "1"
    config = deepseek_v3_671b()
    config.model_spec = model_registry(
        "671B",
        attn_backend="flex",
        moe_comm_backend=comm_backend,
    )
    config.model_spec.model.layers = config.model_spec.model.layers[:num_layers]

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

    # Real 671B assets/dataset (./assets/hf/DeepSeek-V3.1-Base, full
    # streaming "c4") are not present in this checkout -- substitute the
    # working tokenizer/dataset already used by the 16B profile harness,
    # same simplification, not a difference in comm-path behavior.
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )

    config.training.num_tokens_per_microbatch_per_dp_rank = tokens
    config.training.max_context_length = seq_len
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


def deepseek_671b_standard_profile():
    return _deepseek_671b_profile("standard")


def deepseek_671b_gin_profile():
    return _deepseek_671b_profile("gin")


def deepseek_16b_gin_profile():
    return _deepseek_16b_profile("gin")
