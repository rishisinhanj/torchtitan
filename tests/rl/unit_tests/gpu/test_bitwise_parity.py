#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test bitwise parity between vLLM generator and TorchTitan trainer.

Three tests:

  1. test_batch_invariance:
      Trainer prefill(bsz=m) == Trainer prefill(bsz=n, m!=n).
      Guards that model kernels are batch-invariant.
  2. test_trainer_vs_vllm_prefill:
      Trainer prefill == vLLM prefill (prompt-only).
      Ensures trainer and generator forward have bitwise parity.
  3. test_vllm_decode_vs_prefill:
      vLLM decode == vLLM 2nd-pass prefill (generated positions).
      Ensures prefill vs decode (KV-cache) parity.

By transitivity of test 2 and test 3: trainer == vLLM decode.

Run each backend in a separate torchrun invocation:
    torchrun --nproc_per_node=2 -m pytest \
        tests/rl/unit_tests/gpu/test_bitwise_parity.py::TestBitwiseParityVarlen -v

    torchrun --nproc_per_node=2 -m pytest \
        tests/rl/unit_tests/gpu/test_bitwise_parity.py::TestBitwiseParityFlex -v
"""

import copy
import dataclasses
import gc
import logging
import os
import shutil
import tempfile
import unittest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import compute_logprobs, IGNORE_INDEX
from torchtitan.config import apply_overrides, CommConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelismContext, utils as dist_utils
from torchtitan.distributed.batch_invariant import (
    is_in_batch_invariant_mode,
    set_batch_invariance,
)
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.observability.logging import init_logger
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.generator.vllm.registry import (
    register_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
    TORCHTITAN_WORKER_CLS,
    VLLM_MODEL_NAME,
)
from torchtitan.tools import utils
from torchtitan_recipes.rl.alphabet_sort import (
    rl_grpo_qwen3_0_6b_flex_batch_invariant,
    rl_grpo_qwen3_0_6b_varlen_batch_invariant,
    rl_grpo_qwen3_5_9b_varlen_batch_invariant,
)
from torchtitan_recipes.tests.rl.alphabet_sort import (
    rl_grpo_gpt_oss_debug_varlen_batch_invariant,
    rl_grpo_kimi_k3_debug_varlen_batch_invariant,
    rl_grpo_qwen3_5_debug_varlen_batch_invariant,
    rl_grpo_qwen3_moe_debug_varlen_batch_invariant,
)
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)

# Each class runs under its own two-GPU torchrun launch.
pytestmark = pytest.mark.multi_gpu


# ---------------------------------------------------------------------------
# Model and Engine setup
# ---------------------------------------------------------------------------


# TODO: directly testing against Trainer with debug model to avoid OOM
def build_trainer_model(
    config: Controller.Config,
) -> tuple[torch.nn.Module, torch.device, ParallelismContext]:
    """Build, parallelize, and load weights for the trainer model.

    Mirrors Trainer._build_model() without the Monarch actor framework.
    """
    model_config = copy.deepcopy(config.trainer.model)
    hf_assets_path = config.hf_assets_path

    device = utils.get_local_device()
    utils.device_module.set_device(device)

    parallelism = config.trainer.parallelism
    parallelism_context = ParallelismContext(
        dp_shard=parallelism.data_parallel_shard_degree,
        dp_replicate=parallelism.data_parallel_replicate_degree,
        cp=parallelism.context_parallel_degree,
        tp=parallelism.tensor_parallel_degree,
        pp=parallelism.pipeline_parallel_degree,
        ep=parallelism.expert_parallel_degree,
        world_size=dist.get_world_size(),
        enable_sequence_parallel=parallelism.enable_sequence_parallel,
    )
    dist_utils.set_determinism(
        parallelism_context,
        device,
        config.trainer.debug,
        distinct_seed_mesh_axes=["pp"],
    )

    trainer_config = config.trainer

    with (
        parallelism_context.activate_spmd(),
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[trainer_config.training.dtype]),
    ):
        model = model_config.build()

    model = model.parallelize(
        parallelism_context=parallelism_context,
        training=trainer_config.training,
        parallelism=parallelism,
        local_compile_regions=model_config.local_compile_regions,
        activation_remat=trainer_config.activation_remat,
        dump_folder=config.dump_folder,
    )
    with parallelism_context.activate_spmd():
        model.to_empty(device=device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

    # Load HF checkpoint if available
    adapter_cls = type(model).state_dict_adapter_cls
    if adapter_cls is not None and hf_assets_path:
        index_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
        single_path = os.path.join(hf_assets_path, "model.safetensors")
        if os.path.exists(index_path) or os.path.exists(single_path):
            sd_adapter = adapter_cls(model_config, hf_assets_path)
            storage_reader = sd_adapter.get_hf_storage_reader(hf_assets_path)
            hf_state_dict = sd_adapter.to_hf(model.state_dict())
            dcp.load(hf_state_dict, storage_reader=storage_reader)
            tt_state_dict = sd_adapter.from_hf(hf_state_dict)
            set_model_state_dict(
                model=model,
                model_state_dict=tt_state_dict,
                options=StateDictOptions(strict=False),
            )

    model.eval()
    return model, device, parallelism_context


# TODO: directly testing against VLLMGenerator with debug model to avoid OOM
def _set_generator_determinism(debug) -> None:
    """Apply deterministic flags for the generator side.

    Mirrors VLLMGenerator._set_determinism() — the generator doesn't use
    torchtitan's ParallelismContext, so we apply the flags directly.
    """
    if debug.deterministic:
        torch.use_deterministic_algorithms(
            True, warn_only=debug.deterministic_warn_only
        )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if debug.seed is not None:
        torch.manual_seed(debug.seed)


def build_inference_engine(config: Controller.Config) -> LLMEngine:
    """Create a vLLM LLMEngine with torchtitan model from the RL config."""
    gen_config = config.generator

    attention_backend = gen_config.model.first_full_attention.inner_attention
    use_flex = isinstance(attention_backend, FlexInnerAttention.Config)

    # Mirror the production VLLMGenerator so the test exercises the same
    # batch-invariant path (v2 runner is required for the logprob-kernel patch).
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    if use_flex:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
        backend_enum = AttentionBackendEnum.FLEX_ATTENTION
        if gen_config.debug.batch_invariant and torch.version.hip is not None:
            from torchtitan.rl.generator.vllm.batch_invariance import (
                force_flex_page_mapping_for_batch_invariance,
            )

            force_flex_page_mapping_for_batch_invariance()
    else:
        os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"
        if gen_config.debug.batch_invariant:
            set_batch_invariance(True)
            # The v2 logprob Triton kernel bypasses the aten overrides. Apply the
            # same generator-side patch the production VLLMGenerator does.
            from torchtitan.rl.generator.vllm.batch_invariance import (
                force_logprobs_fn_for_batch_invariance,
            )

            force_logprobs_fn_for_batch_invariance()
        backend_enum = AttentionBackendEnum.CUSTOM

    _set_generator_determinism(gen_config.debug)

    enable_ep = gen_config.parallelism.expert_parallel_degree > 1
    engine_kwargs = dict(
        model=config.hf_assets_path,
        trust_remote_code=True,
        # Build the model config from torchtitan's model config via the custom
        # parser registered by register_to_vllm, instead of reading config.json.
        config_format=TORCHTITAN_CONFIG_FORMAT,
        dtype=gen_config.model_dtype,
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        data_parallel_size=gen_config.parallelism.data_parallel_degree,
        enable_expert_parallel=enable_ep,
        worker_cls=TORCHTITAN_WORKER_CLS,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=gen_config.cuda_graph.mode == "NONE",
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
        attention_config=AttentionConfig(backend=backend_enum),
        disable_log_stats=True,
    )

    from torchtitan.tools.utils import has_cuda_capability

    if not has_cuda_capability(9, 0) and not use_flex:
        engine_kwargs["block_size"] = 256  # set blocksize to be 256 to align with FA2

    engine_kwargs["max_model_len"] = gen_config.model.max_context_length
    # Mirror Controller.setup_async for a single engine: derive from active rollout concurrency
    # (the active-buffer capacity num_group_workers, or the validation pass).
    async_loop = config.async_loop
    gen_dp = max(gen_config.parallelism.data_parallel_degree, 1)
    num_group_workers = async_loop.max_active_rollout_groups
    rollout_concurrency = max(
        num_group_workers * async_loop.num_samples_per_prompt,
        async_loop.validation.steps,
    )
    max_num_seqs = min((rollout_concurrency + gen_dp - 1) // gen_dp, 512)
    engine_kwargs["max_num_seqs"] = max_num_seqs
    expert_sequence_parallel_size = gen_config.parallelism.expert_sequence_parallel_size
    vllm_compilation_config = gen_config.cuda_graph.get_vllm_compilation_config(
        max_num_seqs=max_num_seqs,
        expert_sequence_parallel_size=expert_sequence_parallel_size,
        enable_sequence_parallel=gen_config.parallelism.enable_sequence_parallel,
    )
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if gen_config.debug.seed is not None:
        engine_kwargs["seed"] = gen_config.debug.seed

    return LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))


def _sync_trainer_weights_to_vllm(trainer_model, engine) -> None:
    """Copy the trainer model's weights into the vLLM model in-process."""

    wrapper = engine.model_executor.driver_worker.get_model()
    trainer_sd = trainer_model.state_dict()
    # As a model state dict load: fill the pinned CPU copy, then load it with load_state_dict.
    vllm_sd = wrapper._prefetched_model_state_dict

    missing = []
    for name, vparam in vllm_sd.items():
        tparam = trainer_sd.get(name)
        if tparam is None:
            missing.append(name)
            continue
        full = tparam.full_tensor() if isinstance(tparam, DTensor) else tparam
        with torch.inference_mode():
            if isinstance(vparam, DTensor):
                vparam.copy_(
                    distribute_tensor(full, vparam.device_mesh, vparam.placements)
                )
            else:
                vparam.copy_(full)

    wrapper.load_state_dict(vllm_sd)

    if dist.get_rank() == 0 and missing:
        logger.warning("vLLM params not present in trainer state_dict: %s", missing)

    # Sinks were injected during build with uninitialized values (no HF load);
    # re-inject now that the real sink weights are in place.
    if hasattr(wrapper, "_inject_attention_sinks"):
        wrapper._inject_attention_sinks()


# ---------------------------------------------------------------------------
# Logprob helpers
# ---------------------------------------------------------------------------


def _flex_prefill_logprobs(model, input_tensors, seq_lens, device):
    """Compute per-sequence logprobs using flex attention with packed sequences.

    Mirrors the trainer's flex attention path: pack documents into a single
    row, pad each document to block-aligned boundaries in batch-invariant
    mode, build backend-specific attention metadata, and extract per-document
    logprobs.
    """
    inner_attn = model.config.layers[0].attention.inner_attention
    assert isinstance(inner_attn, FlexInnerAttention.Config)
    block_size = inner_attn.block_size

    batch_invariant = is_in_batch_invariant_mode()

    if batch_invariant:
        padded_seq_lens = [
            ((sl + block_size - 1) // block_size) * block_size for sl in seq_lens
        ]
    else:
        padded_seq_lens = list(seq_lens)

    # Build packed token_ids and positions (positions reset to 0 per doc)
    parts, pos_parts = [], []
    for tensor, sl, psl in zip(input_tensors, seq_lens, padded_seq_lens):
        padded = torch.zeros(psl, dtype=tensor.dtype, device=device)
        padded[:sl] = tensor
        parts.append(padded)
        pos_parts.append(torch.arange(psl, device=device))

    packed_ids = torch.cat(parts)
    positions = torch.cat(pos_parts)

    attention_metadata = model._get_attention_metadata(positions)

    logits = model(
        packed_ids, attention_metadata=attention_metadata, positions=positions
    )

    # Build pre-shifted labels matching the trainer convention:
    # labels[i] = packed_ids[i+1] for valid positions, IGNORE_INDEX otherwise.
    labels = torch.full_like(packed_ids, IGNORE_INDEX)
    offset = 0
    for sl, psl in zip(seq_lens, padded_seq_lens):
        labels[offset : offset + sl - 1] = packed_ids[offset + 1 : offset + sl]
        offset += psl

    logprobs = compute_logprobs(
        logits,
        labels,
        vocab_parallel_group=spmd_mesh_group("tp"),
        global_vocab_size=model.config.lm_head.out_features,
    )

    results = []
    offset = 0
    for sl, psl in zip(seq_lens, padded_seq_lens):
        results.append(logprobs[offset : offset + sl - 1])
        offset += psl
    return results


def _varlen_prefill_logprobs(model, input_tensors, seq_lens, device):
    """Compute per-sequence logprobs using packed variable-length segments."""
    packed_ids = torch.cat(input_tensors)
    positions = torch.cat(
        [torch.arange(seq_len, device=device) for seq_len in seq_lens]
    )

    # Explicit positions avoid dynamic rope_cache[0:seqlen] slice in RoPE,
    # which can break torch.compile with symbolic shapes.
    # Hybrid models may require different metadata for each attention type.
    attention_metadata = model._get_attention_metadata(positions)

    logits = model(
        packed_ids, attention_metadata=attention_metadata, positions=positions
    )

    # Build pre-shifted labels matching the trainer convention:
    # labels[i] = packed_ids[i+1] within each segment, IGNORE_INDEX otherwise.
    labels = torch.full_like(packed_ids, IGNORE_INDEX)
    offset = 0
    for t in input_tensors:
        seq_len = t.shape[0]
        labels[offset : offset + seq_len - 1] = t[1:seq_len]
        offset += seq_len

    logprobs = compute_logprobs(
        logits,
        labels,
        vocab_parallel_group=spmd_mesh_group("tp"),
        global_vocab_size=model.config.lm_head.out_features,
    )

    results = []
    offset = 0
    for t in input_tensors:
        seq_len = t.shape[0]
        results.append(logprobs[offset : offset + seq_len - 1])
        offset += seq_len
    return results


def compute_trainer_prefill_logprobs(model, token_ids, device, attn_backend="varlen"):
    """Compute next-token logprobs using the trainer model.

    Args:
        token_ids: A single sequence (list[int]) or a batch of sequences
            (list[list[int]]). Batched sequences are packed with appropriate
            attention metadata.
        attn_backend: 'varlen' or 'flex'.

    Returns:
        Single sequence: float32 tensor with len = len(token_ids) - 1.
        Batch: list of float32 tensors, one per sequence.
    """
    batched = isinstance(token_ids[0], list)
    seqs = token_ids if batched else [token_ids]

    input_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in seqs]
    seq_lens = [t.shape[0] for t in input_tensors]

    if attn_backend == "flex":
        results = _flex_prefill_logprobs(model, input_tensors, seq_lens, device)
    else:
        results = _varlen_prefill_logprobs(model, input_tensors, seq_lens, device)

    return results if batched else results[0]


def _capture_local_tensor(
    captures: dict[str, list[torch.Tensor]],
    name: str,
    value,
) -> None:
    """Save a rank-local tensor at a model boundary for parity diagnostics."""
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"{name}: expected one tensor output, got {len(value)}")
        value = value[0]
    if isinstance(value, DTensor):
        value = value.to_local()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name}: expected tensor, got {type(value)}")
    captures.setdefault(name, []).append(value.detach().clone())


def _register_parity_capture_hooks(
    model: torch.nn.Module,
    captures: dict[str, list[torch.Tensor]],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Capture each Qwen3 layer and final model boundaries."""
    handles = []

    def capture_output(name):
        def hook(_module, _args, output):
            _capture_local_tensor(captures, name, output)

        return hook

    def capture_input(name):
        def hook(_module, args):
            _capture_local_tensor(captures, name, args[0])

        return hook

    def capture_inner_inputs(prefix):
        def hook(_module, args, _kwargs):
            for suffix, value in zip(("q", "k", "v"), args[:3], strict=True):
                _capture_local_tensor(captures, f"{prefix}.{suffix}", value)

        return hook

    def capture_paged_cache(prefix):
        def hook(module, _args, _output):
            from vllm.model_executor.layers.attention.attention import (
                get_attention_context,
            )

            metadata, _, kv_cache, slot_mapping = get_attention_context(
                module.layer_name
            )
            if metadata is None or metadata.num_actual_tokens <= 1:
                return
            for suffix, value in (
                ("block_table", metadata.block_table),
                ("seq_lens", metadata.seq_lens),
                ("query_start_loc", metadata.query_start_loc),
                ("decode_offset", metadata.decode_offset),
                ("doc_ids", metadata.doc_ids),
                ("physical_to_logical", metadata.physical_to_logical),
                ("kv_cache", kv_cache),
                ("slot_mapping", slot_mapping),
            ):
                _capture_local_tensor(captures, f"{prefix}.{suffix}", value)
            block_mask = metadata.block_mask
            if block_mask is None:
                raise AssertionError(f"{prefix}: vLLM did not build a block mask")
            for suffix, value in (
                ("blockmask_kv_num_blocks", block_mask.kv_num_blocks),
                ("blockmask_kv_indices", block_mask.kv_indices),
            ):
                _capture_local_tensor(captures, f"{prefix}.{suffix}", value)
            _capture_local_tensor(
                captures,
                f"{prefix}.blockmask_block_size",
                torch.tensor(block_mask.BLOCK_SIZE, device=kv_cache.device),
            )

        return hook

    handles.append(
        model.tok_embeddings.register_forward_hook(capture_output("embedding"))
    )
    for layer_name, layer in model.layers.items():
        prefix = f"layer{layer_name}"
        attention = layer.attention
        layer_handles = [
            layer.register_forward_pre_hook(capture_input(f"{prefix}.input")),
            layer.register_forward_hook(capture_output(f"{prefix}.output")),
            layer.attention_norm.register_forward_hook(
                capture_output(f"{prefix}.attention_norm")
            ),
            attention.inner_attention.register_forward_pre_hook(
                capture_inner_inputs(prefix),
                with_kwargs=True,
            ),
            attention.inner_attention.register_forward_hook(
                capture_output(f"{prefix}.inner_attention")
            ),
            attention.wo.register_forward_hook(capture_output(f"{prefix}.wo")),
            attention.register_forward_hook(capture_output(f"{prefix}.attention")),
            layer.ffn_norm.register_forward_pre_hook(
                capture_input(f"{prefix}.post_attention_residual")
            ),
            layer.ffn_norm.register_forward_hook(capture_output(f"{prefix}.ffn_norm")),
            layer.feed_forward.register_forward_hook(
                capture_output(f"{prefix}.feed_forward")
            ),
        ]
        if layer_name == "2" and hasattr(attention.inner_attention, "vllm_attn"):
            layer_handles.append(
                attention.inner_attention.vllm_attn.register_forward_hook(
                    capture_paged_cache(prefix)
                )
            )
        handles.extend(layer_handles)
    handles.extend(
        [
            model.norm.register_forward_hook(capture_output("final_norm")),
            model.lm_head.register_forward_hook(capture_output("logits")),
        ]
    )
    return handles


def _select_prefill_capture(
    captures: dict[str, list[torch.Tensor]],
    name: str,
) -> torch.Tensor:
    """Select the longest token-bearing call, excluding one-token decode calls."""
    values = captures.get(name, [])
    if not values:
        raise AssertionError(f"no captures recorded for {name}")
    return max(values, key=lambda tensor: tensor.shape[0])


def _compare_capture(
    name: str,
    trainer: torch.Tensor,
    generator: torch.Tensor,
    num_tokens: int,
) -> tuple[bool, str]:
    """Compare the real-token prefix and describe the first bitwise mismatch."""
    tensor_metadata = (
        f"trainer_shape={tuple(trainer.shape)}, "
        f"vllm_shape={tuple(generator.shape)}, dtype={trainer.dtype}"
    )
    trainer = trainer[:num_tokens].detach().cpu()
    generator = generator[:num_tokens].detach().cpu()
    if trainer.shape != generator.shape:
        return False, f"{name}: shape mismatch; {tensor_metadata}"
    if trainer.dtype != generator.dtype:
        return (
            False,
            f"{name}: dtype mismatch {trainer.dtype} != {generator.dtype}; "
            f"{tensor_metadata}",
        )
    if torch.equal(trainer, generator):
        return (
            True,
            f"{name}: bitwise equal ({trainer.numel()} elements); {tensor_metadata}",
        )

    unequal = trainer != generator
    first_flat = unequal.flatten().nonzero()[0]
    first_index = torch.unravel_index(first_flat, trainer.shape)
    max_delta = (
        (trainer.float() - generator.float()).abs().max().item()
        if trainer.numel()
        else 0.0
    )
    return (
        False,
        (
            f"{name}: first_diff={tuple(index.item() for index in first_index)}, "
            f"num_diff={unequal.sum().item()}/{trainer.numel()}, "
            f"max_delta={max_delta:.6e}, "
            f"trainer={trainer[first_index].item()}, "
            f"vllm={generator[first_index].item()}; "
            f"{tensor_metadata}"
        ),
    )


def _reconstruct_paged_kv(
    captures: dict[str, list[torch.Tensor]],
    prefix: str,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild logical K/V token order from vLLM's packed paged cache."""
    kv_cache = captures[f"{prefix}.kv_cache"][-1]
    block_table = captures[f"{prefix}.block_table"][-1]
    seq_lens = captures[f"{prefix}.seq_lens"][-1]
    if seq_lens.numel() < 1 or seq_lens[0].item() < num_tokens:
        raise AssertionError(
            f"{prefix}: unexpected sequence lengths {seq_lens.tolist()}"
        )

    # vLLM Flex cache: [physical block, KV head, block offset, packed K+V].
    logical_cache = kv_cache.transpose(1, 2)
    head_size = logical_cache.shape[-1] // 2
    key_cache, value_cache = logical_cache.split(head_size, dim=-1)
    block_size = logical_cache.shape[1]
    token_indices = torch.arange(num_tokens, device=block_table.device)
    physical_blocks = block_table[0, token_indices // block_size].long()
    block_offsets = token_indices % block_size
    return (
        key_cache[physical_blocks, block_offsets],
        value_cache[physical_blocks, block_offsets],
    )


def _dump_flex_parity_capture(
    dump_dir: str,
    *,
    rank: int,
    prompt: list[int],
    trainer_captures: dict[str, list[torch.Tensor]],
    generator_captures: dict[str, list[torch.Tensor]],
    contiguous_replay: torch.Tensor,
    paged_k: torch.Tensor,
    paged_v: torch.Tensor,
    scale: float,
    enable_gqa: bool,
) -> str:
    """Persist the minimal layer-2 tensors needed for offline drift analysis."""
    os.makedirs(dump_dir, exist_ok=True)
    prefix = "layer2"
    kv_cache = generator_captures[f"{prefix}.kv_cache"][-1]
    block_table = generator_captures[f"{prefix}.block_table"][-1]
    cache_block_size = kv_cache.shape[2]
    num_prompt_blocks = (len(prompt) + cache_block_size - 1) // cache_block_size
    payload = {
        "rank": rank,
        "prompt": torch.tensor(prompt, dtype=torch.int64),
        "scale": scale,
        "enable_gqa": enable_gqa,
        "trainer_q": _select_prefill_capture(trainer_captures, f"{prefix}.q").cpu(),
        "trainer_k": _select_prefill_capture(trainer_captures, f"{prefix}.k").cpu(),
        "trainer_v": _select_prefill_capture(trainer_captures, f"{prefix}.v").cpu(),
        "generator_q": _select_prefill_capture(
            generator_captures, f"{prefix}.q"
        ).cpu(),
        "generator_k": _select_prefill_capture(
            generator_captures, f"{prefix}.k"
        ).cpu(),
        "generator_v": _select_prefill_capture(
            generator_captures, f"{prefix}.v"
        ).cpu(),
        "trainer_output": _select_prefill_capture(
            trainer_captures, f"{prefix}.inner_attention"
        ).cpu(),
        "generator_output": _select_prefill_capture(
            generator_captures, f"{prefix}.inner_attention"
        ).cpu(),
        "contiguous_replay": contiguous_replay.cpu(),
        "reconstructed_paged_k": paged_k.cpu(),
        "reconstructed_paged_v": paged_v.cpu(),
        "physical_blocks": block_table[0, :num_prompt_blocks].cpu(),
        "packed_cache_blocks": kv_cache[
            block_table[0, :num_prompt_blocks].long()
        ].cpu(),
        "seq_lens": generator_captures[f"{prefix}.seq_lens"][-1].cpu(),
        "query_start_loc": generator_captures[
            f"{prefix}.query_start_loc"
        ][-1].cpu(),
        "decode_offset": generator_captures[f"{prefix}.decode_offset"][-1].cpu(),
        "doc_ids": generator_captures[f"{prefix}.doc_ids"][-1].cpu(),
        "physical_to_logical": generator_captures[
            f"{prefix}.physical_to_logical"
        ][-1].cpu(),
        "blockmask_kv_num_blocks": generator_captures[
            f"{prefix}.blockmask_kv_num_blocks"
        ][-1].cpu(),
        "blockmask_kv_indices": generator_captures[
            f"{prefix}.blockmask_kv_indices"
        ][-1].cpu(),
        "blockmask_block_size": generator_captures[
            f"{prefix}.blockmask_block_size"
        ][-1].cpu(),
        "slot_mapping": generator_captures[f"{prefix}.slot_mapping"][-1][
            : len(prompt)
        ].cpu(),
        "cache_num_blocks": kv_cache.shape[0],
        "cache_block_size": cache_block_size,
    }
    path = os.path.join(dump_dir, f"flex_parity_rank{rank}.pt")
    torch.save(payload, path)
    return path


def _extract_logprobs_from_prompt(output, token_ids, start_pos: int = 0):
    """Extract per-token logprobs from vLLM prompt_logprobs starting at start_pos."""
    logprobs = []
    for i, lp_dict in enumerate(output.prompt_logprobs):
        if lp_dict is None or i < start_pos:
            continue
        tok = token_ids[i]
        if tok in lp_dict:
            logprobs.append(lp_dict[tok].logprob)
        else:
            logprobs.append(max(lp_dict.values(), key=lambda x: x.logprob).logprob)
    return logprobs


# ---------------------------------------------------------------------------
# vLLM operations
# ---------------------------------------------------------------------------

_PREFILL_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1,
    logprobs=1,
    prompt_logprobs=1,
    output_kind=RequestOutputKind.FINAL_ONLY,
)


def _run_engine(engine, request_prefix, batched_ids, sampling_params):
    """Submit requests to vLLM, run to completion, return outputs sorted by ID."""
    for i, ids in enumerate(batched_ids):
        engine.add_request(
            f"{request_prefix}_{i}", {"prompt_token_ids": ids}, sampling_params
        )
    outputs = []
    while engine.has_unfinished_requests():
        outputs.extend(engine.step())
    outputs.sort(key=lambda o: o.request_id)
    assert len(outputs) == len(batched_ids)
    return outputs


def vllm_prefill(engine, all_prompt_ids):
    """Run vLLM prefill and extract prompt logprobs for each sequence."""
    outputs = _run_engine(engine, "prefill", all_prompt_ids, _PREFILL_PARAMS)
    return [
        _extract_logprobs_from_prompt(out, ids)
        for out, ids in zip(outputs, all_prompt_ids, strict=True)
    ]


def vllm_generate(engine, all_prompt_ids, max_tokens):
    """Generate tokens and return (generated_ids, decode_logprobs) per sequence."""
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        logprobs=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    outputs = _run_engine(engine, "generate", all_prompt_ids, params)
    all_ids, all_lps = [], []
    for out in outputs:
        sample = out.outputs[0]
        all_ids.append(list(sample.token_ids))
        all_lps.append([list(d.values())[0].logprob for d in sample.logprobs])
    return all_ids, all_lps


def vllm_2nd_pass_prefill(engine, all_prompt_ids, all_gen_ids):
    """Re-prefill [prompt + generated] and extract logprobs for generated positions."""
    all_combined = [
        list(p) + list(g) for p, g in zip(all_prompt_ids, all_gen_ids, strict=True)
    ]
    outputs = _run_engine(engine, "2nd_prefill", all_combined, _PREFILL_PARAMS)
    return [
        _extract_logprobs_from_prompt(out, combined, start_pos=len(p))
        for out, combined, p in zip(outputs, all_combined, all_prompt_ids, strict=True)
    ]


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

_FILLER_TEXT = (
    "You are a highly skilled mathematician and teacher. Your goal is to "
    "solve complex mathematical problems with detailed step-by-step reasoning. "
    "When presented with a problem, first identify the type of problem and "
    "the relevant mathematical concepts. Then, break down the solution into "
    "clear logical steps. Show all intermediate calculations and explain "
    "each transformation. Finally, verify your answer by substituting back "
    "or using an alternative method. Be precise with notation and careful "
    "with arithmetic. If the problem has multiple valid approaches, mention "
    "the alternatives briefly. Always state your final answer clearly."
)


def _make_prompt_tokens(batch_size, prompt_length, tokenizer):
    """Create token ID sequences of the given prompt_length.

    For batch_size > 1, varies lengths across the batch (first seq at max,
    last seq ~40% of max) to test batch invariance with mixed lengths.
    """
    all_sequences = []
    for idx in range(batch_size):
        frac = 1.0 - (idx * 0.6 / max(batch_size - 1, 1))
        target_len = max(16, int(prompt_length * frac))

        text = ""
        while True:
            text += _FILLER_TEXT + " "
            tokens = tokenizer.encode(text)
            if len(tokens) >= target_len:
                break
        all_sequences.append(tokens[:target_len])

    return all_sequences


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(
    dist.is_initialized() or "RANK" in os.environ,
    "requires torchrun launcher",
)
class BitwiseParityTestBase(unittest.TestCase):
    """Base class for bitwise parity tests. Subclass and set config_fn / attn_backend."""

    __test__ = False

    BATCH_SIZE = 5
    PROMPT_LENGTH = 150
    MAX_GEN_TOKENS = 50

    config_fn = staticmethod(rl_grpo_qwen3_0_6b_varlen_batch_invariant)
    attn_backend: str = "varlen"
    min_world_size: int = 1
    hf_assets_env_var: str = "HF_ASSETS_PATH"
    # Root dir for run artifacts (NCCL flight-recorder comm_traces). CI sets it
    # to $RUNNER_TEMP/artifacts-to-be-uploaded so traces are uploaded; when unset
    # (local runs) a temp dir is used instead.
    dump_folder_env_var: str = "RL_TEST_DUMP_FOLDER"
    # When True, vLLM skips its own checkpoint load and the trainer model's
    # weights (including attention sinks) are copied into the engine in-process.
    sync_weights_from_trainer: bool = False

    # Shared across all tests in the class (built once in setUpClass)
    model: torch.nn.Module
    engine: LLMEngine
    prompt_ids: list[list[int]]

    @classmethod
    def setUpClass(cls):
        world_size = (
            dist.get_world_size()
            if dist.is_initialized()
            else int(os.environ.get("WORLD_SIZE", "1"))
        )
        if world_size < cls.min_world_size:
            raise unittest.SkipTest(
                f"requires at least {cls.min_world_size} GPUs, got {world_size}"
            )

        config = cls.config_fn()
        capture_intermediates = (
            os.environ.get("RL_FLEX_PARITY_DIAGNOSTICS", "0") == "1"
            and cls.attn_backend == "flex"
        )
        if capture_intermediates:
            config.generator.cuda_graph = dataclasses.replace(
                config.generator.cuda_graph,
                mode="NONE",
            )
        hf_path = os.environ.get(cls.hf_assets_env_var)
        if hf_path:
            config.hf_assets_path = hf_path

        from torchtitan.tools.utils import get_cuda_flash_attention_impl

        flash_attention_impl = get_cuda_flash_attention_impl()
        if flash_attention_impl is not None:
            from torch.nn.attention import (
                activate_flash_attention_impl,
                current_flash_attention_impl,
            )

            if current_flash_attention_impl() != flash_attention_impl:
                activate_flash_attention_impl(flash_attention_impl)

        # Enable batch-invariant mode BEFORE init_distributed
        set_batch_invariance(config.trainer.debug.batch_invariant)

        if not dist.is_initialized():
            # Writable base_folder for comm_traces
            dump_root = os.environ.get(cls.dump_folder_env_var)
            if dump_root:
                base_folder = os.path.join(dump_root, cls.__name__)
                os.makedirs(base_folder, exist_ok=True)
            else:
                base_folder = tempfile.mkdtemp(prefix="rl_bitwise_")
                cls.addClassCleanup(shutil.rmtree, base_folder, ignore_errors=True)
            dist_utils.init_distributed(CommConfig(), base_folder=base_folder)

        sync_weights_from_trainer = (
            cls.sync_weights_from_trainer or capture_intermediates
        )
        if sync_weights_from_trainer:
            generator_checkpointer = None
        else:
            generator_checkpointer = CheckpointManager.Config(
                initial_load_in_hf=True,
                initial_load_path=config.hf_assets_path,
            )

        if config.override.imports:
            apply_overrides(config.override, config)
        register_to_vllm(
            config.generator.model,
            parallelism=config.generator.parallelism,
            checkpointer_config=generator_checkpointer,
        )

        # Test runs trainer and generator in the same process, so limit
        # GPU memory for vLLM to leave room for the trainer model.
        config.generator.gpu_memory_limit = 0.5

        cls.model, cls.device, cls.parallelism_context = build_trainer_model(config)

        # Workaround: Clear the DTensor sharding-propagation caches here since
        # this test puts both the trainer and the generator in the same process
        # and builds equal but separate meshes for them. Without this workaround,
        # the generator's FSDP params pick up the trainer's mesh objects and
        # fail fully_shard's mesh identity check. This error happens only in
        # this specific test setup, no production impact.
        #
        # TODO: Revisit DTensor sharding-propagation cache semantics on pytorch side
        torch._C._clear_DTensor_sharding_propagator_cache()
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_clear()

        cls.engine = build_inference_engine(config)
        if sync_weights_from_trainer:
            _sync_trainer_weights_to_vllm(cls.model, cls.engine)

        tokenizer = cls.engine.get_tokenizer()
        cls.prompt_ids = _make_prompt_tokens(
            cls.BATCH_SIZE, cls.PROMPT_LENGTH, tokenizer
        )

    @classmethod
    def tearDownClass(cls):
        # Each class runs in its own torchrun process. Synchronize before
        # releasing local resources, then leave process-group cleanup to exit.
        # vLLM's internal shutdown destroys every registered process group,
        # including the trainer's groups, and deadlocks during global teardown.
        if dist.is_initialized():
            dist.barrier()
        if hasattr(cls, "engine"):
            renderer = getattr(cls.engine, "renderer", None)
            if renderer is not None:
                renderer.shutdown()
            del cls.engine
        if hasattr(cls, "model"):
            del cls.model
        gc.collect()
        torch.cuda.empty_cache()

    def _assert_logprobs_equal(self, name, a, b, label_a="A", label_b="B"):
        """Assert two logprob sequences are bitwise identical."""
        if isinstance(a, list):
            a = torch.tensor(a, dtype=torch.float32)
        else:
            a = a.detach().cpu().float()
        if isinstance(b, list):
            b = torch.tensor(b, dtype=torch.float32)
        else:
            b = b.detach().cpu().float()
        self.assertEqual(
            len(a),
            len(b),
            f"{name}: length mismatch ({label_a}={len(a)}, {label_b}={len(b)})",
        )
        n = len(a)

        max_delta = (a - b).abs().max().item() if n > 0 else 0.0
        num_diff = (a != b).sum().item()
        print(
            f"  {name}: max_delta={max_delta:.2e}, "
            f"num_diff={num_diff}/{n}, "
            f"bitwise_equal={torch.equal(a, b)}"
        )
        self.assertTrue(
            torch.equal(a, b),
            f"{name}: NOT bitwise identical (max_delta={max_delta:.2e})\n"
            f"  {label_a}[:5]: {a[:5].tolist()}\n"
            f"  {label_b}[:5]: {b[:5].tolist()}",
        )

    def test_batch_invariance(self):
        """Trainer prefill(bsz=m) == Trainer prefill(bsz=n) for shared sequences.

        Guards that model kernels are batch-invariant: the same sequence must
        produce bit-identical logits regardless of what other sequences are
        in the batch.
        """
        model = self.model
        n = len(self.prompt_ids)
        mid = max(1, n // 2)

        with type(self).parallelism_context.activate_spmd(), torch.no_grad():
            lps_partial = compute_trainer_prefill_logprobs(
                model,
                self.prompt_ids[:mid],
                self.device,
                attn_backend=self.attn_backend,
            )
            lps_full = compute_trainer_prefill_logprobs(
                model, self.prompt_ids, self.device, attn_backend=self.attn_backend
            )

        if dist.get_rank() == 0:
            for i in range(mid):
                # ``prompt_ids[:mid]`` is a list of sequences, so
                # ``compute_trainer_prefill_logprobs`` always returns a list
                # (even for mid==1) -> index per-sequence.
                partial_lp = lps_partial[i]
                self._assert_logprobs_equal(
                    f"seq {i}: prefill(bsz={mid}) vs prefill(bsz={n})",
                    partial_lp,
                    lps_full[i],
                    f"bsz={mid}",
                    f"bsz={n}",
                )

    def test_trainer_vs_vllm_prefill(self):
        """Trainer prefill == vLLM prefill (prompt-only).

        Ensures the trainer model forward and generator model forward produce
        bitwise identical logprobs.
        """
        model = self.model
        engine = self.engine

        with type(self).parallelism_context.activate_spmd(), torch.no_grad():
            trainer_lps = compute_trainer_prefill_logprobs(
                model, self.prompt_ids, self.device, attn_backend=self.attn_backend
            )

        vllm_lps = vllm_prefill(engine, self.prompt_ids)

        if dist.get_rank() == 0:
            for i in range(len(self.prompt_ids)):
                self._assert_logprobs_equal(
                    f"seq {i}: Trainer prefill vs vLLM prefill",
                    trainer_lps[i],
                    vllm_lps[i],
                    "Trainer",
                    "vLLM",
                )

    def test_vllm_decode_vs_prefill(self):
        """vLLM decode == vLLM 2nd-pass prefill (generated positions).

        Ensures prefill-stage attention and decode-stage KV-cache attention
        produce bitwise identical logprobs.
        """
        engine = self.engine

        gen_ids, decode_lps = vllm_generate(
            engine, self.prompt_ids, self.MAX_GEN_TOKENS
        )
        prefill_2nd_lps = vllm_2nd_pass_prefill(engine, self.prompt_ids, gen_ids)

        if dist.get_rank() == 0:
            for i in range(len(self.prompt_ids)):
                self._assert_logprobs_equal(
                    f"seq {i}: vLLM decode vs vLLM 2nd-pass prefill",
                    decode_lps[i],
                    prefill_2nd_lps[i],
                    "Decode",
                    "2ndPrefill",
                )

    def _check_vllm_prefill_and_decode_batch_invariance(self):
        """Check that vLLM prefill and decode do not depend on batch composition."""
        single_prompt = self.prompt_ids[:1]

        single_prefill_lps = vllm_prefill(self.engine, single_prompt)
        batched_prefill_lps = vllm_prefill(self.engine, self.prompt_ids)

        single_gen_ids, single_decode_lps = vllm_generate(
            self.engine, single_prompt, self.MAX_GEN_TOKENS
        )
        batched_gen_ids, batched_decode_lps = vllm_generate(
            self.engine, self.prompt_ids, self.MAX_GEN_TOKENS
        )

        if dist.get_rank() == 0:
            self._assert_logprobs_equal(
                "seq 0: vLLM prefill(bsz=1) vs prefill(bsz=3)",
                single_prefill_lps[0],
                batched_prefill_lps[0],
                "bsz=1",
                "bsz=3",
            )
            self.assertEqual(
                single_gen_ids[0],
                batched_gen_ids[0],
                "seq 0: greedy decode token IDs differ by batch composition",
            )
            self._assert_logprobs_equal(
                "seq 0: vLLM decode(bsz=1) vs decode(bsz=3)",
                single_decode_lps[0],
                batched_decode_lps[0],
                "bsz=1",
                "bsz=3",
            )


class TestBitwiseParityVarlen(BitwiseParityTestBase):
    """Bitwise parity tests using varlen attention."""

    __test__ = True
    config_fn = staticmethod(rl_grpo_qwen3_0_6b_varlen_batch_invariant)
    attn_backend = "varlen"


class TestBitwiseParityFlex(BitwiseParityTestBase):
    """Bitwise parity tests using flex attention."""

    __test__ = True
    config_fn = staticmethod(rl_grpo_qwen3_0_6b_flex_batch_invariant)
    attn_backend = "flex"

    @unittest.skipUnless(
        os.environ.get("RL_FLEX_PARITY_DIAGNOSTICS", "0") == "1",
        "set RL_FLEX_PARITY_DIAGNOSTICS=1 to capture intermediate tensors",
    )
    def test_first_trainer_vllm_divergence(self):
        """Report the first unequal trainer/vLLM tensor for one prompt."""
        trainer_captures: dict[str, list[torch.Tensor]] = {}
        generator_captures: dict[str, list[torch.Tensor]] = {}
        trainer_handles = _register_parity_capture_hooks(self.model, trainer_captures)
        generator_wrapper = self.engine.model_executor.driver_worker.get_model()
        generator_handles = _register_parity_capture_hooks(
            generator_wrapper.model,
            generator_captures,
        )

        prompt = self.prompt_ids[0]
        try:
            with type(self).parallelism_context.activate_spmd(), torch.no_grad():
                trainer_logprobs = compute_trainer_prefill_logprobs(
                    self.model,
                    [prompt],
                    self.device,
                    attn_backend="flex",
                )[0]
            generator_logprobs = torch.tensor(
                vllm_prefill(self.engine, [prompt])[0],
                dtype=torch.float32,
                device=self.device,
            )
        finally:
            for handle in trainer_handles + generator_handles:
                handle.remove()

        replay_q = generator_captures["layer2.q"][0]
        replay_k = generator_captures["layer2.k"][0]
        replay_v = generator_captures["layer2.v"][0]
        replay_positions = torch.arange(len(prompt), device=self.device)
        replay_metadata = self.model._get_attention_metadata(replay_positions)
        replay_block_mask = next(iter(replay_metadata.values()))
        replay_attention = self.model.layers["2"].attention
        with type(self).parallelism_context.activate_spmd(), torch.no_grad():
            contiguous_replay = replay_attention.inner_attention(
                replay_q,
                replay_k,
                replay_v,
                attention_metadata=replay_block_mask,
                scale=replay_attention.scaling,
                enable_gqa=replay_attention.enable_gqa,
            )

        boundaries = ["embedding"]
        for layer_name in self.model.layers:
            prefix = f"layer{layer_name}"
            boundaries.extend(
                [
                    f"{prefix}.input",
                    f"{prefix}.attention_norm",
                    f"{prefix}.q",
                    f"{prefix}.k",
                    f"{prefix}.v",
                    f"{prefix}.inner_attention",
                    f"{prefix}.wo",
                    f"{prefix}.attention",
                    f"{prefix}.post_attention_residual",
                    f"{prefix}.ffn_norm",
                    f"{prefix}.feed_forward",
                    f"{prefix}.output",
                ]
            )
        boundaries.extend(["final_norm", "logits"])
        paged_k, paged_v = _reconstruct_paged_kv(
            generator_captures,
            "layer2",
            len(prompt),
        )
        dump_dir = os.environ.get("RL_FLEX_PARITY_DUMP_DIR")
        if dump_dir:
            dump_path = _dump_flex_parity_capture(
                dump_dir,
                rank=dist.get_rank(),
                prompt=prompt,
                trainer_captures=trainer_captures,
                generator_captures=generator_captures,
                contiguous_replay=contiguous_replay,
                paged_k=paged_k,
                paged_v=paged_v,
                scale=replay_attention.scaling,
                enable_gqa=replay_attention.enable_gqa,
            )
            print(f"  rank={dist.get_rank()} saved parity capture to {dump_path}")
        first_mismatch = None
        for name in boundaries:
            trainer = _select_prefill_capture(trainer_captures, name)
            generator = _select_prefill_capture(generator_captures, name)
            boundary_tokens = len(prompt) - 1 if name == "logits" else len(prompt)
            equal, summary = _compare_capture(
                name,
                trainer,
                generator,
                boundary_tokens,
            )
            print(f"  rank={dist.get_rank()} {summary}")
            if not equal and first_mismatch is None:
                first_mismatch = summary
            if name == "layer2.v":
                for cache_name, source, reconstructed in (
                    ("layer2.paged_k", generator_captures["layer2.k"][0], paged_k),
                    ("layer2.paged_v", generator_captures["layer2.v"][0], paged_v),
                ):
                    cache_equal, cache_summary = _compare_capture(
                        cache_name,
                        source,
                        reconstructed,
                        len(prompt),
                    )
                    print(f"  rank={dist.get_rank()} {cache_summary}")
                    if not cache_equal and first_mismatch is None:
                        first_mismatch = cache_summary
            if name == "layer2.inner_attention":
                for replay_name, reference in (
                    ("layer2.contiguous_replay_vs_trainer", trainer),
                    ("layer2.contiguous_replay_vs_vllm", generator),
                ):
                    _, replay_summary = _compare_capture(
                        replay_name,
                        reference,
                        contiguous_replay,
                        len(prompt),
                    )
                    print(f"  rank={dist.get_rank()} {replay_summary}")

        logprobs_equal, logprobs_summary = _compare_capture(
            "logprobs",
            trainer_logprobs,
            generator_logprobs,
            len(prompt) - 1,
        )
        print(f"  rank={dist.get_rank()} {logprobs_summary}")
        if not logprobs_equal and first_mismatch is None:
            first_mismatch = logprobs_summary

        self.assertIsNone(
            first_mismatch,
            f"first trainer/vLLM divergence on rank {dist.get_rank()}: "
            f"{first_mismatch}",
        )


class TestBitwiseParityQwen35Varlen(BitwiseParityTestBase):
    """Test Qwen3.5 trainer/generator parity with head-sharded GDN at TP=2."""

    __test__ = True
    config_fn = staticmethod(rl_grpo_qwen3_5_9b_varlen_batch_invariant)
    attn_backend = "varlen"
    min_world_size = 2


def _qwen3_5_debug_bitwise_config() -> Controller.Config:
    """Build a two-GPU random-weight Qwen3.5 parity configuration."""
    config = rl_grpo_qwen3_5_debug_varlen_batch_invariant()
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=1,
        ),
    )
    return config


class TestBitwiseParityQwen35DebugVarlen(BitwiseParityTestBase):
    """Qwen3.5 GDN parity with random weights and matched TP=2."""

    __test__ = True
    config_fn = staticmethod(_qwen3_5_debug_bitwise_config)
    attn_backend = "varlen"
    min_world_size = 2
    sync_weights_from_trainer = True
    BATCH_SIZE = 3
    PROMPT_LENGTH = 64
    MAX_GEN_TOKENS = 16

    def test_vllm_prefill_and_decode_batch_invariance(self):
        """vLLM prefill and decode must not depend on batch composition."""
        self._check_vllm_prefill_and_decode_batch_invariance()


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() >= (9, 0),
    "Batch-invariant Attention Gym KDA uses fixed-tile TMA kernels (SM90+)",
)
class KimiK3BitwiseParityTestBase(BitwiseParityTestBase):
    """Run Kimi K3 parity checks with the trainer in training mode."""

    __test__ = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model.train()


def _kimi_k3_debug_bitwise_config() -> Controller.Config:
    """Build a one-GPU random-weight Kimi K3 parity configuration."""
    config = rl_grpo_kimi_k3_debug_varlen_batch_invariant()
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=1,
        ),
    )
    return config


class TestBitwiseParityKimiK3DebugVarlen(KimiK3BitwiseParityTestBase):
    """Kimi K3 KDA/MLA parity with random weights and matched TP=1."""

    __test__ = True
    config_fn = staticmethod(_kimi_k3_debug_bitwise_config)
    attn_backend = "varlen"
    sync_weights_from_trainer = True
    BATCH_SIZE = 3
    PROMPT_LENGTH = 64
    MAX_GEN_TOKENS = 128

    def test_vllm_prefill_and_decode_batch_invariance(self):
        """vLLM prefill and decode must not depend on batch composition."""
        self._check_vllm_prefill_and_decode_batch_invariance()


class TestBitwiseParityMoEEP(BitwiseParityTestBase):
    """Test bitwise parity between trainer and vLLM generator with MoE EP.

    On 4 GPUs: trainer uses dp_shard=2, TP=2, EP=4; the generator maps
    dp_shard=2 to vLLM data parallelism with TP=2, EP=4.

    Uses the bundled debug MoE assets by default. Override with
    MOE_HF_ASSETS_PATH if needed.
    """

    __test__ = True

    BATCH_SIZE = 3
    PROMPT_LENGTH = 100
    MAX_GEN_TOKENS = 30

    config_fn = staticmethod(rl_grpo_qwen3_moe_debug_varlen_batch_invariant)
    attn_backend = "varlen"
    min_world_size = 4
    hf_assets_env_var = "MOE_HF_ASSETS_PATH"


class TestBitwiseParityGptOssVarlen(BitwiseParityTestBase):
    """Bitwise parity for GPT-OSS varlen attention."""

    __test__ = True
    config_fn = staticmethod(rl_grpo_gpt_oss_debug_varlen_batch_invariant)
    attn_backend = "varlen"
    sync_weights_from_trainer = True


if __name__ == "__main__":
    init_logger()
    unittest.main()
