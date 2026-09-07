from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


_HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_extension():
    rccl_prefix = Path(
        os.environ.get("GIN_RCCL_PREFIX", "/workspace/rccl")
    )
    rocm_prefix = Path(
        os.environ.get(
            "ROCM_HOME",
            os.environ.get("ROCM_PATH", "/opt/rocm"),
        )
    )
    include_dir = rccl_prefix / "include"
    lib_dir = rccl_prefix / "lib"
    rocm_include_dir = rocm_prefix / "include"
    rocm_lib_dir = rocm_prefix / "lib"
    backend_library = lib_dir / "libgin_ep_backend.so"
    source = _HERE / "csrc" / "gin_ep.cu"

    if not (include_dir / "nccl.h").exists():
        raise RuntimeError(
            f"RCCL headers not found under {include_dir}. "
            "Set GIN_RCCL_PREFIX to the Stage 2 RCCL installation."
        )
    if not backend_library.exists():
        raise RuntimeError(
            f"GIN backend library not found at {backend_library}. "
            "Rebuild the Stage 2 integration image."
        )
    if "libgin_ep_backend.so" not in os.environ.get("LD_PRELOAD", ""):
        raise RuntimeError(
            "libgin_ep_backend.so must be in LD_PRELOAD before Python starts "
            "so RCCL and the GDA kernel share one rocSHMEM device image."
        )

    hip_compiler = rocm_prefix / "bin" / "amdclang++"
    if "CXX" not in os.environ and hip_compiler.exists():
        os.environ["CXX"] = str(hip_compiler)

    return load(
        name="torchtitan_gin_stage2",
        sources=[str(source)],
        extra_include_paths=[
            str(include_dir),
            str(rocm_include_dir),
        ],
        extra_cflags=[
            "-O2",
            "-DENABLE_DEVICE_API",
            "-DNCCL_OS_LINUX",
        ],
        extra_cuda_cflags=[
            "-O2",
            "-DENABLE_DEVICE_API",
            "-DNCCL_OS_LINUX",
        ],
        extra_ldflags=[
            f"-L{lib_dir}",
            f"-L{rocm_lib_dir}",
            "-lgin_ep_backend",
            "-lrccl",
            "-lamdhip64",
            f"-Wl,-rpath,{lib_dir}",
            f"-Wl,-rpath,{rocm_lib_dir}",
        ],
        with_cuda=True,
        verbose=os.environ.get("GIN_BUILD_VERBOSE") == "1",
    )


def version_info() -> tuple[int, int]:
    header_version, runtime_version = _load_extension().version_info()
    return int(header_version), int(runtime_version)


def get_unique_id() -> bytes:
    return _load_extension().get_unique_id()


def create_communicator(
    unique_id: bytes,
    *,
    rank: int,
    world_size: int,
    device: int,
):
    return _load_extension().GinCommunicator.create(
        unique_id=unique_id,
        rank=rank,
        world_size=world_size,
        device=device,
    )


class _FixedAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        communicator: Any,
        input: torch.Tensor,
    ) -> torch.Tensor:
        ctx.communicator = communicator
        return communicator.fixed_all_to_all(input)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor]:
        return (
            None,
            ctx.communicator.fixed_all_to_all(
                grad_output.contiguous()
            ),
        )


def fixed_all_to_all(
    communicator: Any,
    input: torch.Tensor,
) -> torch.Tensor:
    return _FixedAllToAll.apply(communicator, input)


def fixed_all_to_all_out(
    communicator: Any,
    input: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    return communicator.fixed_all_to_all_out(input, output)


def variable_all_to_all(
    communicator: Any,
    input: torch.Tensor,
    *,
    input_splits: list[int],
    output_splits: list[int],
    capacity_per_peer: int,
) -> torch.Tensor:
    world_size = communicator.world_size
    if len(input_splits) != world_size:
        raise ValueError("input_splits must contain one entry per rank")
    if len(output_splits) != world_size:
        raise ValueError("output_splits must contain one entry per rank")
    if any(count < 0 or count > capacity_per_peer for count in input_splits):
        raise ValueError("input split exceeds the per-peer capacity")
    if any(count < 0 or count > capacity_per_peer for count in output_splits):
        raise ValueError("output split exceeds the per-peer capacity")
    if sum(input_splits) != input.shape[0]:
        raise ValueError("input_splits do not sum to the input row count")

    # Balanced routing already lays out one full, contiguous capacity slice per
    # peer. Preserve that layout across the fixed all-to-all instead of
    # allocating/zeroing a packed tensor and concatenating the result.
    balanced_splits = [capacity_per_peer] * world_size
    if input_splits == balanced_splits and output_splits == balanced_splits:
        with torch.profiler.record_function(
            "gin_phase::balanced_fixed_exchange"
        ):
            packed = input.view(
                world_size,
                capacity_per_peer,
                *input.shape[1:],
            )
            return fixed_all_to_all(communicator, packed).view_as(input)

    with torch.profiler.record_function("gin_phase::variable_pack"):
        packed = input.new_zeros(
            (world_size, capacity_per_peer, *input.shape[1:])
        )
        input_offset = 0
        for peer, count in enumerate(input_splits):
            packed[peer, :count].copy_(
                input[input_offset : input_offset + count]
            )
            input_offset += count

    with torch.profiler.record_function("gin_phase::fixed_exchange"):
        received = fixed_all_to_all(communicator, packed)
    with torch.profiler.record_function("gin_phase::variable_unpack"):
        return torch.cat(
            [
                received[peer, :count]
                for peer, count in enumerate(output_splits)
            ],
            dim=0,
        )