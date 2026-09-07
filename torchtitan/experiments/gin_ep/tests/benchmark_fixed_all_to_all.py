import argparse
import os

import torch
import torch.distributed as dist

from torchtitan.experiments.gin_ep.api import (
    create_communicator,
    fixed_all_to_all_out,
    get_unique_id,
)


def _time_ms(operation, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier()
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    dist.barrier()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes-per-peer", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    unique_id = get_unique_id() if rank == 0 else bytes(128)
    unique_id_tensor = torch.tensor(
        list(unique_id),
        dtype=torch.uint8,
        device=local_rank,
    )
    dist.broadcast(unique_id_tensor, src=0)

    comm = create_communicator(
        bytes(unique_id_tensor.cpu().tolist()),
        rank=rank,
        world_size=world_size,
        device=local_rank,
    )
    total_bytes = world_size * args.bytes_per_peer
    comm.allocate_symmetric_buffers(bytes=total_bytes)
    comm.create_device_communicator(cta_count=64)

    payload = torch.randint(
        0,
        256,
        (world_size, args.bytes_per_peer),
        dtype=torch.uint8,
        device=local_rank,
    )
    rccl_output = torch.empty_like(payload)
    gin_output = torch.empty_like(payload)

    gin_operation = lambda: fixed_all_to_all_out(
        comm,
        payload,
        gin_output,
    )
    rccl_operation = lambda: dist.all_to_all_single(rccl_output, payload)

    for _ in range(args.warmup):
        gin_operation()
        rccl_operation()
    torch.cuda.synchronize()

    gin_output = gin_operation()
    rccl_operation()
    torch.testing.assert_close(gin_output, rccl_output, rtol=0, atol=0)

    gin_ms = _time_ms(gin_operation, args.iterations)
    rccl_ms = _time_ms(rccl_operation, args.iterations)
    max_times = torch.tensor(
        [gin_ms, rccl_ms],
        dtype=torch.float64,
        device=local_rank,
    )
    dist.all_reduce(max_times, op=dist.ReduceOp.MAX)
    gin_ms, rccl_ms = max_times.cpu().tolist()

    if rank == 0:
        print(
            f"bytes_per_peer={args.bytes_per_peer} "
            f"world_size={world_size} "
            f"gin_ms={gin_ms:.3f} "
            f"rccl_ms={rccl_ms:.3f} "
            f"speedup={rccl_ms / gin_ms:.3f}x",
            flush=True,
        )

    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
