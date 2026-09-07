import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.distributed import utils as dist_utils
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    GINAllToAllTokenDispatcher,
)


def _expert_counts(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.bincount(
        expert_ids.flatten(),
        minlength=num_experts,
    )


def main() -> None:
    if not hasattr(torch.compiler, "_is_non_strict_tracing"):
        torch.compiler._is_non_strict_tracing = lambda: False

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    dist_utils.set_spmd_backend("partial_dtensor")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    ep_mesh = init_device_mesh(
        "cuda",
        (world_size,),
        mesh_dim_names=("ep",),
    )

    standard = AllToAllTokenDispatcher(
        AllToAllTokenDispatcher.Config(num_experts=2, top_k=1)
    )
    standard.wire_meshes(ep_mesh=ep_mesh)

    gin = GINAllToAllTokenDispatcher(
        GINAllToAllTokenDispatcher.Config(
            num_experts=2,
            top_k=1,
            hidden_dim=3,
            num_max_tokens_per_rank=4,
        )
    )
    gin.wire_meshes(ep_mesh=ep_mesh)

    x = (
        torch.arange(12, dtype=torch.float32, device=local_rank)
        .view(4, 3)
        .add(rank * 100)
    )
    x.requires_grad_(True)
    expert_ids = torch.tensor(
        [[0], [1], [1], [1]]
        if rank == 0
        else [[0], [0], [1], [1]],
        dtype=torch.int64,
        device=local_rank,
    )
    scores = torch.ones((4, 1), dtype=torch.float32, device=local_rank)
    counts = _expert_counts(expert_ids, num_experts=2)

    standard_routed, standard_counts, _ = standard.dispatch(
        x.detach(),
        scores,
        expert_ids,
        counts,
    )
    gin_routed, gin_counts, metadata = gin.dispatch(
        x,
        scores,
        expert_ids,
        counts,
    )
    torch.testing.assert_close(gin_routed, standard_routed, rtol=0, atol=0)
    torch.testing.assert_close(gin_counts, standard_counts, rtol=0, atol=0)

    combined = gin.combine(gin_routed, metadata, x)
    torch.testing.assert_close(combined, x, rtol=0, atol=0)
    combined.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)

    static_gin = GINAllToAllTokenDispatcher(
        GINAllToAllTokenDispatcher.Config(
            num_experts=2,
            top_k=1,
            hidden_dim=3,
            num_max_tokens_per_rank=4,
            capacity_factor=1.0,
            static_balanced_routing=True,
        )
    )
    static_gin.wire_meshes(ep_mesh=ep_mesh)
    balanced_x = x.detach().clone().requires_grad_(True)
    balanced_expert_ids = torch.tensor(
        [[0], [1], [0], [1]],
        dtype=torch.int64,
        device=local_rank,
    )
    balanced_counts = _expert_counts(balanced_expert_ids, num_experts=2)
    standard_balanced, standard_balanced_counts, _ = standard.dispatch(
        balanced_x.detach(),
        scores,
        balanced_expert_ids,
        balanced_counts,
    )
    gin_balanced, gin_balanced_counts, balanced_metadata = static_gin.dispatch(
        balanced_x,
        scores,
        balanced_expert_ids,
        balanced_counts,
    )
    assert balanced_metadata.input_splits == [2, 2]
    assert balanced_metadata.output_splits == [2, 2]
    torch.testing.assert_close(
        gin_balanced, standard_balanced, rtol=0, atol=0
    )
    torch.testing.assert_close(
        gin_balanced_counts, standard_balanced_counts, rtol=0, atol=0
    )
    balanced_combined = static_gin.combine(
        gin_balanced, balanced_metadata, balanced_x
    )
    torch.testing.assert_close(
        balanced_combined, balanced_x, rtol=0, atol=0
    )
    balanced_combined.sum().backward()
    torch.testing.assert_close(
        balanced_x.grad, torch.ones_like(balanced_x), rtol=0, atol=0
    )

    dist.barrier()
    assert gin._gin_communicator is not None
    gin._gin_communicator.close()
    assert static_gin._gin_communicator is not None
    static_gin._gin_communicator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
