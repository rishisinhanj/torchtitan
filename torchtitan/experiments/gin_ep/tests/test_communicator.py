import os

import torch
import torch.distributed as dist

from torchtitan.experiments.gin_ep.api import (
    create_communicator,
    fixed_all_to_all,
    get_unique_id,
    variable_all_to_all,
    version_info,
)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    comm = None
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == 2, f"expected 2 ranks, got {world_size}"

        header_version, runtime_version = version_info()
        assert header_version == runtime_version

        unique_id = get_unique_id() if rank == 0 else bytes(128)
        unique_id_tensor = torch.tensor(
            list(unique_id),
            dtype=torch.uint8,
            device=local_rank,
        )
        dist.broadcast(unique_id_tensor, src=0)
        unique_id = bytes(unique_id_tensor.cpu().tolist())

        comm = create_communicator(
            unique_id,
            rank=rank,
            world_size=world_size,
            device=local_rank,
        )
        assert comm.rank == rank
        assert comm.world_size == world_size
        assert comm.device == local_rank
        assert not comm.closed

        properties = comm.query_properties()
        assert set(properties) == {
            "rank",
            "world_size",
            "device",
            "nvml_device",
            "device_api_support",
            "multimem_support",
            "gin_type",
            "num_lsa_teams",
            "host_rma_support",
            "railed_gin_type",
        }
        assert properties["rank"] == comm.rank == rank
        assert properties["world_size"] == comm.world_size == world_size
        assert properties["device"] == comm.device == local_rank
        assert properties["nvml_device"] >= 0
        assert type(properties["device_api_support"]) is bool
        assert type(properties["multimem_support"]) is bool
        assert type(properties["host_rma_support"]) is bool
        assert type(properties["gin_type"]) is int
        assert type(properties["railed_gin_type"]) is int
        assert properties["num_lsa_teams"] >= 0

        assert properties["device_api_support"]
        assert properties["gin_type"] != 0

        comm.allocate_symmetric_buffers(bytes=4096)
        assert comm.symmetric_buffers_allocated
        assert comm.symmetric_buffer_bytes == 4096

        comm.create_device_communicator(cta_count=64)
        assert comm.device_communicator_created
        assert comm.device_cta_count == 64

        try:
            comm.create_device_communicator(cta_count=64)
        except RuntimeError as error:
            assert "has already been created" in str(error)
        else:
            raise AssertionError(
                "created a second device communicator"
            )

        comm.destroy_device_communicator()
        assert not comm.device_communicator_created
        comm.destroy_device_communicator()

        comm.release_symmetric_buffers()
        assert not comm.symmetric_buffers_allocated
        comm.release_symmetric_buffers()

        comm.allocate_symmetric_buffers(bytes=4096)
        comm.create_device_communicator(cta_count=64)
        assert comm.device_communicator_created

        device_info = comm.device_communicator_info()
        assert device_info["rank"] == rank
        assert device_info["world_size"] == world_size
        assert 1 <= device_info["lsa_size"] <= world_size
        assert device_info["gin_connection_count"] >= 1
        assert device_info["gin_type"] == 4
        assert device_info["gin_handle_available"]
        assert device_info["gin_signal_count"] >= 1
        assert device_info["gin_signals_available"]

        element = torch.arange(
            256,
            dtype=torch.int32,
            device=local_rank,
        )
        payload = torch.stack(
            [
                rank * 100_000 + peer * 1_000 + element
                for peer in range(world_size)
            ]
        )
        actual = fixed_all_to_all(comm, payload)
        expected = torch.stack(
            [
                peer * 100_000 + rank * 1_000 + element
                for peer in range(world_size)
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        differentiable_payload = payload.to(torch.float32)
        differentiable_payload.requires_grad_(True)
        fixed_all_to_all(comm, differentiable_payload).sum().backward()
        torch.testing.assert_close(
            differentiable_payload.grad,
            torch.ones_like(differentiable_payload),
            rtol=0,
            atol=0,
        )

        balanced_input = (
            torch.arange(12, dtype=torch.float32, device=local_rank)
            .view(4, 3)
            .add(rank * 1_000)
        )
        balanced_input.requires_grad_(True)
        expected_balanced = torch.empty_like(balanced_input)
        dist.all_to_all_single(expected_balanced, balanced_input.detach())
        actual_balanced = variable_all_to_all(
            comm,
            balanced_input,
            input_splits=[2, 2],
            output_splits=[2, 2],
            capacity_per_peer=2,
        )
        torch.testing.assert_close(
            actual_balanced,
            expected_balanced,
            rtol=0,
            atol=0,
        )
        output_gradient = torch.arange(
            actual_balanced.numel(),
            dtype=actual_balanced.dtype,
            device=local_rank,
        ).view_as(actual_balanced)
        expected_input_gradient = torch.empty_like(balanced_input)
        dist.all_to_all_single(expected_input_gradient, output_gradient)
        (actual_balanced * output_gradient).sum().backward()
        torch.testing.assert_close(
            balanced_input.grad,
            expected_input_gradient,
            rtol=0,
            atol=0,
        )

        input_splits = [1, 3] if rank == 0 else [2, 1]
        output_splits = [1, 2] if rank == 0 else [3, 1]
        variable_input = torch.arange(
            sum(input_splits) * 3,
            dtype=torch.float32,
            device=local_rank,
        ).view(-1, 3)
        variable_input = variable_input + rank * 1_000
        variable_input.requires_grad_(True)

        expected_variable = torch.empty(
            (sum(output_splits), 3),
            dtype=variable_input.dtype,
            device=local_rank,
        )
        dist.all_to_all_single(
            expected_variable,
            variable_input.detach(),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )
        actual_variable = variable_all_to_all(
            comm,
            variable_input,
            input_splits=input_splits,
            output_splits=output_splits,
            capacity_per_peer=4,
        )
        torch.testing.assert_close(
            actual_variable,
            expected_variable,
            rtol=0,
            atol=0,
        )
        actual_variable.sum().backward()
        torch.testing.assert_close(
            variable_input.grad,
            torch.ones_like(variable_input),
            rtol=0,
            atol=0,
        )
        try:
            variable_all_to_all(
                comm,
                variable_input.detach(),
                input_splits=[1, 3],
                output_splits=[1, 2],
                capacity_per_peer=2,
            )
        except ValueError as error:
            assert "input split exceeds" in str(error)
        else:
            raise AssertionError("accepted a split larger than capacity")

        dist.barrier()
        comm.close()
        assert comm.closed
        assert not comm.device_communicator_created
        assert not comm.symmetric_buffers_allocated
        comm.close()
        assert comm.closed

        try:
            comm.query_properties()
        except RuntimeError as error:
            assert "cannot query properties of a closed communicator" in str(
                error
            )
        else:
            raise AssertionError(
                "query_properties() succeeded after close()"
            )

        dist.barrier()
    finally:
        if comm is not None and not comm.closed:
            comm.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
