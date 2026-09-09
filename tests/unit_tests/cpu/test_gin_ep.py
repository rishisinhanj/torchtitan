import torch

from torchtitan.experiments.gin_ep.api import variable_all_to_all


class _FakeCommunicator:
    world_size = 2

    def __init__(self) -> None:
        self.operations: list[int] = []

    def fixed_all_to_all(
        self,
        input: torch.Tensor,
        operation: int = 0,
    ) -> torch.Tensor:
        self.operations.append(operation)
        return input.flip(0)


def test_variable_all_to_all_balanced_fast_path_and_backward() -> None:
    communicator = _FakeCommunicator()
    input = torch.tensor(
        [[10.0], [11.0], [20.0], [21.0]],
        requires_grad=True,
    )

    output = variable_all_to_all(
        communicator,
        input,
        input_splits=[2, 2],
        output_splits=[2, 2],
        capacity_per_peer=2,
        operation="dispatch",
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[20.0], [21.0], [10.0], [11.0]]),
    )
    output.backward(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
    torch.testing.assert_close(
        input.grad,
        torch.tensor([[3.0], [4.0], [1.0], [2.0]]),
    )
    assert communicator.operations == [1, 3]


def test_variable_all_to_all_packing_and_backward() -> None:
    communicator = _FakeCommunicator()
    input = torch.tensor([[10.0], [20.0], [21.0]], requires_grad=True)

    output = variable_all_to_all(
        communicator,
        input,
        input_splits=[1, 2],
        output_splits=[2, 1],
        capacity_per_peer=2,
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[20.0], [21.0], [10.0]]),
    )
    output.backward(torch.tensor([[1.0], [2.0], [3.0]]))
    torch.testing.assert_close(
        input.grad,
        torch.tensor([[3.0], [1.0], [2.0]]),
    )


def test_variable_all_to_all_rejects_capacity_overflow() -> None:
    communicator = _FakeCommunicator()
    input = torch.zeros((3, 1))

    try:
        variable_all_to_all(
            communicator,
            input,
            input_splits=[1, 2],
            output_splits=[2, 1],
            capacity_per_peer=1,
        )
    except ValueError as error:
        assert "input split exceeds" in str(error)
    else:
        raise AssertionError("capacity overflow was accepted")
