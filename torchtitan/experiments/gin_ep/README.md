# GIN Expert-Parallel All-to-All Experiment

This experiment integrates the RCCL device API and rocSHMEM GDA into
TorchTitan's mixture-of-experts (MoE) token dispatcher. It keeps TorchTitan's
existing count exchange and token routing logic, but replaces dispatch and
combine payload transfers with a hybrid all-to-all:

- remote-node peers use GIN with the rocSHMEM GDA provider;
- same-node peers use RCCL Local Symmetric Access (LSA).

The current implementation targets AMD MI300X (`gfx942`) and eager execution.

## Architecture

The payload path is:

```text
TorchTitan MoE
  -> GINAllToAllTokenDispatcher
  -> variable_all_to_all() packing
  -> GinCommunicator.fixed_all_to_all()
  -> symmetric send/receive windows
  -> HybridAlltoAllKernel
       CTA 0: remote GIN puts
       CTA 1..N: local LSA copies
  -> unpack received tokens
```

The kernel is built into a preloaded shared library rather than the PyTorch JIT
extension. This is required because the GIN kernel and RCCL must use the same
rocSHMEM device-global provider state. Building them as separate device images
caused remote GDA operations to see an uninitialized NIC provider.

## New experiment files

- `api.py`: Python API, JIT extension loader, autograd wrapper, and variable
  split packing.
- `__init__.py`: public experiment exports.
- `csrc/gin_ep.cu`: C++/HIP PyBind extension that owns the RCCL communicator,
  symmetric allocations, registered windows, device communicator, and launch
  lifecycle.
- `tests/test_communicator.py`: two-rank communicator, window, kernel,
  correctness, gradient, and cleanup test.
- `tests/test_dispatcher.py`: two-rank parity test against TorchTitan's standard
  token dispatcher.
- `tests/benchmark_fixed_all_to_all.py`: fixed all-to-all latency comparison
  between GIN and regular RCCL.

Related files outside this directory:

- `../../../tests/unit_tests/cpu/test_gin_ep.py`: CPU tests for packing,
  backward ordering, and capacity validation.
- `../../../../container/gin_ep_backend.cu`: hybrid GIN+LSA device kernel.
- `../../../../container/Dockerfile`: Stage 2 RCCL/rocSHMEM build and preloaded
  kernel library.
- `../../../../container/smoke_collective.py`: basic RCCL all-reduce smoke test.
- `../../../../../scripts/run_gin_gda_lsa.sh`: two-node `rccl-tests` GIN run.
- `../../../../../scripts/run_rccl_default.sh`: matching RCCL baseline.
- `../../../../../scripts/analyze_final_sweeps.py`: validates and summarizes
  repeated benchmark logs.

All authoritative GPU sources use the `.cu` extension. They are still compiled
for ROCm: the standalone kernel uses `amdclang++ -x hip`, and PyTorch selects
its HIP extension path on a ROCm build. PyTorch HIPify may create a generated
`.hip` file while compiling; that file is not source and should not be
committed.

## TorchTitan changes

The integration makes small changes to existing TorchTitan files:

- `torchtitan/models/common/config_utils.py` accepts `comm_backend="gin"` and
  constructs `GINAllToAllTokenDispatcher.Config`.
- `torchtitan/models/common/token_dispatcher.py` adds the GIN dispatcher,
  validates fixed capacity, caches communicators by EP group and buffer
  configuration, and overrides only payload dispatch/combine.
- `torchtitan/trainer.py` closes cached GIN communicators during shutdown.
- `pyproject.toml` packages `csrc/*.cu` with the experiment.

No changes are required in RCCL or rocSHMEM source for the TorchTitan
integration; the container builds the existing Stage 2 branches.

## Requirements

- Linux hosts with AMD MI300X GPUs.
- A ROCm-enabled PyTorch build.
- Docker access to `/dev/kfd` and `/dev/dri`.
- RCCL built with `--rocshmem-gin`.
- The same container image on every node for multi-node runs.
- RDMA devices and working passwordless SSH between benchmark containers for
  the two-node `rccl-tests` script.

The examples below assume:

```bash
export ROOT=/scratch/users/rissinha/gin-stage2-clean
export IMAGE=torchtitan-gin-stage2:gfx942
cd "$ROOT"
```

## Build the integration image

Build from the clean Stage 2 root because the Dockerfile copies both
`rocm-systems` and `torchtitan-gin-integration`:

```bash
docker build \
  -f torchtitan-gin-integration/container/Dockerfile \
  -t "$IMAGE" \
  .
```

The build:

1. builds RCCL with rocSHMEM GIN for `gfx942`;
2. compiles `container/gin_ep_backend.cu` as HIP;
3. device-links rocSHMEM into `libgin_ep_backend.so`;
4. installs Stage 2 RCCL under `/workspace/rccl`;
5. sets `LD_PRELOAD=/workspace/rccl/lib/libgin_ep_backend.so`.

Do not remove the preload setting. The Python loader rejects startup without
it to avoid using two incompatible rocSHMEM device images.

## Common container settings

The GPU tests can be run with:

```bash
docker run --rm \
  --network host \
  --ipc host \
  --shm-size=64G \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --privileged \
  -e GIN_RCCL_PREFIX=/workspace/rccl \
  -e TORCH_EXTENSIONS_DIR=/tmp/torch-extensions \
  -e PYTORCH_ROCM_ARCH=gfx942 \
  -e NCCL_GIN_ENABLE=1 \
  -e NCCL_GIN_TYPE=4 \
  -e NCCL_CUMEM_ENABLE=1 \
  -e NCCL_DMABUF_ENABLE=1 \
  -e RCCL_ENABLE_INTRANET=1 \
  -e NCCL_MSCCL_ENABLE=0 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -v "$ROOT/torchtitan-gin-integration/torchtitan:/workspace/torchtitan" \
  -w /workspace/torchtitan \
  "$IMAGE" \
  bash -lc 'pip install -q -e . && <COMMAND>'
```

Installing the mounted TorchTitan tree supplies dependencies such as
`spmd_types` and `tyro` that are needed by the dispatcher test and training
path. Replace `<COMMAND>` with one of the commands below.

## Inspect payload sizes

Set `GIN_LOG_PAYLOAD_SIZE=1` to report the payload for every GIN dispatch and
combine operation:

```bash
-e GIN_LOG_PAYLOAD_SIZE=1
```

For example, add that option to the common `docker run` command and run the
dispatcher test. Each EP rank prints a line similar to:

```text
[GIN payload] operation=dispatch ep_rank=0 world_size=2 dtype=torch.float32 \
row_bytes=12 actual_send_rows=4 actual_receive_rows=3 \
actual_send_bytes=48 actual_receive_bytes=36 \
capacity_rows_per_peer=4 fixed_bytes_per_peer=48 \
fixed_exchange_bytes=96 padding_bytes=48 \
lsa_peers_including_self=2 lsa_bytes=96 \
remote_gin_peers=0 remote_gin_bytes=0
```

The fields distinguish the logical MoE payload from the fixed-capacity data
processed by the kernel:

- `actual_send_bytes`: valid routed token data originating on this rank.
- `actual_receive_bytes`: valid token data expected by this rank.
- `fixed_bytes_per_peer`: bytes reserved and processed for each destination.
- `fixed_exchange_bytes`: total fixed-capacity bytes processed by this rank,
  including its self slot.
- `padding_bytes`: fixed-capacity send bytes that do not contain valid tokens.
- `lsa_bytes`: bytes handled through local LSA windows, including the self
  slot.
- `remote_gin_bytes`: bytes issued through `gin.put()` to remote-node peers.

The calculation is:

```text
row_bytes            = hidden_dim * dtype_size
actual_send_bytes    = sum(input_splits) * row_bytes
fixed_bytes_per_peer = capacity_per_peer * row_bytes
fixed_exchange_bytes = EP_world_size * fixed_bytes_per_peer
remote_gin_bytes     = remote_peer_count * fixed_bytes_per_peer
```

Payload logging is disabled by default because printing on every MoE operation
would distort performance measurements. Enable it for a short correctness run,
record representative sizes, then disable it before benchmarking.

## Static balanced benchmark contract

Set `GIN_PROFILE_STATIC_BALANCED=1` only with
`config.debug.moe_force_load_balance = True`, a fixed full-size microbatch, and
`GIN_PROFILE_CAPACITY_FACTOR=1.0`. This benchmark-only mode synthesizes the
per-expert receive counts and equal peer splits instead of running the RCCL
count all-to-all.

The dispatcher requires token assignments per rank to be divisible by both the
number of experts and the EP size. It also launches an asynchronous device
assert that every local expert received the expected round-robin count. Leave
the option disabled for learned, skewed, partial-batch, or token-dropping
routing; those cases retain the normal dynamic count exchange and capacity
checks.

`GIN_PROFILE_BARRIER_ELISION=1` keeps the first world barrier to calibrate the
GIN signal baseline, then uses a monotonically increasing signal epoch for
later exchanges. Early ranks can therefore put data before late ranks launch
the matching kernel without racing the late rank's signal read. This option is
experimental, requires static balanced routing, and assumes every EP rank
launches the same dispatch/combine sequence on one ordered stream.

## Run the tests

### CPU packing tests

These tests do not create an RCCL communicator:

```bash
PYTHONPATH=. pytest -q tests/unit_tests/cpu/test_gin_ep.py
```

They validate variable split packing, gradient ordering, and capacity overflow.

### Two-GPU communicator test

```bash
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  torchtitan/experiments/gin_ep/tests/test_communicator.py
```

This validates:

- RCCL header/runtime version agreement;
- GIN and RCCL device API availability;
- symmetric allocation and window lifecycle;
- device communicator creation and provider selection;
- fixed all-to-all against an exact expected tensor;
- variable all-to-all against `dist.all_to_all_single`;
- backward gradients;
- idempotent cleanup and closed-state checks.

The test is silent on success and returns exit code zero.

### Two-GPU dispatcher parity test

```bash
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  torchtitan/experiments/gin_ep/tests/test_dispatcher.py
```

This compares GIN dispatch output and expert counts with the standard
TorchTitan dispatcher, then verifies combine output and gradients.

### Fixed all-to-all benchmark

For a 4 MiB payload per peer:

```bash
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  torchtitan/experiments/gin_ep/tests/benchmark_fixed_all_to_all.py \
  --bytes-per-peer 4194304 \
  --warmup 10 \
  --iterations 50
```

The benchmark checks GIN output against RCCL before printing the worst-rank
latencies and speedup:

```text
bytes_per_peer=4194304 world_size=2 gin_ms=... rccl_ms=... speedup=...x
```

## Run the two-node transport benchmark

The top-level scripts exercise RCCL's GIN GDA+LSA all-to-all directly, before
the TorchTitan packing and dispatcher layers.

The default run expects:

- nodes `ctr-cx65-mi300x-12` and `ctr-cx65-mi300x-21`;
- eight GPUs per node;
- image `rccl-gin-stage2-clean:gfx942` on both nodes;
- socket interface `eth1`.

Override those defaults as needed:

```bash
cd "$ROOT"

N0=<first-host> \
N1=<second-host> \
SLOTS=8 \
IMAGE=rccl-gin-stage2-clean:gfx942 \
SOCKET_IFNAME=eth1 \
MIN_BYTES=128 \
MAX_BYTES=128M \
LSA_CTAS=64 \
./scripts/run_gin_gda_lsa.sh
```

Run the matching baseline:

```bash
N0=<first-host> \
N1=<second-host> \
SLOTS=8 \
IMAGE=rccl-gin-stage2-clean:gfx942 \
SOCKET_IFNAME=eth1 \
MIN_BYTES=128 \
MAX_BYTES=128M \
./scripts/run_rccl_default.sh
```

Each script records a timestamped log under `results/` and fails unless the log
contains both:

```text
Out of bounds values : 0 OK
Collective test concluded: alltoall_perf
```

Repeated sweeps can be summarized with:

```bash
python scripts/analyze_final_sweeps.py results/final-5repeat
```

## Enable GIN in a TorchTitan model

Select GIN when constructing a model's routed-expert configuration:

```python
model_spec = model_registry(
    "debugmodel_moe",
    moe_comm_backend="gin",
)
```

The runtime configuration must also use expert parallelism:

```text
parallelism.expert_parallel_degree > 1
```

TorchTitan derives the default maximum tokens per rank from:

```text
training.num_tokens_per_microbatch_per_dp_rank
------------------------------------------------
context_parallel_degree * tensor_parallel_degree
```

The GIN dispatcher reserves `num_max_tokens_per_rank * top_k` token slots per
peer. A manually configured capacity must be at least the derived value.

Run training inside the integration image using the normal TorchTitan
`torchtitan_train --module ... --config ...` workflow. Ensure the selected
model registry entry uses `moe_comm_backend="gin"` and that the dispatcher is
not captured by `torch.compile`; this experiment currently rejects compiled or
non-strict-traced execution.

## Runtime behavior and cleanup

Communicators are cached by:

```text
(EP ranks, device, capacity per peer, hidden dimension, CTA count)
```

The first payload lazily allocates symmetric send/receive buffers and creates
the device communicator. Buffers are reused for later exchanges with the same
configuration.

`Trainer.close()` closes all cached communicators. Explicit communicator
cleanup performs, in order:

1. wait for the last completion event;
2. deregister send and receive windows;
3. destroy the device communicator;
4. destroy the host RCCL communicator;
5. free symmetric allocations.

## Current limitations

- The container build is fixed to `gfx942`.
- The dispatcher supports eager execution only.
- Variable token counts are padded to a fixed per-peer capacity.
- Input and output are staged through owned symmetric buffers.
- Local ranks are expected to form the LSA team used by the hybrid kernel.
- The two-rank tests validate the extension and dispatcher, while the retained
  two-node sweeps validate the lower-level RCCL GIN transport. A complete
  multi-node TorchTitan training run should be used as the final end-to-end
  validation.
