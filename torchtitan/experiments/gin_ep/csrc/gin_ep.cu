#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <nccl.h> // RCCL host APIs
#define NCCL_HOSTLIB_ONLY
#include <nccl_device.h>
#include <torch/extension.h> // pybind integration
#include <cstring>
#include <memory>
#include <string>
#include <tuple>

namespace py = pybind11;

extern "C" hipError_t gin_ep_launch_hybrid_all_to_all(
    ncclWindow_t send_window,
    ncclWindow_t recv_window,
    size_t bytes_per_peer,
    const ncclDevComm* device_comm,
    int cta_count,
    uint64_t signal_sequence,
    uint64_t* signal_base,
    hipStream_t stream);

// If RCCL fails, throw a Python-visible exception.
static void check_nccl(ncclResult_t result, const char* operation) {
  TORCH_CHECK(
      result == ncclSuccess,
      operation,
      " failed: ",
      ncclGetErrorString(result));
}

// same as check_nccl, but for HIP
static void check_hip(hipError_t result, const char* operation) {
  TORCH_CHECK(
      result == hipSuccess,
      operation,
      " failed: ",
      hipGetErrorString(result));
}

// reads current device and sets it
class HipDeviceGuard final {
 public:
  explicit HipDeviceGuard(int device) {
    check_hip(hipGetDevice(&previous_device_), "hipGetDevice");
    check_hip(hipSetDevice(device), "hipSetDevice");
  }

  ~HipDeviceGuard() noexcept {
    (void)hipSetDevice(previous_device_);
  }

  HipDeviceGuard(const HipDeviceGuard&) = delete;
  HipDeviceGuard& operator=(const HipDeviceGuard&) = delete;

 private:
  int previous_device_ = -1;
};

static_assert(
    sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES,
    "Unexpected RCCL unique ID size");



    // copies all bytes into the ncclUniqueId struct
static ncclUniqueId decode_unique_id(const py::bytes& unique_id) {
  const std::string bytes = unique_id;

  TORCH_CHECK(
      bytes.size() == NCCL_UNIQUE_ID_BYTES,
      "unique_id must contain exactly ",
      NCCL_UNIQUE_ID_BYTES,
      " bytes, got ",
      bytes.size());

  ncclUniqueId id{};
  std::memcpy(id.internal, bytes.data(), NCCL_UNIQUE_ID_BYTES);
  return id;
}

// Reject invalid rank, world size, or device before collective initialization.
static void validate_communicator_args(
    int rank,
    int world_size,
    int device) {
  TORCH_CHECK(world_size > 0, "world_size must be positive");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "rank must be in [0, world_size), got ",
      rank);

  int device_count = 0;
  check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");

  TORCH_CHECK(
      device >= 0 && device < device_count,
      "device must be in [0, ",
      device_count,
      "), got ",
      device);
}

class GinCommunicator final {
 public:
  ~GinCommunicator() noexcept {
    ncclComm_t comm = comm_;
    comm_ = nullptr;
    if (comm == nullptr) {
      return;
    }

    int previous_device = -1;
    const bool can_restore =
        hipGetDevice(&previous_device) == hipSuccess;

    if (hipSetDevice(device_) == hipSuccess) {
      if (completion_event_ != nullptr) {
        (void)hipEventSynchronize(completion_event_);
        (void)hipEventDestroy(completion_event_);
      }
      (void)ncclCommAbort(comm);
      if (send_buffer_ != nullptr) {
        (void)ncclMemFree(send_buffer_);
      }
      if (recv_buffer_ != nullptr) {
        (void)ncclMemFree(recv_buffer_);
      }
      if (signal_base_ != nullptr) {
        (void)hipFree(signal_base_);
      }
    }

    if (can_restore) {
      (void)hipSetDevice(previous_device);
    }
  }

  GinCommunicator(const GinCommunicator&) = delete;
  GinCommunicator& operator=(const GinCommunicator&) = delete;
  GinCommunicator(GinCommunicator&&) = delete;
  GinCommunicator& operator=(GinCommunicator&&) = delete;

  void close() {
    if (comm_ == nullptr) {
      return;
    }

    HipDeviceGuard device_guard(device_);

    ncclComm_t comm = comm_;
    comm_ = nullptr;

    ncclResult_t result = ncclSuccess;
    const char* failed_operation = nullptr;
    auto record_failure = [&result, &failed_operation](
                              ncclResult_t candidate,
                              const char* operation) {
      if (result == ncclSuccess && candidate != ncclSuccess) {
        result = candidate;
        failed_operation = operation;
      }
    };

    {
      py::gil_scoped_release release;

      if (completion_event_ != nullptr) {
        const hipError_t event_result =
            hipEventSynchronize(completion_event_);
        if (event_result == hipSuccess) {
          (void)hipEventDestroy(completion_event_);
        } else if (result == ncclSuccess) {
          result = ncclSystemError;
          failed_operation = "hipEventSynchronize";
        }
        completion_event_ = nullptr;
      }

      if (symmetric_buffers_allocated_) {
        record_failure(
            ncclCommWindowDeregister(comm, send_window_),
            "ncclCommWindowDeregister(send)");
        record_failure(
            ncclCommWindowDeregister(comm, recv_window_),
            "ncclCommWindowDeregister(recv)");
      }

      if (result == ncclSuccess && device_comm_created_) {
        record_failure(
            ncclDevCommDestroy(comm, &device_comm_),
            "ncclDevCommDestroy");
      }
      if (signal_base_ != nullptr) {
        (void)hipFree(signal_base_);
      }

      if (result == ncclSuccess) {
        record_failure(ncclCommDestroy(comm), "ncclCommDestroy");
      } else {
        (void)ncclCommAbort(comm);
      }

      if (send_buffer_ != nullptr) {
        record_failure(ncclMemFree(send_buffer_), "ncclMemFree(send)");
      }
      if (recv_buffer_ != nullptr) {
        record_failure(ncclMemFree(recv_buffer_), "ncclMemFree(recv)");
      }
    }

    send_buffer_ = nullptr;
    recv_buffer_ = nullptr;
    send_window_ = {};
    recv_window_ = {};
    symmetric_buffer_bytes_ = 0;
    symmetric_buffers_allocated_ = false;
    device_comm_ = {};
    device_comm_created_ = false;
    device_cta_count_ = 0;
    signal_base_ = nullptr;
    launch_sequence_ = 0;

    check_nccl(
        result,
        failed_operation == nullptr ? "communicator cleanup" : failed_operation);
  }

  bool closed() const noexcept {
    return comm_ == nullptr;
  }

  int rank() const noexcept {
    return rank_;
  }

  int world_size() const noexcept {
    return world_size_;
  }

  int device() const noexcept {
    return device_;
  }

  py::dict query_properties() {
    TORCH_CHECK(
        comm_ != nullptr,
        "cannot query properties of a closed communicator");

    HipDeviceGuard device_guard(device_);
    ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    ncclResult_t result;

    {
      py::gil_scoped_release release;
      result = ncclCommQueryProperties(comm_, &properties);
    }

    check_nccl(result, "ncclCommQueryProperties");

    py::dict output;
    output["rank"] = properties.rank;
    output["world_size"] = properties.nRanks;
    output["device"] = properties.cudaDev;
    output["nvml_device"] = properties.nvmlDev;
    output["device_api_support"] = properties.deviceApiSupport;
    output["multimem_support"] = properties.multimemSupport;
    output["gin_type"] = static_cast<int>(properties.ginType);
    output["num_lsa_teams"] = properties.nLsaTeams;
    output["host_rma_support"] = properties.hostRmaSupport;
    output["railed_gin_type"] =
        static_cast<int>(properties.railedGinType);
    return output;
  }

  void create_device_communicator(int cta_count) {
    TORCH_CHECK(
        comm_ != nullptr,
        "cannot create a device communicator from a closed communicator");
    TORCH_CHECK(
        !device_comm_created_,
        "device communicator has already been created");
    TORCH_CHECK(
        cta_count >= 2,
        "cta_count must be at least 2 for the hybrid GDA+LSA path");

    HipDeviceGuard device_guard(device_);

    ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    ncclResult_t result;
    {
      py::gil_scoped_release release;
      result = ncclCommQueryProperties(comm_, &properties);
    }
    check_nccl(result, "ncclCommQueryProperties");

    TORCH_CHECK(
        properties.deviceApiSupport,
        "RCCL device API is not supported by this communicator");
    TORCH_CHECK(
        properties.ginType != NCCL_GIN_TYPE_NONE,
        "GIN is not enabled for this communicator");

    ncclDevCommRequirements_t requirements =
        NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    requirements.barrierCount = 1;
    requirements.lsaBarrierCount = cta_count - 1;
    requirements.ginSignalCount = 1;
    requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;

    ncclDevComm_t device_comm{};
    {
      py::gil_scoped_release release;
      result = ncclDevCommCreate(
          comm_,
          &requirements,
          &device_comm);
    }
    check_nccl(result, "ncclDevCommCreate");

    if (
        device_comm.ginConnectionCount == 0 ||
        device_comm.ginNetDeviceTypes[0] !=
            NCCL_NET_DEVICE_GIN_ROCSHMEM_GDA) {
      (void)ncclDevCommDestroy(comm_, &device_comm);
      TORCH_CHECK(
          false,
          "device communicator did not select the rocSHMEM GDA backend");
    }

    uint64_t* signal_base = nullptr;
    const hipError_t allocation_result = hipMalloc(
        reinterpret_cast<void**>(&signal_base),
        sizeof(*signal_base));
    if (allocation_result != hipSuccess) {
      (void)ncclDevCommDestroy(comm_, &device_comm);
      check_hip(allocation_result, "hipMalloc(signal_base)");
    }

    device_comm_ = device_comm;
    signal_base_ = signal_base;
    device_comm_created_ = true;
    device_cta_count_ = cta_count;
    launch_sequence_ = 0;
  }

  void destroy_device_communicator() {
    TORCH_CHECK(
        comm_ != nullptr,
        "cannot destroy a device communicator from a closed communicator");
    if (!device_comm_created_) {
      return;
    }

    HipDeviceGuard device_guard(device_);
    if (completion_event_ != nullptr) {
      check_hip(
          hipEventSynchronize(completion_event_),
          "hipEventSynchronize");
    }
    ncclResult_t result;
    {
      py::gil_scoped_release release;
      result = ncclDevCommDestroy(comm_, &device_comm_);
    }
    check_nccl(result, "ncclDevCommDestroy");
    check_hip(hipFree(signal_base_), "hipFree(signal_base)");

    device_comm_ = {};
    device_comm_created_ = false;
    device_cta_count_ = 0;
    signal_base_ = nullptr;
    launch_sequence_ = 0;
  }

  void set_steady_state_barrier_elision(bool enabled) {
    TORCH_CHECK(
        launch_sequence_ == 0,
        "barrier elision must be configured before the first all-to-all");
    steady_state_barrier_elision_ = enabled;
  }

  bool device_communicator_created() const noexcept {
    return device_comm_created_;
  }

  int device_cta_count() const noexcept {
    return device_cta_count_;
  }

  py::dict device_communicator_info() const {
    TORCH_CHECK(
        device_comm_created_,
        "device communicator has not been created");

    py::dict output;
    output["rank"] = device_comm_.rank;
    output["world_size"] = device_comm_.nRanks;
    output["lsa_rank"] = device_comm_.lsaRank;
    output["lsa_size"] = device_comm_.lsaSize;
    output["gin_connection_count"] =
        static_cast<int>(device_comm_.ginConnectionCount);
    output["gin_type"] =
        device_comm_.ginConnectionCount == 0
        ? 0
        : static_cast<int>(device_comm_.ginNetDeviceTypes[0]);
    output["gin_handle_available"] =
        device_comm_.ginConnectionCount != 0 &&
        device_comm_.ginHandles[0] != nullptr;
    output["gin_signal_count"] = device_comm_.ginSignalCount;
    output["gin_signals_available"] =
        device_comm_.ginSignalShadows != nullptr;
    return output;
  }

  torch::Tensor fixed_all_to_all(const torch::Tensor& input) {
    return fixed_all_to_all_impl(input, torch::empty_like(input));
  }

  torch::Tensor fixed_all_to_all_out(
      const torch::Tensor& input,
      torch::Tensor output) {
    return fixed_all_to_all_impl(input, std::move(output));
  }

  torch::Tensor fixed_all_to_all_impl(
      const torch::Tensor& input,
      torch::Tensor output) {
    TORCH_CHECK(
        comm_ != nullptr,
        "cannot run all-to-all on a closed communicator");
    TORCH_CHECK(
        symmetric_buffers_allocated_,
        "symmetric buffers must be allocated before all-to-all");
    TORCH_CHECK(
        device_comm_created_,
        "device communicator must be created before all-to-all");
    TORCH_CHECK(input.is_cuda(), "input must be a GPU tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(
        input.dim() >= 1 && input.size(0) == world_size_,
        "input first dimension must equal world_size");
    TORCH_CHECK(input.numel() > 0, "input must not be empty");
    TORCH_CHECK(
        input.get_device() == device_,
        "input must be on communicator device ",
        device_);
    TORCH_CHECK(output.is_cuda(), "output must be a GPU tensor");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(
        output.device() == input.device(),
        "output must be on the same device as input");
    TORCH_CHECK(
        output.scalar_type() == input.scalar_type(),
        "output must have the same dtype as input");
    TORCH_CHECK(
        output.sizes() == input.sizes(),
        "output must have the same shape as input");

    const size_t total_bytes = input.nbytes();
    TORCH_CHECK(
        total_bytes <= symmetric_buffer_bytes_,
        "input requires ",
        total_bytes,
        " bytes but symmetric buffer capacity is ",
        symmetric_buffer_bytes_);
    TORCH_CHECK(
        total_bytes % world_size_ == 0,
        "input byte size must be divisible by world_size");
    const size_t bytes_per_peer = total_bytes / world_size_;

    HipDeviceGuard device_guard(device_);
    hipStream_t stream =
        at::cuda::getCurrentCUDAStream(device_).stream();

    if (completion_event_ == nullptr) {
      check_hip(
          hipEventCreateWithFlags(
              &completion_event_,
              hipEventDisableTiming),
          "hipEventCreateWithFlags");
    } else {
      check_hip(
          hipStreamWaitEvent(stream, completion_event_, 0),
          "hipStreamWaitEvent");
    }

    check_hip(
        hipMemcpyAsync(
            send_buffer_,
            input.data_ptr(),
            total_bytes,
            hipMemcpyDeviceToDevice,
            stream),
        "hipMemcpyAsync(input)");

    check_hip(
        gin_ep_launch_hybrid_all_to_all(
            send_window_,
            recv_window_,
            bytes_per_peer,
            &device_comm_,
            device_cta_count_,
            steady_state_barrier_elision_ ? launch_sequence_++ : 0,
            signal_base_,
            stream),
        "HybridAlltoAllKernel launch");

    check_hip(
        hipMemcpyAsync(
            output.data_ptr(),
            recv_buffer_,
            total_bytes,
            hipMemcpyDeviceToDevice,
            stream),
        "hipMemcpyAsync(output)");
    check_hip(
        hipEventRecord(completion_event_, stream),
        "hipEventRecord");
    return output;
  }

  void allocate_symmetric_buffers(size_t bytes) {
    TORCH_CHECK(
        comm_ != nullptr,
        "cannot allocate buffers for a closed communicator");
    TORCH_CHECK(
        !symmetric_buffers_allocated_,
        "symmetric buffers have already been allocated");
    TORCH_CHECK(bytes > 0, "buffer size must be positive");

    HipDeviceGuard device_guard(device_);

    void* send_buffer = nullptr;
    void* recv_buffer = nullptr;
    ncclWindow_t send_window{};
    ncclWindow_t recv_window{};
    ncclResult_t result;

    {
      py::gil_scoped_release release;
      result = ncclMemAlloc(&send_buffer, bytes);
      if (result == ncclSuccess) {
        result = ncclMemAlloc(&recv_buffer, bytes);
      }
      if (result == ncclSuccess) {
        result = ncclCommWindowRegister(
            comm_,
            send_buffer,
            bytes,
            &send_window,
            NCCL_WIN_COLL_SYMMETRIC);
      }
      if (result == ncclSuccess) {
        result = ncclCommWindowRegister(
            comm_,
            recv_buffer,
            bytes,
            &recv_window,
            NCCL_WIN_COLL_SYMMETRIC);
      }

      if (result != ncclSuccess) {
        if (send_window != nullptr) {
          (void)ncclCommWindowDeregister(comm_, send_window);
        }
        if (recv_window != nullptr) {
          (void)ncclCommWindowDeregister(comm_, recv_window);
        }
        if (send_buffer != nullptr) {
          (void)ncclMemFree(send_buffer);
        }
        if (recv_buffer != nullptr) {
          (void)ncclMemFree(recv_buffer);
        }
      }
    }

    check_nccl(result, "allocate and register symmetric buffers");

    send_buffer_ = send_buffer;
    recv_buffer_ = recv_buffer;
    send_window_ = send_window;
    recv_window_ = recv_window;
    symmetric_buffer_bytes_ = bytes;
    symmetric_buffers_allocated_ = true;
  }

  void release_symmetric_buffers() {
    TORCH_CHECK(
        comm_ != nullptr,
        "cannot release buffers from a closed communicator");
    if (!symmetric_buffers_allocated_) {
      return;
    }

    HipDeviceGuard device_guard(device_);
    if (completion_event_ != nullptr) {
      check_hip(
          hipEventSynchronize(completion_event_),
          "hipEventSynchronize");
      check_hip(
          hipEventDestroy(completion_event_),
          "hipEventDestroy");
      completion_event_ = nullptr;
    }
    ncclResult_t send_result;
    ncclResult_t recv_result;
    ncclResult_t send_free_result = ncclSuccess;
    ncclResult_t recv_free_result = ncclSuccess;

    {
      py::gil_scoped_release release;
      send_result = ncclCommWindowDeregister(comm_, send_window_);
      recv_result = ncclCommWindowDeregister(comm_, recv_window_);
      if (send_result == ncclSuccess) {
        send_free_result = ncclMemFree(send_buffer_);
      }
      if (recv_result == ncclSuccess) {
        recv_free_result = ncclMemFree(recv_buffer_);
      }
    }

    if (send_result == ncclSuccess && send_free_result == ncclSuccess) {
      send_buffer_ = nullptr;
      send_window_ = {};
    }
    if (recv_result == ncclSuccess && recv_free_result == ncclSuccess) {
      recv_buffer_ = nullptr;
      recv_window_ = {};
    }
    symmetric_buffers_allocated_ =
        send_buffer_ != nullptr || recv_buffer_ != nullptr;
    if (!symmetric_buffers_allocated_) {
      symmetric_buffer_bytes_ = 0;
    }

    check_nccl(send_result, "ncclCommWindowDeregister(send)");
    check_nccl(recv_result, "ncclCommWindowDeregister(recv)");
    check_nccl(send_free_result, "ncclMemFree(send)");
    check_nccl(recv_free_result, "ncclMemFree(recv)");
  }

  bool symmetric_buffers_allocated() const noexcept {
    return symmetric_buffers_allocated_;
  }

  size_t symmetric_buffer_bytes() const noexcept {
    return symmetric_buffer_bytes_;
  }

  static std::unique_ptr<GinCommunicator> create(
      const py::bytes& unique_id,
      int rank,
      int world_size,
      int device) {
    validate_communicator_args(rank, world_size, device);
    const ncclUniqueId id = decode_unique_id(unique_id);

    auto owner = std::unique_ptr<GinCommunicator>(
        new GinCommunicator());

    HipDeviceGuard device_guard(device);

    ncclComm_t comm = nullptr;
    ncclResult_t result;

    {
      py::gil_scoped_release release;
      result = ncclCommInitRank(&comm, world_size, id, rank);

      if (result != ncclSuccess && comm != nullptr) {
        (void)ncclCommAbort(comm);
        comm = nullptr;
      }
    }

    check_nccl(result, "ncclCommInitRank");

    auto abort_unowned = [&comm]() noexcept {
      if (comm != nullptr) {
        {
          py::gil_scoped_release release;
          (void)ncclCommAbort(comm);
        }
        comm = nullptr;
      }
    };

    int actual_world_size = -1;
    int actual_rank = -1;
    int actual_device = -1;

    result = ncclCommCount(comm, &actual_world_size);
    if (result != ncclSuccess) {
      abort_unowned();
      check_nccl(result, "ncclCommCount");
    }

    result = ncclCommUserRank(comm, &actual_rank);
    if (result != ncclSuccess) {
      abort_unowned();
      check_nccl(result, "ncclCommUserRank");
    }

    result = ncclCommCuDevice(comm, &actual_device);
    if (result != ncclSuccess) {
      abort_unowned();
      check_nccl(result, "ncclCommCuDevice");
    }

    if (
        actual_world_size != world_size ||
        actual_rank != rank ||
        actual_device != device) {
      abort_unowned();

      TORCH_CHECK(
          false,
          "RCCL communicator identity mismatch: expected rank=",
          rank,
          ", world_size=",
          world_size,
          ", device=",
          device,
          "; got rank=",
          actual_rank,
          ", world_size=",
          actual_world_size,
          ", device=",
          actual_device);
    }

    owner->comm_ = comm;
    comm = nullptr;
    owner->rank_ = actual_rank;
    owner->world_size_ = actual_world_size;
    owner->device_ = actual_device;
    return owner;
  }

 private:
  GinCommunicator() = default;

  ncclComm_t comm_ = nullptr;
  ncclDevComm_t device_comm_{};
  bool device_comm_created_ = false;
  int device_cta_count_ = 0;
  void* send_buffer_ = nullptr;
  void* recv_buffer_ = nullptr;
  ncclWindow_t send_window_{};
  ncclWindow_t recv_window_{};
  uint64_t* signal_base_ = nullptr;
  uint64_t launch_sequence_ = 0;
  bool steady_state_barrier_elision_ = false;
  size_t symmetric_buffer_bytes_ = 0;
  bool symmetric_buffers_allocated_ = false;
  hipEvent_t completion_event_ = nullptr;
  int rank_ = -1;
  int world_size_ = 0;
  int device_ = -1;
};

py::bytes get_unique_id() {
  ncclUniqueId id{};
  check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
  return py::bytes(id.internal, NCCL_UNIQUE_ID_BYTES);
}

std::tuple<int, int> version_info() {
  int runtime_version = 0;
  check_nccl(ncclGetVersion(&runtime_version), "ncclGetVersion");
  return {NCCL_VERSION_CODE, runtime_version};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<GinCommunicator, std::unique_ptr<GinCommunicator>>(
      module, "GinCommunicator", py::dynamic_attr())
      .def_static(
          "create",
          &GinCommunicator::create,
          py::arg("unique_id"),
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("device"))
      .def("close", &GinCommunicator::close)
      .def("query_properties", &GinCommunicator::query_properties)
      .def(
          "create_device_communicator",
          &GinCommunicator::create_device_communicator,
          py::arg("cta_count") = 64)
      .def(
          "destroy_device_communicator",
          &GinCommunicator::destroy_device_communicator)
      .def(
          "set_steady_state_barrier_elision",
          &GinCommunicator::set_steady_state_barrier_elision,
          py::arg("enabled"))
      .def(
          "allocate_symmetric_buffers",
          &GinCommunicator::allocate_symmetric_buffers,
          py::arg("bytes"))
      .def(
          "release_symmetric_buffers",
          &GinCommunicator::release_symmetric_buffers)
      .def(
          "fixed_all_to_all",
          &GinCommunicator::fixed_all_to_all,
          py::arg("input"))
      .def(
          "fixed_all_to_all_out",
          &GinCommunicator::fixed_all_to_all_out,
          py::arg("input"),
          py::arg("output"))
      .def(
          "device_communicator_info",
          &GinCommunicator::device_communicator_info)
      .def_property_readonly("closed", &GinCommunicator::closed)
      .def_property_readonly(
          "device_communicator_created",
          &GinCommunicator::device_communicator_created)
      .def_property_readonly(
          "device_cta_count",
          &GinCommunicator::device_cta_count)
      .def_property_readonly(
          "symmetric_buffers_allocated",
          &GinCommunicator::symmetric_buffers_allocated)
      .def_property_readonly(
          "symmetric_buffer_bytes",
          &GinCommunicator::symmetric_buffer_bytes)
      .def_property_readonly("rank", &GinCommunicator::rank)
      .def_property_readonly("world_size", &GinCommunicator::world_size)
      .def_property_readonly("device", &GinCommunicator::device);

  module.def(
      "version_info",
      &version_info,
      "Return RCCL header and runtime-library versions");

  module.def(
      "get_unique_id",
      &get_unique_id,
      "Return a unique identifier for the communicator");
}

