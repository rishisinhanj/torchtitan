#!/usr/bin/env bash
# Run and preserve the ROCm trainer-vLLM FlexAttention localization artifacts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-titanrl:rocm100-flex-bi}"
ASSETS_DIR="${ASSETS_DIR:-${REPO_ROOT}/assets/hf/Qwen3-0.6B}"
VLLM_WHEELS_DIR="${VLLM_WHEELS_DIR:-${HOME}/vllm-wheels}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/outputs/rocm_flex_parity}"
GPU_IDS="${GPU_IDS:-0,1}"
PYTEST_TARGET="${PYTEST_TARGET:-tests/rl/unit_tests/gpu/test_bitwise_parity.py::TestBitwiseParityFlex::test_first_trainer_vllm_divergence}"

mkdir -p "${ARTIFACT_DIR}"

set +e
docker run --rm \
  --name titanrl-rocm-flex-parity-diagnostic \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --network=host \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  -e "HIP_VISIBLE_DEVICES=${GPU_IDS}" \
  -e VLLM_BATCH_INVARIANT=1 \
  -e RL_FLEX_PARITY_DIAGNOSTICS=1 \
  -e RL_FLEX_PARITY_DUMP_DIR=/artifacts \
  -e "PYTEST_TARGET=${PYTEST_TARGET}" \
  -e HF_ASSETS_PATH=/assets \
  -e PYTHONPATH=/work \
  -v "${REPO_ROOT}:/src:ro" \
  -v "${ASSETS_DIR}:/assets:ro" \
  -v "${VLLM_WHEELS_DIR}:/wheels:ro" \
  -v "${ARTIFACT_DIR}:/artifacts" \
  -w /work \
  --entrypoint bash \
  "${IMAGE}" -lc '
set -euo pipefail
cp -a /src/. /work
python3 -m pip install --force-reinstall --no-deps /wheels/*.whl >/dev/null
python3 -m pip install --no-cache-dir --pre --no-deps --upgrade torchvision \
  --index-url https://download.pytorch.org/whl/nightly/rocm10.0 >/dev/null
python3 - <<"PY"
from pathlib import Path


def replace_if_present(path, old, new):
    path = Path(path)
    text = path.read_text()
    if old in text:
        if text.count(old) != 1:
            raise RuntimeError(f"{path}: expected one occurrence, found {text.count(old)}")
        path.write_text(text.replace(old, new, 1))


replace_if_present(
    "/usr/local/lib/python3.12/dist-packages/vllm/env_override.py",
    "\n_maybe_promote_torch_symbols_for_rocm()\n",
    "\n# Disabled for ROCm nightly Triton compatibility.\n",
)
replace_if_present(
    "/work/torchtitan/rl/generator/vllm/registry.py",
    "from vllm.transformers_utils.config import PretrainedConfig, register_config_parser",
    "from transformers import PretrainedConfig\n"
    "    from vllm.transformers_utils.config import register_config_parser",
)
replace_if_present(
    "/work/torchtitan/rl/generator/generator.py",
    "                pass_config=PassConfig(\n"
    "                    enable_sp=enable_sequence_parallel,\n"
    "                    sp_min_token_num=1 if enable_sequence_parallel else None,\n"
    "                ),",
    "                pass_config=PassConfig(),",
)
replace_if_present(
    "/work/torchtitan/rl/generator/generator.py",
    "            pass_config=PassConfig(\n"
    "                enable_sp=enable_sequence_parallel,\n"
    "                sp_min_token_num=1 if enable_sequence_parallel else None,\n"
    "            ),",
    "            pass_config=PassConfig(),",
)
PY
torchrun --standalone --nproc_per_node=2 -m pytest \
  "${PYTEST_TARGET}" \
  -v -s 2>&1 | tee /artifacts/diagnostic.log
'
test_status=$?
set -e

shopt -s nullglob
captures=("${ARTIFACT_DIR}"/flex_parity_rank*.pt)
for capture in "${captures[@]}"; do
  capture_name="$(basename "${capture}")"
  docker run --rm \
    -v "${REPO_ROOT}:/workspace/torchtitan:ro" \
    -v "${ARTIFACT_DIR}:/artifacts:ro" \
    -w /workspace/torchtitan \
    --entrypoint python3 \
    "${IMAGE}" \
    scripts/analyze_rocm_flex_parity_capture.py \
    "/artifacts/${capture_name}" | tee "${capture%.pt}_analysis.txt"
done

exit "${test_status}"
