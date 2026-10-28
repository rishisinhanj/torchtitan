#!/usr/bin/env bash
# Run a configurable multi-step TitanRL ROCm Flex BI validation with W&B.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-titanrl:rocm100-flex-bi}"
MODEL_FLAVOR="${MODEL_FLAVOR:-0.6B}"
NUM_STEPS="${NUM_STEPS:-25}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.8}"
ASSETS_DIR="${ASSETS_DIR:-${REPO_ROOT}/assets/hf/Qwen3-${MODEL_FLAVOR}}"
VLLM_WHEELS_DIR="${VLLM_WHEELS_DIR:-${HOME}/vllm-wheels}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/outputs/rocm_flex_bi_${MODEL_FLAVOR}_${NUM_STEPS}step}"
WANDB_PROJECT="${WANDB_PROJECT:-rocm_batch_invariance}"
WANDB_NAME="${WANDB_NAME:-rocm-flex-bi-qwen3-${MODEL_FLAVOR}-${NUM_STEPS}step}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY must be set" >&2
  exit 2
fi
if [[ ! -d "${ASSETS_DIR}" ]]; then
  echo "Model assets not found: ${ASSETS_DIR}" >&2
  exit 2
fi

mkdir -p "${ARTIFACT_DIR}"

docker run --rm \
  --name "titanrl-rocm-flex-bi-${MODEL_FLAVOR,,}" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --network=host \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  -e VLLM_BATCH_INVARIANT=1 \
  -e HF_ASSETS_PATH=/assets \
  -e PYTHONPATH=/work \
  -e "ROCM_BI_MODEL_FLAVOR=${MODEL_FLAVOR}" \
  -e "ROCM_BI_NUM_STEPS=${NUM_STEPS}" \
  -e "ROCM_BI_NUM_SAMPLES=${NUM_SAMPLES}" \
  -e "ROCM_BI_TEMPERATURE=${TEMPERATURE}" \
  -e WANDB_API_KEY \
  -e "WANDB_PROJECT=${WANDB_PROJECT}" \
  -e "WANDB_NAME=${WANDB_NAME}" \
  -v "${REPO_ROOT}:/src:ro" \
  -v "${ASSETS_DIR}:/assets:ro" \
  -v "${VLLM_WHEELS_DIR}:/wheels:ro" \
  -v "${ARTIFACT_DIR}:/artifacts" \
  -w /work \
  --entrypoint bash \
  "${IMAGE}" -lc '
set -euo pipefail
tar -C /src \
  --exclude=.git \
  --exclude=assets \
  --exclude=outputs \
  -cf - . | tar -C /work -xf -
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
            raise RuntimeError(
                f"{path}: expected one occurrence, found {text.count(old)}"
            )
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
python3 -m torchtitan.rl.train \
  --module scripts.rocm_flex_bi_long_run \
  --config rocm_flex_bi_long_run \
  --output-dir /artifacts/run 2>&1 | tee /artifacts/long_run.log
'
