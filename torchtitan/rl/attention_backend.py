# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from torchtitan.models.common.attention import FlexInnerAttention


def vllm_attention_backend(attention_backend) -> AttentionBackendEnum:
    """Pick the vLLM backend serving the model spec's full-attention layers.

    CUSTOM is torchtitan's torch varlen backend, which asserts on
    ``vllm_flash_attn_version``; vLLM returns None for it on ROCm (it does not
    ship vllm_flash_attn there), so that backend cannot serve a ROCm generator.
    ROCm routes to vLLM's AITER FlashAttention backend instead. See PR #4866.
    """
    if isinstance(attention_backend, FlexInnerAttention.Config):
        return AttentionBackendEnum.FLEX_ATTENTION
    if current_platform.is_rocm():
        return AttentionBackendEnum.ROCM_AITER_FA
    return AttentionBackendEnum.CUSTOM
