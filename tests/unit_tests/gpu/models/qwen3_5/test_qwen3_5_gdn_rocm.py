# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated DeltaNet's ROCm trainer kernels against the float32 gated delta rule.

The trainer runs flash-linear-attention's chunk kernel and a torch convolution.
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

_ON_ROCM = torch.cuda.is_available() and torch.version.hip is not None

# Shape suffixes: T tokens, C channels, H heads, K key head dim, V value head dim,
# W convolution width.
_NUM_K_HEADS, _NUM_V_HEADS, _HEAD_K_DIM, _HEAD_V_DIM, _WIDTH = 2, 4, 128, 128, 4


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _gated_delta(
    q_THK: torch.Tensor,
    k_THK: torch.Tensor,
    v_THV: torch.Tensor,
    g_TH: torch.Tensor,
    beta_TH: torch.Tensor,
    initial_state_HVK: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential float32 recurrence over one sequence; value head h reads key head h // r."""
    num_v_heads, head_v_dim = v_THV.shape[1:]
    head_k_dim = q_THK.shape[-1]
    ratio = num_v_heads // q_THK.shape[1]
    q_THK = _l2norm(q_THK.float()).repeat_interleave(ratio, 1) * head_k_dim**-0.5
    k_THK = _l2norm(k_THK.float()).repeat_interleave(ratio, 1)
    state_HVK = (
        q_THK.new_zeros(num_v_heads, head_v_dim, head_k_dim)
        if initial_state_HVK is None
        else initial_state_HVK.float()
    )
    outputs = []
    for t in range(q_THK.shape[0]):
        state_HVK = state_HVK * g_TH[t].float().exp()[:, None, None]
        delta_HV = (
            v_THV[t].float() - torch.einsum("hvk,hk->hv", state_HVK, k_THK[t])
        ) * beta_TH[t].float()[:, None]
        state_HVK = state_HVK + torch.einsum("hv,hk->hvk", delta_HV, k_THK[t])
        outputs.append(torch.einsum("hvk,hk->hv", state_HVK, q_THK[t]))
    return torch.stack(outputs), state_HVK


def _causal_conv(x_TC: torch.Tensor, weight_CW: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv + SiLU over one sequence, in float32."""
    width = weight_CW.shape[-1]
    x_1CT = F.pad(x_TC.float().transpose(0, 1).unsqueeze(0), [width - 1, 0])
    out_1CT = F.conv1d(x_1CT, weight_CW.float().unsqueeze(1), groups=x_TC.shape[1])
    return F.silu(out_1CT).squeeze(0).transpose(0, 1)


def _gates(
    a_TH: torch.Tensor,
    b_TH: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_H: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H.float())
    return g_TH, torch.sigmoid(b_TH.float())


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.float()
    return ((actual.float() - expected).norm() / expected.norm()).item()


@unittest.skipUnless(_ON_ROCM, "ROCm GDN kernels need an AMD GPU")
class TestTrainerGDNOnROCm(unittest.TestCase):
    """flash-linear-attention chunk GDN and the torch convolution (trainer)."""

    # A one-token document, one exactly one chunk long, and chunk-straddling ones.
    _CU_SEQLENS = [0, 1, 65, 200, 330]

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.cu_seqlens = torch.tensor(
            self._CU_SEQLENS, dtype=torch.int32, device=self.device
        )

    def test_causal_conv1d_varlen_matches_per_document(self):
        from torchtitan.models.common.attention.gdn import _torch_causal_conv1d_varlen

        num_tokens, channels = self._CU_SEQLENS[-1], 96
        x_TC = torch.randn(
            num_tokens, channels, device=self.device, dtype=torch.bfloat16
        ).requires_grad_()
        weight_C1W = torch.randn(
            channels, 1, _WIDTH, device=self.device, dtype=torch.bfloat16
        ).requires_grad_()
        actual_TC = _torch_causal_conv1d_varlen(x_TC, weight_C1W, self.cu_seqlens)
        expected_TC = torch.cat(
            [
                _causal_conv(x_TC[start:end], weight_C1W[:, 0])
                for start, end in zip(self._CU_SEQLENS[:-1], self._CU_SEQLENS[1:])
            ]
        )
        self.assertEqual(actual_TC.dtype, torch.bfloat16)
        self.assertLess(_relative_error(actual_TC, expected_TC), 1e-2)

        grad_TC = torch.randn_like(expected_TC)
        actual_grads = torch.autograd.grad(
            actual_TC.float(), (x_TC, weight_C1W), grad_TC
        )
        expected_grads = torch.autograd.grad(expected_TC, (x_TC, weight_C1W), grad_TC)
        for actual, expected in zip(actual_grads, expected_grads):
            self.assertLess(_relative_error(actual, expected), 1e-2)

    def test_chunk_gdn_matches_recurrence(self):
        from torchtitan.models.common.attention.gdn import GDNKernel

        kernel = GDNKernel.Config().build()
        num_tokens = self._CU_SEQLENS[-1]
        bf16 = dict(device=self.device, dtype=torch.bfloat16)
        q_THK = torch.randn(num_tokens, _NUM_K_HEADS, _HEAD_K_DIM, **bf16)
        k_THK = torch.randn(num_tokens, _NUM_K_HEADS, _HEAD_K_DIM, **bf16)
        v_THV = torch.randn(num_tokens, _NUM_V_HEADS, _HEAD_V_DIM, **bf16)
        a_TH = torch.randn(num_tokens, _NUM_V_HEADS, **bf16)
        b_TH = torch.randn(num_tokens, _NUM_V_HEADS, **bf16)
        A_log_H = torch.empty(_NUM_V_HEADS, device=self.device).uniform_(0, 1.5)
        dt_bias_H = torch.randn(_NUM_V_HEADS, device=self.device)
        inputs = [
            tensor.requires_grad_() for tensor in (q_THK, k_THK, v_THV, a_TH, b_TH)
        ]
        # The model's gate arithmetic: float32 decay, model-dtype update gate.
        g_TH = -torch.exp(A_log_H) * F.softplus(a_TH.float() + dt_bias_H)
        beta_TH = torch.sigmoid(b_TH)
        actual_THV = kernel(
            q_THK, k_THK, v_THV, g_TH, beta_TH, cu_seqlens=self.cu_seqlens
        )

        expected_THV = torch.cat(
            [
                _gated_delta(
                    q_THK[start:end],
                    k_THK[start:end],
                    v_THV[start:end],
                    *_gates(a_TH[start:end], b_TH[start:end], A_log_H, dt_bias_H),
                )[0]
                for start, end in zip(self._CU_SEQLENS[:-1], self._CU_SEQLENS[1:])
            ]
        )
        self.assertEqual(actual_THV.dtype, torch.bfloat16)
        self.assertLess(_relative_error(actual_THV, expected_THV), 2e-2)

        grad_THV = torch.randn_like(expected_THV)
        actual_grads = torch.autograd.grad(actual_THV.float(), inputs, grad_THV)
        expected_grads = torch.autograd.grad(expected_THV, inputs, grad_THV)
        for name, actual, expected in zip(
            ("q", "k", "v", "a", "b"), actual_grads, expected_grads
        ):
            with self.subTest(grad=name):
                self.assertLess(_relative_error(actual, expected), 3e-2)

    def test_unsupported_modes_raise(self):
        from torchtitan.models.common.attention import gdn
        from torchtitan.models.common.attention.cp_gdn import ContextParallelInnerGDN

        kernel = gdn.GDNKernel.Config().build()
        num_tokens = self._CU_SEQLENS[-1]
        bf16 = dict(device=self.device, dtype=torch.bfloat16)
        q_THK = torch.randn(num_tokens, _NUM_K_HEADS, _HEAD_K_DIM, **bf16)
        v_THV = torch.randn(num_tokens, _NUM_V_HEADS, _HEAD_V_DIM, **bf16)
        g_TH = -torch.rand(num_tokens, _NUM_V_HEADS, device=self.device)
        beta_TH = torch.rand(num_tokens, _NUM_V_HEADS, **bf16)
        args = (q_THK, q_THK, v_THV, g_TH, beta_TH)

        with mock.patch.object(gdn, "is_in_batch_invariant_mode", return_value=True):
            with self.assertRaisesRegex(NotImplementedError, "Batch-invariant"):
                kernel(*args, cu_seqlens=self.cu_seqlens)

        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with self.assertRaisesRegex(
                RuntimeError, "no deterministic implementation"
            ):
                kernel(*args, cu_seqlens=self.cu_seqlens)
            torch.use_deterministic_algorithms(True, warn_only=True)
            with self.assertWarnsRegex(UserWarning, "no deterministic implementation"):
                kernel(*args, cu_seqlens=self.cu_seqlens)
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

        with self.assertRaisesRegex(NotImplementedError, "context parallelism"):
            ContextParallelInnerGDN.Config(
                conv_kernel_size=_WIDTH, kernel=gdn.GDNKernel.Config()
            ).build()


if __name__ == "__main__":
    unittest.main()
