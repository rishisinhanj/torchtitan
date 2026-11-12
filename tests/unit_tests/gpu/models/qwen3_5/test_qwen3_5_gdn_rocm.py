# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated DeltaNet's ROCm kernels against the float32 gated delta rule.

The trainer runs flash-linear-attention's chunk kernel and a torch convolution;
the vLLM generator runs vLLM's Triton kernels on its paged state pools.
"""

import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

_ON_ROCM = torch.cuda.is_available() and torch.version.hip is not None

# Shape suffixes: T tokens, C channels, H heads, K key head dim, V value head dim,
# W convolution width, S state slots.
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
    a_TH: torch.Tensor, b_TH: torch.Tensor, A_log_H: torch.Tensor, dt_bias_H: torch.Tensor
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
        actual_grads = torch.autograd.grad(actual_TC.float(), (x_TC, weight_C1W), grad_TC)
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
        actual_THV = kernel(q_THK, k_THK, v_THV, g_TH, beta_TH, cu_seqlens=self.cu_seqlens)

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


@unittest.skipUnless(_ON_ROCM, "ROCm GDN kernels need an AMD GPU")
class TestGeneratorGDNOnROCm(unittest.TestCase):
    """vLLM's Triton GDN kernels on paged state through prefill, mixed and decode steps."""

    _NUM_SLOTS = 12
    _MAX_NUM_SEQS = 8

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        from torchtitan.rl.generator.vllm.attention.gdn import VLLMInnerGDN

        self.channels = 2 * _NUM_K_HEADS * _HEAD_K_DIM + _NUM_V_HEADS * _HEAD_V_DIM
        inner = VLLMInnerGDN.__new__(VLLMInnerGDN)
        torch.nn.Module.__init__(inner)
        inner.local_num_k_heads = _NUM_K_HEADS
        inner.local_num_v_heads = _NUM_V_HEADS
        inner.head_k_dim = _HEAD_K_DIM
        inner.head_v_dim = _HEAD_V_DIM
        inner.local_key_dim = _NUM_K_HEADS * _HEAD_K_DIM
        # Garbage in every slot: fresh requests must not read it.
        inner.kv_cache = (
            torch.randn(
                self._NUM_SLOTS,
                _WIDTH - 1,
                self.channels,
                device=self.device,
                dtype=torch.bfloat16,
            ),
            torch.randn(
                self._NUM_SLOTS,
                _NUM_V_HEADS,
                _HEAD_V_DIM,
                _HEAD_K_DIM,
                device=self.device,
            ),
        )
        self.inner = inner
        self.conv_weight_CW = 0.5 * torch.randn(
            self.channels, _WIDTH, device=self.device, dtype=torch.bfloat16
        )
        self.A_log_H = torch.empty(_NUM_V_HEADS, device=self.device).uniform_(0, 1.5)
        self.dt_bias_H = torch.randn(_NUM_V_HEADS, device=self.device)
        # Each request's tokens across steps, for the full-sequence reference.
        self.history: dict[int, list[tuple[torch.Tensor, ...]]] = {}

    def _tokens(self, slot: int, num_tokens: int) -> tuple[torch.Tensor, ...]:
        bf16 = dict(device=self.device, dtype=torch.bfloat16)
        tokens = (
            torch.randn(num_tokens, self.channels, **bf16),
            torch.randn(num_tokens, _NUM_V_HEADS, **bf16),
            torch.randn(num_tokens, _NUM_V_HEADS, **bf16),
        )
        self.history.setdefault(slot, []).append(tokens)
        return tokens

    def _packed_metadata(
        self, slots: list[int], lengths: list[int], fresh: list[bool], num_decodes: int
    ) -> SimpleNamespace:
        """Packed-step metadata with the builder's extents: capacity + 2 offsets, + 1 slots."""
        from torchtitan.rl.generator.vllm.attention.linear_attention import (
            LinearAttentionExecutionPath,
        )

        num_actual_tokens = sum(lengths)
        capacity = min(self._MAX_NUM_SEQS, num_actual_tokens)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        offsets += [num_actual_tokens] * (capacity + 2 - len(offsets))
        padding = capacity + 1 - len(slots)
        return SimpleNamespace(
            execution_path=LinearAttentionExecutionPath.PACKED,
            num_actual_tokens=num_actual_tokens,
            num_decodes=num_decodes,
            num_prefills=len(slots) - num_decodes,
            non_spec_query_start_loc=torch.tensor(
                offsets, dtype=torch.int32, device=self.device
            ),
            non_spec_state_indices_tensor=torch.tensor(
                slots + [0] * padding, dtype=torch.int32, device=self.device
            ),
            has_initial_state=torch.tensor(
                [not f for f in fresh] + [False] * padding, device=self.device
            ),
        )

    def _decode_metadata(self, slots: list[int], num_rows: int) -> SimpleNamespace:
        """Single-token metadata padded to num_rows with null slots, as for a FULL graph."""
        from torchtitan.rl.generator.vllm.attention.linear_attention import (
            LinearAttentionExecutionPath,
        )

        padding = num_rows - len(slots)
        return SimpleNamespace(
            execution_path=LinearAttentionExecutionPath.SINGLE_TOKEN,
            num_actual_tokens=num_rows,
            num_decodes=len(slots),
            num_prefills=0,
            non_spec_query_start_loc=torch.arange(
                num_rows + 1, dtype=torch.int32, device=self.device
            ),
            non_spec_state_indices_tensor=torch.tensor(
                slots + [0] * padding, dtype=torch.int32, device=self.device
            ),
            has_initial_state=torch.tensor(
                [True] * len(slots) + [False] * padding, device=self.device
            ),
        )

    def _run(
        self, metadata: SimpleNamespace, tokens: list[tuple[torch.Tensor, ...]], num_rows: int
    ) -> torch.Tensor:
        mixed_qkv_TC, a_TH, b_TH = (torch.cat(parts) for parts in zip(*tokens))
        extra = num_rows - mixed_qkv_TC.shape[0]
        mixed_qkv_TC, a_TH, b_TH = (
            F.pad(tensor, (0, 0, 0, extra)) for tensor in (mixed_qkv_TC, a_TH, b_TH)
        )
        return self.inner._forward_vllm_kernels(
            mixed_qkv_TC,
            a_TH,
            b_TH,
            self.conv_weight_CW,
            self.A_log_H,
            self.dt_bias_H,
            metadata,
            (num_rows, _NUM_V_HEADS, _HEAD_V_DIM),
        )

    def _reference(self, slot: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Output rows, final SSM state and convolution history over the request's tokens."""
        x_TC, a_TH, b_TH = (torch.cat(parts) for parts in zip(*self.history[slot]))
        conv_TC = _causal_conv(x_TC, self.conv_weight_CW)
        key_dim = _NUM_K_HEADS * _HEAD_K_DIM
        q_THK = conv_TC[:, :key_dim].unflatten(1, (_NUM_K_HEADS, _HEAD_K_DIM))
        k_THK = conv_TC[:, key_dim : 2 * key_dim].unflatten(1, (_NUM_K_HEADS, _HEAD_K_DIM))
        v_THV = conv_TC[:, 2 * key_dim :].unflatten(1, (_NUM_V_HEADS, _HEAD_V_DIM))
        out_THV, state_HVK = _gated_delta(
            q_THK, k_THK, v_THV, *_gates(a_TH, b_TH, self.A_log_H, self.dt_bias_H)
        )
        return out_THV, state_HVK, x_TC[-(_WIDTH - 1) :]

    def _assert_request_matches(
        self, slot: int, actual_THV: torch.Tensor, num_new_tokens: int
    ) -> None:
        out_THV, state_HVK, conv_history_WC = self._reference(slot)
        with self.subTest(slot=slot):
            self.assertLess(_relative_error(actual_THV, out_THV[-num_new_tokens:]), 2e-2)
            self.assertLess(
                _relative_error(self.inner.kv_cache[1][slot], state_HVK), 2e-2
            )
            torch.testing.assert_close(
                self.inner.kv_cache[0][slot], conv_history_WC, rtol=0, atol=0
            )

    def _prefill_then_mixed(self) -> None:
        # Three fresh prompts: shorter than, one and two-plus chunks long.
        tokens = [self._tokens(3, 37), self._tokens(5, 64), self._tokens(7, 129)]
        output = self._run(
            self._packed_metadata([3, 5, 7], [37, 64, 129], [True] * 3, 0),
            tokens,
            num_rows=236,
        )
        self._assert_request_matches(3, output[:37], 37)
        self._assert_request_matches(5, output[37:101], 64)
        self._assert_request_matches(7, output[101:230], 129)
        self.assertEqual(output[230:].abs().max().item(), 0)

        # Two decodes, a continued prefill and a fresh one, decodes first.
        tokens = [
            self._tokens(3, 1),
            self._tokens(5, 1),
            self._tokens(7, 20),
            self._tokens(9, 50),
        ]
        output = self._run(
            self._packed_metadata([3, 5, 7, 9], [1, 1, 20, 50], [False] * 3 + [True], 2),
            tokens,
            num_rows=72,
        )
        self._assert_request_matches(3, output[0:1], 1)
        self._assert_request_matches(5, output[1:2], 1)
        self._assert_request_matches(7, output[2:22], 20)
        self._assert_request_matches(9, output[22:72], 50)

    def test_prefill_mixed_and_decode_steps_match_reference(self):
        self._prefill_then_mixed()
        # Decode-only, padded with two null rows; one more row past the metadata.
        tokens = [self._tokens(slot, 1) for slot in (3, 5, 7, 9)]
        metadata = self._decode_metadata([3, 5, 7, 9], num_rows=6)
        output = self._run(metadata, tokens, num_rows=7)
        for row, slot in enumerate((3, 5, 7, 9)):
            self._assert_request_matches(slot, output[row : row + 1], 1)
        self.assertEqual(output[4:].abs().max().item(), 0)

    def test_decode_graph_replay_matches_eager(self):
        self._prefill_then_mixed()
        slots = [3, 5, 7, 9]
        mixed_qkv_TC, a_TH, b_TH = (
            torch.cat(parts)
            for parts in zip(*(self._tokens(slot, 1) for slot in slots))
        )
        num_rows = 6
        static_inputs = [
            F.pad(tensor, (0, 0, 0, num_rows - len(slots)))
            for tensor in (mixed_qkv_TC, a_TH, b_TH)
        ]
        metadata = self._decode_metadata(slots, num_rows)
        start_state = copy.deepcopy(self.inner.kv_cache)

        def step() -> torch.Tensor:
            return self.inner._forward_vllm_kernels(
                *static_inputs,
                self.conv_weight_CW,
                self.A_log_H,
                self.dt_bias_H,
                metadata,
                (num_rows, _NUM_V_HEADS, _HEAD_V_DIM),
            )

        eager_output = step()
        eager_state = copy.deepcopy(self.inner.kv_cache)

        # Capture records without running; replay from the pre-step state.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = step()
        for pool, start in zip(self.inner.kv_cache, start_state):
            pool.copy_(start)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
        for pool, expected in zip(self.inner.kv_cache, eager_state):
            torch.testing.assert_close(pool, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
