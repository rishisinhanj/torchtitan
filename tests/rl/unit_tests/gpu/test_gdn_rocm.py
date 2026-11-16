# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The vLLM generator's GDN on ROCm (vLLM's Triton kernels) against the float32 gated
delta rule, through prefill, mixed and decode steps on paged state pools.

Shape suffixes: T tokens, C channels, H value heads, HK key heads, K key head dim,
V value head dim, W convolution width.
"""

import copy

import pytest
import torch
import torch.nn.functional as F
from torchtitan.rl.generator.vllm.attention import gdn
from torchtitan.rl.generator.vllm.attention.linear_attention import (
    LinearAttentionExecutionPath,
    TorchTitanLinearAttentionMetadataBuilder,
)
from vllm.config import SchedulerConfig, VllmConfig
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="the vLLM-kernel GDN path runs on ROCm",
)

NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, WIDTH = 2, 4, 128, 128, 4
CHANNELS = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM
NUM_SLOTS, MAX_NUM_SEQS = 12, 8


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _gated_delta(
    q_THK: torch.Tensor,
    k_THK: torch.Tensor,
    v_THV: torch.Tensor,
    g_TH: torch.Tensor,
    beta_TH: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential float32 recurrence over one sequence; value head h reads key head h // r."""
    num_v_heads, head_v_dim = v_THV.shape[1:]
    head_k_dim = q_THK.shape[-1]
    ratio = num_v_heads // q_THK.shape[1]
    q_THK = _l2norm(q_THK.float()).repeat_interleave(ratio, 1) * head_k_dim**-0.5
    k_THK = _l2norm(k_THK.float()).repeat_interleave(ratio, 1)
    state_HVK = q_THK.new_zeros(num_v_heads, head_v_dim, head_k_dim)
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
    x_1CT = F.pad(x_TC.float().transpose(0, 1).unsqueeze(0), [WIDTH - 1, 0])
    out_1CT = F.conv1d(x_1CT, weight_CW.float().unsqueeze(1), groups=x_TC.shape[1])
    return F.silu(out_1CT).squeeze(0).transpose(0, 1)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.float()
    return ((actual.float() - expected).norm() / expected.norm()).item()


class TestGeneratorGDNOnROCm:
    """vLLM's Triton GDN kernels on paged state through prefill, mixed and decode steps."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setattr(gdn, "is_in_batch_invariant_mode", lambda: False)
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        layer = gdn.VLLMInnerGDN.__new__(gdn.VLLMInnerGDN)
        torch.nn.Module.__init__(layer)
        layer.prefix = "gdn"
        layer.local_num_k_heads, layer.local_num_v_heads = NUM_K_HEADS, NUM_V_HEADS
        layer.head_k_dim, layer.head_v_dim = HEAD_K_DIM, HEAD_V_DIM
        # Garbage in every slot: fresh requests must not read it.
        layer.kv_cache = (
            torch.randn(
                NUM_SLOTS, WIDTH - 1, CHANNELS, device=self.device, dtype=torch.bfloat16
            ),
            torch.randn(
                NUM_SLOTS, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM, device=self.device
            ),
        )
        self.layer = layer
        self.conv_weight_CW = 0.5 * torch.randn(
            CHANNELS, WIDTH, device=self.device, dtype=torch.bfloat16
        )
        self.A_log_H = torch.empty(NUM_V_HEADS, device=self.device).uniform_(0, 1.5)
        self.dt_bias_H = torch.randn(NUM_V_HEADS, device=self.device)
        # Each request's tokens across steps, for the full-sequence reference.
        self.history: dict[int, list[tuple[torch.Tensor, ...]]] = {}
        config = VllmConfig()
        config.scheduler_config = SchedulerConfig(
            max_model_len=1024,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=1024,
            is_encoder_decoder=False,
        )
        config.cache_config.mamba_cache_mode = "none"
        spec = MambaSpec(
            block_size=1024,
            shapes=((WIDTH - 1, CHANNELS), (NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM)),
            dtypes=(torch.bfloat16, torch.float32),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        )
        self.builder = TorchTitanLinearAttentionMetadataBuilder(
            spec, [], config, self.device
        )

    def _tokens(self, slot: int, num_tokens: int) -> tuple[torch.Tensor, ...]:
        bf16 = dict(device=self.device, dtype=torch.bfloat16)
        tokens = (
            torch.randn(num_tokens, CHANNELS, **bf16),
            torch.randn(num_tokens, NUM_V_HEADS, **bf16),
            torch.randn(num_tokens, NUM_V_HEADS, **bf16),
        )
        self.history.setdefault(slot, []).append(tokens)
        return tokens

    def _metadata(self, slots: list[int], lengths: list[int], num_tokens: int):
        """The builder's metadata for requests in ``slots`` scheduling ``lengths`` tokens,
        padded to ``num_tokens`` rows (a FULL bucket when larger)."""
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        seq_lens = [
            sum(part[0].shape[0] for part in self.history[slot]) for slot in slots
        ]
        query_start_loc = torch.tensor(offsets, dtype=torch.int32)
        return self.builder.build(
            0,
            CommonAttentionMetadata(
                query_start_loc=query_start_loc.to(self.device),
                query_start_loc_cpu=query_start_loc,
                seq_lens=torch.tensor(seq_lens, device=self.device, dtype=torch.int32),
                num_reqs=len(slots),
                num_actual_tokens=num_tokens,
                max_query_len=max(lengths),
                max_seq_len=max(seq_lens),
                block_table_tensor=torch.tensor(
                    slots, device=self.device, dtype=torch.int32
                ).unsqueeze(1),
                slot_mapping=torch.zeros(
                    num_tokens, device=self.device, dtype=torch.int64
                ),
            ),
        )

    def _run(self, metadata, tokens: list[tuple[torch.Tensor, ...]], num_rows: int):
        mixed_qkv_TC, a_TH, b_TH = (
            F.pad(
                torch.cat(parts), (0, 0, 0, num_rows - sum(p.shape[0] for p in parts))
            )
            for parts in zip(*tokens)
        )
        context = ForwardContext(
            no_compile_layers={}, attn_metadata={"gdn": metadata}, slot_mapping={}
        )
        with torch.inference_mode(), override_forward_context(context):
            return self.layer._forward(
                mixed_qkv_TC,
                a_TH,
                b_TH,
                self.conv_weight_CW,
                None,
                self.A_log_H,
                self.dt_bias_H,
            )

    def _reference(self, slot: int) -> tuple[torch.Tensor, ...]:
        """Output rows, final SSM state and convolution history over the request's tokens."""
        x_TC, a_TH, b_TH = (torch.cat(parts) for parts in zip(*self.history[slot]))
        conv_TC = _causal_conv(x_TC, self.conv_weight_CW)
        key_dim = NUM_K_HEADS * HEAD_K_DIM
        g_TH = -torch.exp(self.A_log_H) * F.softplus(a_TH.float() + self.dt_bias_H)
        out_THV, state_HVK = _gated_delta(
            conv_TC[:, :key_dim].unflatten(1, (NUM_K_HEADS, HEAD_K_DIM)),
            conv_TC[:, key_dim : 2 * key_dim].unflatten(1, (NUM_K_HEADS, HEAD_K_DIM)),
            conv_TC[:, 2 * key_dim :].unflatten(1, (NUM_V_HEADS, HEAD_V_DIM)),
            g_TH,
            torch.sigmoid(b_TH.float()),
        )
        # The history of a request shorter than the window starts with zeros.
        return out_THV, state_HVK, F.pad(x_TC, (0, 0, WIDTH - 1, 0))[-(WIDTH - 1) :]

    def _assert_request_matches(
        self, slot: int, actual_THV: torch.Tensor, num_new_tokens: int
    ) -> None:
        out_THV, state_HVK, conv_history_WC = self._reference(slot)
        assert _relative_error(actual_THV, out_THV[-num_new_tokens:]) < 2e-2, slot
        assert _relative_error(self.layer.kv_cache[1][slot], state_HVK) < 2e-2, slot
        torch.testing.assert_close(
            self.layer.kv_cache[0][slot], conv_history_WC, rtol=0, atol=0
        )

    def _prefill_then_mixed(self) -> None:
        # Three fresh prompts: shorter than, one and two-plus chunks long.
        tokens = [self._tokens(3, 37), self._tokens(5, 64), self._tokens(7, 129)]
        metadata = self._metadata([3, 5, 7], [37, 64, 129], 230)
        assert metadata.execution_path is LinearAttentionExecutionPath.PACKED
        output = self._run(metadata, tokens, num_rows=236)
        self._assert_request_matches(3, output[:37], 37)
        self._assert_request_matches(5, output[37:101], 64)
        self._assert_request_matches(7, output[101:230], 129)
        assert not output[230:].any()

        # Two decodes, a continued prefill and a fresh one, decodes first; then a
        # fresh one-token prompt among continued prefills.
        tokens = [
            self._tokens(3, 1),
            self._tokens(5, 1),
            self._tokens(7, 20),
            self._tokens(9, 50),
        ]
        output = self._run(self._metadata([3, 5, 7, 9], [1, 1, 20, 50], 72), tokens, 72)
        self._assert_request_matches(3, output[0:1], 1)
        self._assert_request_matches(5, output[1:2], 1)
        self._assert_request_matches(7, output[2:22], 20)
        self._assert_request_matches(9, output[22:72], 50)
        tokens = [self._tokens(11, 1), self._tokens(7, 5), self._tokens(9, 3)]
        output = self._run(self._metadata([11, 7, 9], [1, 5, 3], 9), tokens, 9)
        self._assert_request_matches(11, output[0:1], 1)
        self._assert_request_matches(7, output[1:6], 5)
        self._assert_request_matches(9, output[6:9], 3)

    def _decode_metadata(self, slots: list[int], num_rows: int):
        for slot in slots:
            self._tokens(slot, 1)
        metadata = self._metadata(slots, [1] * len(slots), num_rows)
        assert metadata.execution_path is LinearAttentionExecutionPath.SINGLE_TOKEN
        return metadata

    def test_prefill_mixed_and_decode_steps_match_reference(self):
        self._prefill_then_mixed()
        slots = [3, 5, 7, 9]
        # Decode-only, padded with two null rows (a FULL bucket of 6), the last as a
        # negative sentinel; one more row past the metadata.
        metadata = self._decode_metadata(slots, num_rows=6)
        metadata.non_spec_state_indices_tensor[5] = -1
        untouched = [slot for slot in range(NUM_SLOTS) if slot not in slots]
        before = [pool[untouched].clone() for pool in self.layer.kv_cache]
        output = self._run(
            metadata, [self.history[slot][-1] for slot in slots], num_rows=7
        )
        for row, slot in enumerate(slots):
            self._assert_request_matches(slot, output[row : row + 1], 1)
        assert not output[4:].any()
        # Null rows leave every other slot, slot 0 included, bitwise unchanged.
        for pool, expected in zip(self.layer.kv_cache, before):
            assert torch.equal(
                pool[untouched].view(torch.uint8), expected.view(torch.uint8)
            )

    def test_decode_graph_replay_matches_eager(self):
        self._prefill_then_mixed()
        slots = [3, 5, 7, 9]
        metadata = self._decode_metadata(slots, num_rows=6)
        tokens = [self.history[slot][-1] for slot in slots]
        static_inputs = [
            F.pad(torch.cat(parts), (0, 0, 0, 2)) for parts in zip(*tokens)
        ]
        start_state = copy.deepcopy(self.layer.kv_cache)
        context = ForwardContext(
            no_compile_layers={}, attn_metadata={"gdn": metadata}, slot_mapping={}
        )

        def step() -> torch.Tensor:
            return self.layer._forward(
                *static_inputs,
                self.conv_weight_CW,
                None,
                self.A_log_H,
                self.dt_bias_H,
            )

        with torch.inference_mode(), override_forward_context(context):
            eager_output = step()
            eager_state = copy.deepcopy(self.layer.kv_cache)
            # Capture records without running; replay from the pre-step state.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = step()
            for pool, start in zip(self.layer.kv_cache, start_state):
                pool.copy_(start)
            graph_output.fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
        for pool, expected in zip(self.layer.kv_cache, eager_state):
            torch.testing.assert_close(pool, expected, rtol=0, atol=0)

    def test_unsupported_modes_raise(self, monkeypatch):
        tokens = [self._tokens(3, 37)]
        metadata = self._metadata([3], [37], 37)
        context = ForwardContext(
            no_compile_layers={}, attn_metadata={"gdn": metadata}, slot_mapping={}
        )
        inputs = [torch.cat(parts) for parts in zip(*tokens)]
        with torch.inference_mode(), override_forward_context(context):
            with pytest.raises(NotImplementedError, match="FULL_DECODE_ONLY"):
                with torch.cuda.graph(torch.cuda.CUDAGraph()):
                    self.layer._forward(
                        *inputs, self.conv_weight_CW, None, self.A_log_H, self.dt_bias_H
                    )
            monkeypatch.setattr(gdn, "is_in_batch_invariant_mode", lambda: True)
            with pytest.raises(NotImplementedError, match="Batch-invariant"):
                self.layer._forward(
                    *inputs, self.conv_weight_CW, None, self.A_log_H, self.dt_bias_H
                )
