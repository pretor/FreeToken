from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers import LinearRowParallel, silu_and_mul
from freetoken.layers.moe import OffloadMoELayer
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Prefill MoE all-reduce in token groups so GEMM(group i+1) overlaps AR(group i) on a
# second stream (capy-flash T62). 0 keeps one all-reduce of the whole chunk.
_AR_GROUP = int(os.getenv("FREETOKEN_MOE_AR_GROUP", "0") or 0)


def token_groups(num_tokens: int, group: int) -> Iterator[tuple[int, int]]:
    """Inclusive-length slices ``(start, n)`` covering ``[0, num_tokens)``."""
    if group <= 0 or num_tokens <= group:
        yield 0, num_tokens
        return
    start = 0
    while start < num_tokens:
        n = min(group, num_tokens - start)
        yield start, n
        start += n


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.

    TP: the offload experts are sharded along the intermediate axis and the bf16 shared expert is
    row-parallel, so both produce partial sums; ``routed + gate * shared`` is linear in them and
    is reduced once (one all-reduce per MoE layer instead of two).
    """

    def __init__(
        self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""
    ) -> None:
        super().__init__(config, layer_id=layer_id, prefix=prefix)
        self._comm = DistributedCommunicator()
        self._tp_size = get_tp_info().size
        self._ar_stream: torch.cuda.Stream | None = None

    def _side_stream(self, device: torch.device) -> torch.cuda.Stream:
        if self._ar_stream is None:
            self._ar_stream = torch.cuda.Stream(device=device)
        return self._ar_stream

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        res = self.forward_parts(hidden_states)
        if isinstance(res, tuple):
            routed, shared, gate = res
            return shared_gate_mul_add(routed, shared, gate)
        return res

    def forward_parts(self, hidden_states: torch.Tensor):
        """The epilogue split into parts: the caller may fold mul-add into its own kernel."""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        se, ex = self.shared_expert, self.experts
        if (
            self._tp_size > 1
            and isinstance(se.down_proj, LinearRowParallel)
            and isinstance(ex, OffloadMoELayer)
        ):
            is_prefill = get_global_ctx().batch.is_prefill
            if is_prefill:
                act = silu_and_mul(se.gate_up_proj.forward(hidden_states))
                shared = se.down_proj.quant_method.apply(se.down_proj, act)
                if _AR_GROUP > 0 and num_tokens > _AR_GROUP:
                    out = self._prefill_grouped(ex, hidden_states, router_logits, shared, gate)
                else:
                    routed = ex.prefill_forward(hidden_states, router_logits)
                    out = self._comm.all_reduce(shared_gate_mul_add(routed, shared, gate))
                    if ex.offload_cache is not None:
                        ex.offload_cache.mark_layer_reduced(ex.layer_id)
            elif ex.can_overlap_shared(hidden_states):
                def calc_shared():
                    act = silu_and_mul(se.gate_up_proj.forward(hidden_states))
                    return se.down_proj.quant_method.apply(se.down_proj, act)
                routed, shared = ex.decode_forward_with_shared(hidden_states, router_logits, calc_shared)
                out = self._comm.all_reduce(shared_gate_mul_add(routed, shared, gate))
            else:
                act = silu_and_mul(se.gate_up_proj.forward(hidden_states))
                shared = se.down_proj.quant_method.apply(se.down_proj, act)
                routed = ex.decode_forward(hidden_states, router_logits)
                out = self._comm.all_reduce(shared_gate_mul_add(routed, shared, gate))
            return out.view(num_tokens, hidden_dim)

        can_overlap = getattr(ex, "can_overlap_shared", None)
        if can_overlap is not None and can_overlap(hidden_states):
            routed, shared = ex.decode_forward_with_shared(
                hidden_states, router_logits, lambda: se.forward(hidden_states)
            )
        else:
            shared = se.forward(hidden_states)
            routed = ex.forward(hidden_states=hidden_states, router_logits=router_logits)
        return routed.view(num_tokens, hidden_dim), shared, gate

    def _prefill_grouped(
        self,
        ex: OffloadMoELayer,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Routed GEMM in token groups; each group's all-reduce runs on a side stream so the
        next GEMM overlaps it. ``mark_layer_reduced`` runs after the last group is queued and
        the compute stream has waited, so ``prev_ar`` prefetch still starts after this layer's
        all-reduces.
        """
        bind = ex.prefill_bind(hidden_states, router_logits)
        compute = torch.cuda.current_stream(hidden_states.device)
        ar_stream = self._side_stream(hidden_states.device)
        out = torch.empty_like(hidden_states)
        for start, n in token_groups(hidden_states.shape[0], _AR_GROUP):
            sl = slice(start, start + n)
            routed = ex._expert_gemm(
                ex.offload_cache,
                hidden_states[sl],
                bind.topk_weights[sl],
                bind.topk_ids[sl],
                views=bind.views,
                n=bind.n,
                alphas=bind.alphas,
                is_prefill=True,
            )
            chunk = shared_gate_mul_add(routed, shared[sl], gate[sl])
            out[sl].copy_(chunk)
            ready = torch.cuda.Event()
            ready.record(compute)
            ar_stream.wait_event(ready)
            self._all_reduce_on(out[sl], ar_stream)
        if bind.release and ex.offload_cache is not None:
            ex.offload_cache.release_prefill_layer(ex.layer_id)
        compute.wait_stream(ar_stream)
        if ex.offload_cache is not None:
            ex.offload_cache.mark_layer_reduced(ex.layer_id)
        return out

    def _all_reduce_on(self, x: torch.Tensor, stream: torch.cuda.Stream) -> None:
        import tvm_ffi

        # PyNCCL reads TVMFFIEnvGetStream; use_torch_stream keeps that in sync with ``stream``.
        with tvm_ffi.use_torch_stream(torch.cuda.stream(stream)):
            self._comm.all_reduce(x)


__all__ = ["Qwen4ExpMoE", "token_groups"]
