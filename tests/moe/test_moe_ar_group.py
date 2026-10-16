import os

import pytest
import torch


def test_token_groups_one_shot_when_disabled_or_small():
    from freetoken.models.qwen4_exp.moe import token_groups

    assert list(token_groups(4096, 0)) == [(0, 4096)]
    assert list(token_groups(1024, 2048)) == [(0, 1024)]
    assert list(token_groups(2048, 2048)) == [(0, 2048)]


def test_token_groups_covers_a_wide_chunk():
    from freetoken.models.qwen4_exp.moe import token_groups

    groups = list(token_groups(4096, 2048))
    assert groups == [(0, 2048), (2048, 2048)]
    groups = list(token_groups(5000, 2048))
    assert groups == [(0, 2048), (2048, 2048), (4096, 904)]
    assert sum(n for _, n in groups) == 5000


def test_grouped_path_reduces_each_slice_then_marks(monkeypatch):
    """The side-stream all-reduce is called once per group; mark_layer_reduced after the last."""
    monkeypatch.setenv("FREETOKEN_MOE_AR_GROUP", "2")
    import importlib

    import freetoken.models.qwen4_exp.moe as mod

    importlib.reload(mod)
    monkeypatch.setattr(mod, "shared_gate_mul_add", lambda routed, shared, gate: routed)

    reduced = []
    marks = []

    class FakeCache:
        def release_prefill_layer(self, layer_id):
            reduced.append(("release", layer_id))

        def mark_layer_reduced(self, layer_id):
            marks.append(layer_id)

    class FakeExperts:
        layer_id = 3
        offload_cache = FakeCache()

        def prefill_bind(self, hidden, logits):
            from freetoken.layers.moe import PrefillBind

            t = hidden.shape[0]
            return PrefillBind(
                topk_weights=torch.ones(t, 2),
                topk_ids=torch.zeros(t, 2, dtype=torch.int32),
                views=(),
                n=1,
                alphas=None,
                release=True,
            )

        def _expert_gemm(self, cache, hidden, weights, ids, **kwargs):
            return hidden.clone()

    class FakeComm:
        def __init__(self):
            self.calls = []

        def all_reduce(self, x):
            self.calls.append(x.shape[0])
            return x

    moe = object.__new__(mod.Qwen4ExpMoE)
    moe._comm = FakeComm()
    moe._ar_stream = None
    hidden = torch.arange(6, dtype=torch.float32).view(6, 1).expand(6, 4).contiguous()
    shared = torch.zeros_like(hidden)
    gate = torch.ones(6, 1)
    # CPU: skip CUDA stream overlap, still exercise the group loop via a patched helper
    monkeypatch.setattr(mod.Qwen4ExpMoE, "_side_stream", lambda self, device: _FakeStream())
    monkeypatch.setattr(
        mod.Qwen4ExpMoE,
        "_all_reduce_on",
        lambda self, x, stream: self._comm.all_reduce(x),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _FakeStream())
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)

    out = moe._prefill_grouped(FakeExperts(), hidden, None, shared, gate)
    assert moe._comm.calls == [2, 2, 2]
    assert marks == [3]
    assert reduced == [("release", 3)]
    assert out.shape == hidden.shape


class _FakeStream:
    def wait_stream(self, other):
        return None

    def wait_event(self, ev):
        return None


class _FakeEvent:
    def record(self, stream=None):
        return None
