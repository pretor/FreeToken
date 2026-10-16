from contextlib import contextmanager

import torch


def _overlap_cache(monkeypatch, mode: str, num_layers: int = 3):
    import freetoken.moe.offload_cache as oc

    monkeypatch.setattr(oc, "_COPY_DEFER", mode)
    num_experts = 4
    cache = oc.OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=8,
        device=torch.device("cpu"),
        prefill_overlap=True,
    )
    cache.set_bank_sources({
        "gate_up": list(torch.zeros(num_layers * num_experts, 32, 8).split(num_experts)),
        "down": list(torch.zeros(num_layers * num_experts, 8, 16).split(num_experts)),
    })
    log: list[str] = []

    class FakeStream:
        def wait_event(self, event):
            log.append(f"wait {event.name}")

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream=None):
            log.append(f"record {self.name}")

    @contextmanager
    def fake_cuda_stream(stream):
        yield

    cache.prefill_copy_stream = FakeStream()
    cache.prefill_begin_event = FakeEvent("begin")
    cache.prefill_ready_events = [FakeEvent("ready0"), FakeEvent("ready1")]
    cache.prefill_release_events = [FakeEvent("release0"), FakeEvent("release1")]
    monkeypatch.setattr("torch.cuda.stream", fake_cuda_stream)
    monkeypatch.setattr("torch.cuda.current_stream", lambda device=None: object())
    return cache, log


def test_off_records_the_release_at_the_gemms(monkeypatch):
    cache, log = _overlap_cache(monkeypatch, "off")
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    cache.release_prefill_layer(0)
    assert log[-1] == "record release0"
    cache.mark_layer_reduced(0)
    assert log[-1] == "record release0"
    cache.prefetch_prefill_layer(2)
    assert log[-2:] == ["wait release0", "record ready0"]


def test_prev_ar_records_the_release_after_the_all_reduce(monkeypatch):
    cache, log = _overlap_cache(monkeypatch, "prev_ar")
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    cache.release_prefill_layer(0)
    assert "record release0" not in log
    cache.mark_layer_reduced(0)
    assert log[-1] == "record release0"
    cache.prefetch_prefill_layer(2)
    assert log[-3:] == ["record release0", "wait release0", "record ready0"]


def test_prev_ar_without_an_all_reduce_records_the_release_at_the_reuse(monkeypatch):
    cache, log = _overlap_cache(monkeypatch, "prev_ar")
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    cache.release_prefill_layer(0)
    cache.prefetch_prefill_layer(2)
    assert log[-3:] == ["record release0", "wait release0", "record ready0"]
