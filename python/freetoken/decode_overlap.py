"""Default-off, same-layer expert miss gather and shared expert fork-join overlap.

Captures cleanly into CUDA graphs by keeping a persistent side stream and pre-materialized
event pairs per layer. Ensures that the missing expert PCIe gather runs in parallel with
the shared expert MLP on the compute stream, masking host-to-device PCIe latency during decode.
"""
from __future__ import annotations

import os
from typing import Callable, TypeVar

T = TypeVar("T")

ENV = "FREETOKEN_DECODE_SHARED_OVERLAP"
PRIORITY_ENV = "FREETOKEN_DECODE_OVERLAP_PRIORITY"


def _flag(name: str) -> bool:
    val = os.getenv(name)
    if val is None:
        val = os.getenv("QWEN36_" + name.replace("FREETOKEN_", ""), "0")
    return val.strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    return _flag(ENV)


class DecodeSharedOverlap:
    def __init__(self, cuda, device, num_layers: int = 40):
        self.cuda = cuda
        self.device = device
        self.num_layers = num_layers
        priority = -1 if _flag(PRIORITY_ENV) else 0
        self.stream = cuda.Stream(device=device, priority=priority)
        self.ready = [cuda.Event() for _ in range(num_layers)]
        self.done = [cuda.Event() for _ in range(num_layers)]

        # Materialize CUDA handles before graph capture
        producer = cuda.current_stream(device)
        for i in range(num_layers):
            self.ready[i].record(producer)
            self.stream.wait_event(self.ready[i])
            self.done[i].record(self.stream)
            producer.wait_event(self.done[i])

    def run(self, layer_id: int, copy_missing: Callable[[], None], shared: Callable[[], T]) -> T:
        producer = self.cuda.current_stream(self.device)
        ready_evt = self.ready[layer_id]
        done_evt = self.done[layer_id]

        ready_evt.record(producer)
        try:
            with self.cuda.stream(self.stream):
                self.stream.wait_event(ready_evt)
                try:
                    copy_missing()
                finally:
                    done_evt.record(self.stream)
            return shared()
        finally:
            producer.wait_event(done_evt)
