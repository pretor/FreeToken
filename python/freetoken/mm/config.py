"""Runtime knobs of the multimodal path. The architecture side (vision_config, mrope) lives in ModelConfig."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# encoder tower kinds a family can register; --mm-disable <kind> leaves that tower unbuilt and refuses its inputs
ENCODER_KINDS = ("vision", "audio")
# checkpoint config sections that describe encoder towers; the parser never sees the section of a tower this process does not build
ENCODER_SECTIONS = ("vision_config", "audio_config")
# families whose tower --mm-encoder-weights cpu can run outside the engine: the shared Qwen VL tower
CPU_ENCODER_PROCESSORS = ("freetoken.mm.processors.qwen_vl:QwenVLMMProcessor",)
# the image token cap --mm-encoder-weights cpu applies unless --image-max-tokens is given: the Qwen VL
# checkpoints allow 16k tokens per image, minutes of CPU time
CPU_ENCODER_DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class MultimodalConfig:
    # encoder kinds (ENCODER_KINDS) whose tower this process does not build; --text-model-only names them all
    disabled_encoders: frozenset[str] = frozenset()
    # Where encoded items wait between prefill chunks. "cpu": pinned host memory, "cuda": the device.
    embed_cache_device: Literal["cpu", "cuda"] = "cpu"
    # Encoder tower block weights. "host": pinned host banks streamed two blocks at a time behind the compute, "gpu": resident,
    # "cpu": the whole tower in host RAM, run by its own CPU process before a request reaches the engine (no VRAM).
    encoder_weights: Literal["gpu", "host", "cpu"] = "host"
    # per-image token budget; the family's MMProcessor converts it to its image processor's own limits, None keeps the checkpoint defaults
    image_min_tokens: int | None = None
    image_max_tokens: int | None = None
    # extra keyword arguments the family's MMProcessor passes to the image processor call, after the token budget
    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    # the "cpu" encoder process: torch threads (None: one NUMA node's physical cores), the CPUs it is pinned to
    # (None: inherited) and its cache of encoded images in MiB (chat clients resend an image every turn)
    encoder_threads: int | None = None
    encoder_cpus: tuple[int, ...] | None = None
    encoder_cache_mb: int = 256

    @property
    def text_model_only(self) -> bool:
        return set(ENCODER_KINDS) <= self.disabled_encoders

    @property
    def encoder_out_of_process(self) -> bool:
        return self.encoder_weights == "cpu"


def parse_cpu_list(text: str) -> tuple[int, ...]:
    """A CPU list as sysfs, taskset and numactl spell it ("22-43", "0-3,8") -> the sorted ids; ValueError if malformed."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        lo, _, hi = part.strip().partition("-")
        first, last = int(lo), int(hi or lo)
        if first < 0 or last < first:
            raise ValueError(f"bad CPU range {part!r}")
        cpus.update(range(first, last + 1))
    return tuple(sorted(cpus))


def check_cpu_encoder_support(mm_processor: str | None) -> None:
    """Raise unless the family's tower is one the CPU encoder process can run."""
    if mm_processor not in CPU_ENCODER_PROCESSORS:
        raise ValueError(
            "--mm-encoder-weights cpu runs the Qwen VL vision tower only; this model needs "
            "--mm-encoder-weights gpu or host (or --text-model-only)"
        )


__all__ = [
    "CPU_ENCODER_DEFAULT_MAX_TOKENS",
    "CPU_ENCODER_PROCESSORS",
    "ENCODER_KINDS",
    "ENCODER_SECTIONS",
    "MultimodalConfig",
    "check_cpu_encoder_support",
    "parse_cpu_list",
]
