"""Opt-in decode expert gather (``FREETOKEN_DECODE_GATHER=tuned``).

Unset (the default), the offload cache's ``copy_missing`` keeps launching
``fast_index_copy_multi<1024, 8>`` exactly as before and this module's kernel is never
loaded. ``tuned`` makes decode misses (``ensure_experts`` plans; the whole-layer
``materialize_layer`` copy and the prefill hit-D2D gather are untouched) launch the kernel
pinned in ``TUNED_DECODE_GATHER`` from ``csrc/jit/decode_gather.cuh``: the call's missed bytes
are split evenly over a small fixed grid instead of a fixed set of CTAs per bank, and the
reads of mapped host memory are issued as TMA bulk copies (``cp.async.bulk``) through a
few-stage shared-memory ring per CTA, keeping the PCIe read queue full.

Provenance: the kernel is one JIT module (``module_spec``), listed by ``aot.py`` so the
kernel-cache build can produce it. A release that enables it must ship that .so in the
prebuilt kernel cache and bind it in ``kernel_cache_manifest.json`` (recipe digest + .so
sha256) like every other prebuilt module; otherwise the module is JIT-built into the private
recipe-digest-keyed dir (or refused under FREETOKEN_DISABLE_JIT).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from .fast_index_copy import _skip_fast_index_copy_enabled
from .utils import CppArgList, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module

DECODE_GATHER_ENV = "FREETOKEN_DECODE_GATHER"
# ("tma", CTAs, ring stages, chunk bytes) -> DecodeGatherTmaKernel<CTAs, STAGES, CHUNK>.
# 8 CTAs x 3 stages x 16 KiB = 384 KiB of reads in flight, 48 KiB of shared memory per CTA (no
# opt-in needed): the most consistent winner on the RTX 5070 Ti / 5060 Ti / PCIe 5.0 x16 host
# across 4 KiB and THP host banks, eager and graph replay.
TUNED_DECODE_GATHER = ("tma", 8, 3, 16384)
_DEFAULT_VALUES = {"", "0", "off", "default"}


def decode_gather_config() -> tuple | None:
    """The tuned gather's config tuple, or ``None`` for the default kernel.

    Fail closed: any value other than unset/``0``/``off``/``default``/``tuned`` raises, so a
    typo cannot silently fall back to the default path.
    """
    raw = os.getenv("FREETOKEN_DECODE_GATHER")
    if raw is None:
        raw = os.getenv("QWEN36_DECODE_GATHER", "")
    value = raw.strip().lower()
    if value in _DEFAULT_VALUES:
        return None
    if value == "tuned":
        return TUNED_DECODE_GATHER
    raise ValueError(f"FREETOKEN_DECODE_GATHER={raw!r}: expected 'tuned' (or unset for the default gather)")


def module_spec(config: tuple) -> tuple[str, CppArgList, str]:
    """(load_jit name stem, template args, launch wrapper) of a gather config; aot.py builds
    the prebuilt from exactly this, so the runtime recipe and the cached one cannot drift."""
    kind, *targs = config
    if kind != "tma" or len(targs) != 3:
        raise ValueError(f"unknown decode gather config {config!r}")
    args = make_cpp_args(*targs)
    return "decode_gather_tma", args, f"&DecodeGatherTmaKernel<{args}>::run"


@lru_cache(maxsize=None)
def _jit_decode_gather_module(config: tuple) -> Module:
    stem, args, wrapper = module_spec(config)
    return load_jit(
        stem,
        *args,
        cuda_files=["decode_gather.cuh"],
        cuda_wrappers=[("launch", wrapper)],
    )


def decode_gather_jit(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    feat_bytes: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor | None = None,
    *,
    config: tuple = TUNED_DECODE_GATHER,
) -> None:
    """Drop-in for ``fast_index_copy_multi_jit`` (same arguments, same bytes moved).

    Copies row ``src_indices[i]`` -> ``dst_indices[i]`` of every bank for ``i < num_indices``
    (read on the device: one fixed-shape launch, CUDA-graph capturable). At most 8 banks;
    per-row byte sizes must be multiples of 16 and the bases 16-byte aligned (the offload
    cache's fused copy plan guarantees both).
    """
    if _skip_fast_index_copy_enabled():
        return
    module = _jit_decode_gather_module(tuple(config))
    module.launch(dst_ptrs, src_ptrs, feat_bytes, dst_indices, src_indices, num_indices)
