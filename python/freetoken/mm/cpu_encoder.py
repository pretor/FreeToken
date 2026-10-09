"""The vision tower on the CPU, outside the engine (--mm-encoder-weights cpu).

One process holds the Qwen VL tower in host RAM, turns the images of a request into the rows the engine
would have encoded, and forwards the request to the scheduler with ``precomputed_embeddings`` set: the TP
ranks build no tower and spend no VRAM on it. Text-only requests never pass through here.
"""

from __future__ import annotations

import glob
import os
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterator, List

import torch

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.message import MMItem, UserMsg
    from freetoken.mm.config import MultimodalConfig

logger = init_logger(__name__, "vision_cpu")


def default_encoder_threads() -> int:
    """The physical cores of the largest NUMA node this process may run on: the TP schedulers and the
    host-bank copies keep the rest of the machine."""
    from freetoken.mm.config import parse_cpu_list

    if not hasattr(os, "sched_getaffinity"):
        return max(1, (os.cpu_count() or 2) // 2)
    allowed = set(os.sched_getaffinity(0))
    nodes = []
    for path in glob.glob("/sys/devices/system/node/node[0-9]*/cpulist"):
        with open(path) as f:
            cpus = set(parse_cpu_list(f.read())) & allowed
        if cpus:
            nodes.append(cpus)
    # hyperthread siblings share one core's FMA units: one thread per distinct sibling set
    cores = set()
    for cpu in max(nodes, key=len) if nodes else allowed:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                cores.add(f.read().strip())
        except OSError:
            cores.add(str(cpu))
    return max(1, len(cores))


def configure_encoder_process(mm: MultimodalConfig) -> int:
    """Pin and deprioritize this process and size its torch pool; returns the thread count."""
    if mm.encoder_cpus:
        os.sched_setaffinity(0, mm.encoder_cpus)
    # a decode step of the TP schedulers must never wait behind an image
    os.nice(10)
    threads = mm.encoder_threads or (len(mm.encoder_cpus) if mm.encoder_cpus else default_encoder_threads())
    # set explicitly: launch scripts often export a small OMP_NUM_THREADS for the engine
    torch.set_num_threads(threads)
    return threads


def iter_vision_tensors(model_path: str) -> Iterator[tuple[str, torch.Tensor]]:
    """The tower's tensors as the engine's state dict names them (``visual.`` prefix), unsharded."""
    from freetoken.checkpoint.ftw import is_ftw_checkpoint, iter_ftw_weights
    from freetoken.models.config import VISION_KEY_PREFIXES

    if is_ftw_checkpoint(model_path):
        # an FTW stores the engine's TP=1 state dict, fused qkv included
        yield from iter_ftw_weights(model_path, keep=lambda name: name.startswith(VISION_KEY_PREFIXES))
        return
    from freetoken.models.weight import load_vision_weight

    yield from load_vision_weight(model_path, torch.device("cpu"))


def load_cpu_vision_tower(
    model_path: str, *, dtype: torch.dtype = torch.float32, rope_dtype: torch.dtype = torch.bfloat16
) -> Any:
    """The checkpoint's Qwen VL tower on the CPU in ``dtype``.

    FP32 by default: CPUs without AVX512-BF16 or AMX run BF16 matmuls slower than FP32, and FP32 stays closest
    to the reference. ``rope_dtype`` is the engine dtype, so the rope rows round as the GPU tower's do.
    """
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.quantization import finalize_quant
    from freetoken.models.qwen3_vl.config import parse_vision_config
    from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel
    from freetoken.utils import cached_load_hf_config, torch_dtype

    # the tower's parallel layers size themselves from the TP info: the encoder process holds it whole,
    # and an offline engine that encodes in-process runs at TP=1
    tp = try_get_tp_info()
    if tp is None:
        set_tp_info(rank=0, size=1)
    elif tp.size != 1:
        raise RuntimeError(f"the CPU vision tower needs a TP=1 process, this one is rank {tp.rank} of {tp.size}")
    vc = parse_vision_config(cached_load_hf_config(model_path))
    if vc is None:
        raise ValueError(f"{model_path} has no vision_config: there is no vision tower to run")
    with torch.device("meta"), torch_dtype(dtype):
        tower = Qwen3VLVisionModel(vc, quant_config=None, prefix="visual", rope_dtype=rope_dtype)
    # copy=True: the FTW reader lends transient buffers, and a tensor already in dtype would stay one
    state = {name.removeprefix("visual."): t.to(dtype=dtype, copy=True) for name, t in iter_vision_tensors(model_path)}
    tower.load_state_dict(state)
    finalize_quant(tower)
    return tower


class EmbeddingLRU:
    """Encoded images by content hash, bounded in bytes: chat clients resend an image every turn, and this
    process encodes before the scheduler's radix cache could tell that the rows are no longer needed."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.nbytes = 0
        self._entries: OrderedDict[int, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: int) -> torch.Tensor | None:
        emb = self._entries.get(key)
        if emb is not None:
            self._entries.move_to_end(key)
        return emb

    def put(self, key: int, emb: torch.Tensor) -> None:
        size = emb.numel() * emb.element_size()
        if key in self._entries or size > self.max_bytes:
            return
        self._entries[key] = emb
        self.nbytes += size
        while self.nbytes > self.max_bytes:
            _, old = self._entries.popitem(last=False)
            self.nbytes -= old.numel() * old.element_size()


class CpuVisionEncoder:
    """Swaps the pixels of a request's images (``feature``) for their embedding rows (``precomputed_embeddings``)."""

    def __init__(self, tower: Any, *, out_dtype: torch.dtype, cache_bytes: int) -> None:
        self.tower = tower
        self.out_dtype = out_dtype
        self.cache = EmbeddingLRU(cache_bytes)

    @classmethod
    def from_checkpoint(cls, model_path: str, mm: MultimodalConfig, out_dtype: torch.dtype) -> CpuVisionEncoder:
        """Load the tower and run the family's smallest image through it, so a bad load fails at startup."""
        from freetoken.mm.config import check_cpu_encoder_support
        from freetoken.mm.processor import get_mm_processor
        from freetoken.models.register import get_model_spec
        from freetoken.utils import cached_load_hf_config

        check_cpu_encoder_support(get_model_spec(cached_load_hf_config(model_path).architectures[0]).mm_processor)
        encoder = cls(
            load_cpu_vision_tower(model_path, rope_dtype=out_dtype),
            out_dtype=out_dtype,
            cache_bytes=mm.encoder_cache_mb << 20,
        )
        for item in get_mm_processor(model_path, mm).dummy_items(out_dtype, torch.device("cpu")):
            encoder.tower.forward(item.feature, [item.grid_thw])
        return encoder

    @torch.inference_mode()
    def encode_item(self, item: MMItem) -> tuple[torch.Tensor, bool]:
        """The item's embedding rows and whether they came from the cache."""
        emb = self.cache.get(item.hash)
        hit = emb is not None
        if emb is None:
            emb = self.tower.forward(item.feature, [item.grid_thw]).to(self.out_dtype).contiguous()
            self.cache.put(item.hash, emb)
        if emb.shape[0] != item.num_tokens:
            raise ValueError(f"the tower made {emb.shape[0]} rows for an image the prompt gives {item.num_tokens} tokens")
        return emb, hit

    def encode_msg(self, msg: UserMsg) -> tuple[int, int, int]:
        """Encode every image of msg in place; returns (images, cache hits, image tokens)."""
        images = hits = tokens = 0
        for item in msg.mm_items or ():
            if item.feature is None:
                continue
            emb, hit = self.encode_item(item)
            item.precomputed_embeddings = emb
            item.feature = None
            item.validate()
            images, hits, tokens = images + 1, hits + hit, tokens + emb.shape[0]
        return images, hits, tokens


def handle_backend_msg(encoder: CpuVisionEncoder, msg: Any, send_backend: Any, send_frontend: Any) -> None:
    """Encode each request of msg and forward it to the scheduler; a request whose image fails ends with an error reply."""
    from freetoken.message import BatchBackendMsg, UserMsg, UserReply

    msgs: List[Any] = msg.data if isinstance(msg, BatchBackendMsg) else [msg]
    for user_msg in msgs:
        if not isinstance(user_msg, UserMsg):
            send_backend.put(user_msg)
            continue
        t0 = time.perf_counter()
        try:
            images, hits, tokens = encoder.encode_msg(user_msg)
        except Exception as exc:  # noqa: BLE001 -- one bad image must not take the encoder down
            logger.warning(f"image encoding failed for request {user_msg.uid}: {exc!r}")
            send_frontend.put(
                UserReply(uid=user_msg.uid, incremental_output="", finished=True, error=f"could not encode image: {exc}")
            )
            continue
        logger.info(
            f"request {user_msg.uid}: {images} image(s), {tokens} tokens, {hits} cached, "
            f"{(time.perf_counter() - t0) * 1000:.0f} ms"
        )
        send_backend.put(user_msg)


def serve_vision_encoder(
    encoder: CpuVisionEncoder,
    *,
    addr: str,
    backend_addr: str,
    frontend_addr: str,
    ack_queue: Any = None,
) -> None:
    from freetoken.message import BaseBackendMsg, BaseFrontendMsg
    from freetoken.utils import ZmqPullQueue, ZmqPushQueue

    recv = ZmqPullQueue(addr, create=True, decoder=BaseBackendMsg.decoder)
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    if ack_queue is not None:
        ack_queue.put("Vision encoder is ready")
    try:
        while True:
            handle_backend_msg(encoder, recv.get(), send_backend, send_frontend)
    except KeyboardInterrupt:
        pass


__all__ = [
    "CpuVisionEncoder",
    "EmbeddingLRU",
    "configure_encoder_process",
    "default_encoder_threads",
    "handle_backend_msg",
    "iter_vision_tensors",
    "load_cpu_vision_tower",
    "serve_vision_encoder",
]
