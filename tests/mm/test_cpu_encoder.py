"""The CPU vision encoder process (--mm-encoder-weights cpu): requests leave it with embedding rows instead of pixels."""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import AbortBackendMsg, BaseBackendMsg, BatchBackendMsg, MMItem, UserMsg, UserReply
from freetoken.mm.config import parse_cpu_list
from freetoken.mm.cpu_encoder import CpuVisionEncoder, EmbeddingLRU, default_encoder_threads, handle_backend_msg

D = 8


class _Tower:
    """A 2x2-merging stand-in: h * w / 4 rows, each holding the first pixel so outputs track their input."""

    def __init__(self):
        self.calls = 0

    def forward(self, feature, grid_thw):
        self.calls += 1
        _, h, w = grid_thw[0]
        return torch.full((h * w // 4, D), float(feature[0, 0]))


class _Queue(list):
    put = list.append


def _item(h, value, n_tokens=4):
    return MMItem(
        modality="image", hash=h, pad_value=h, offsets=[[3, 3 + n_tokens]],
        feature=torch.full((16, 6), float(value)), model_specific_data={"grid_thw": [1, 4, 4]},
    )


def _msg(uid, *items):
    return UserMsg(
        uid=uid, input_ids=torch.zeros(12, dtype=torch.int32), sampling_params=SamplingParams(max_tokens=1),
        mm_items=list(items),
    )


def _encoder(cache_bytes=1 << 20):
    return CpuVisionEncoder(_Tower(), out_dtype=torch.bfloat16, cache_bytes=cache_bytes)


def test_items_leave_with_embeddings_instead_of_pixels():
    enc = _encoder()
    msg = _msg(1, _item(7, 3.0))
    assert enc.encode_msg(msg) == (1, 0, 4)
    (item,) = msg.mm_items
    assert item.feature is None
    assert item.precomputed_embeddings.dtype == torch.bfloat16
    assert item.precomputed_embeddings.shape == (4, D)
    assert torch.all(item.precomputed_embeddings == 3.0)
    item.validate()


def test_a_resent_image_comes_from_the_cache():
    enc = _encoder()
    enc.encode_msg(_msg(1, _item(7, 3.0)))
    msg = _msg(2, _item(7, 3.0))
    assert enc.encode_msg(msg) == (1, 1, 4)
    assert enc.tower.calls == 1
    assert torch.all(msg.mm_items[0].precomputed_embeddings == 3.0)


def test_a_disabled_cache_encodes_every_time():
    enc = _encoder(cache_bytes=0)
    enc.encode_msg(_msg(1, _item(7, 3.0)))
    enc.encode_msg(_msg(2, _item(7, 3.0)))
    assert enc.tower.calls == 2


def test_rows_the_prompt_does_not_expect_are_refused():
    with pytest.raises(ValueError, match="rows"):
        _encoder().encode_msg(_msg(1, _item(7, 3.0, n_tokens=5)))


def test_lru_evicts_the_least_recently_used_entry_by_bytes():
    rows = torch.zeros(4, D, dtype=torch.bfloat16)  # 64 bytes
    lru = EmbeddingLRU(max_bytes=128)
    lru.put(1, rows)
    lru.put(2, rows.clone())
    assert lru.get(1) is not None  # 1 is now the most recent
    lru.put(3, rows.clone())
    assert lru.get(2) is None and lru.get(1) is not None and lru.get(3) is not None
    assert lru.nbytes == 128
    lru.put(4, torch.zeros(100, D, dtype=torch.bfloat16))  # bigger than the whole cache: not kept
    assert lru.get(4) is None and len(lru) == 2


def test_encoded_requests_go_to_the_scheduler_and_failures_to_the_client():
    backend, frontend = _Queue(), _Queue()
    good, bad = _msg(1, _item(7, 3.0)), _msg(2, _item(8, 1.0, n_tokens=9))
    handle_backend_msg(_encoder(), BatchBackendMsg(data=[good, bad]), backend, frontend)
    assert backend == [good] and good.mm_items[0].precomputed_embeddings is not None
    (reply,) = frontend
    assert isinstance(reply, UserReply) and reply.uid == 2 and reply.finished
    assert "could not encode image" in reply.error


def test_an_encoded_request_survives_the_wire():
    # the rows come out of inference mode and are serialized outside it, on the way to the scheduler
    msg = _msg(1, _item(7, 3.0))
    _encoder().encode_msg(msg)
    back = BaseBackendMsg.decoder(BaseBackendMsg.encoder(msg))
    (item,) = back.mm_items
    assert item.feature is None and item.grid_thw == [1, 4, 4]
    assert item.precomputed_embeddings.dtype == torch.bfloat16
    assert torch.equal(item.precomputed_embeddings, msg.mm_items[0].precomputed_embeddings)
    item.validate()


def test_other_messages_pass_through_untouched():
    backend = _Queue()
    abort = AbortBackendMsg(uid=3)
    handle_backend_msg(_encoder(), abort, backend, _Queue())
    assert backend == [abort]


def test_cpu_lists_parse_like_numactl():
    assert parse_cpu_list("22-25") == (22, 23, 24, 25)
    assert parse_cpu_list("0-1,8, 4") == (0, 1, 4, 8)
    for bad in ("", "3-1", "-2", "a-b"):
        with pytest.raises(ValueError):
            parse_cpu_list(bad)


def test_default_threads_is_a_positive_core_count():
    assert 1 <= default_encoder_threads() <= (os.cpu_count() or 1)
