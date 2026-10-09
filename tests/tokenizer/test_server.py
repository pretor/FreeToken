"""Tokenizer worker routing: with the CPU vision encoder, image requests detour through it and nothing else does."""

from __future__ import annotations

import torch

from freetoken.core import SamplingParams
from freetoken.message import BatchBackendMsg, MMItem, UserMsg
from freetoken.tokenizer.server import _route_user_msgs


class _Queue(list):
    put = list.append


def _msg(uid, image=False):
    items = [MMItem(modality="image", hash=1, pad_value=1, offsets=[[0, 4]], feature=torch.zeros(1))] if image else None
    return UserMsg(
        uid=uid, input_ids=torch.zeros(8, dtype=torch.int32), sampling_params=SamplingParams(max_tokens=1),
        mm_items=items,
    )


def test_without_the_cpu_encoder_every_request_goes_to_the_scheduler():
    backend = _Queue()
    msgs = [_msg(1), _msg(2, image=True)]
    _route_user_msgs(backend, None, msgs)
    (batch,) = backend
    assert isinstance(batch, BatchBackendMsg) and batch.data == msgs


def test_image_requests_detour_through_the_cpu_encoder():
    backend, encoder = _Queue(), _Queue()
    text, image = _msg(1), _msg(2, image=True)
    _route_user_msgs(backend, encoder, [text, image])
    assert backend == [text] and encoder == [image]


def test_a_text_only_drain_never_touches_the_encoder():
    backend, encoder = _Queue(), _Queue()
    _route_user_msgs(backend, encoder, [_msg(1), _msg(2)])
    assert encoder == [] and isinstance(backend[0], BatchBackendMsg) and len(backend[0].data) == 2
