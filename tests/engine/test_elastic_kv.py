"""Elastic KV: pin the pool at X, grow it toward Y, and give the freed bytes back to MoE."""

from __future__ import annotations

import pytest

from freetoken.engine.cache_budget import (
    ElasticKv,
    elastic_grow_tokens,
    max_elastic_tokens,
    plan_elastic_kv,
)


def _plan(**kwargs):
    base = dict(
        budget_bytes=2_000,
        per_expert_bytes=100,
        cache_per_page=10,
        page_size=10,
        num_experts=4,
        total_experts=50,
        prefill_overlap=True,
        max_slots=50,
        moe_floor=8,
    )
    base.update(kwargs)
    return plan_elastic_kv(**base)


def test_initial_pool_is_the_requested_size_and_moe_takes_the_rest():
    moe, pages, overlap = _plan(kv_tokens=32)
    assert pages == 4  # 32 tokens, page 10 -> 4 pages (40 tokens)
    assert moe == (2_000 - 4 * 10) // 100  # 19
    assert overlap is True
    assert moe * 100 + pages * 10 <= 2_000


def test_growing_kv_shrinks_moe_and_does_not_exceed_the_budget():
    small_moe, small_pages, _ = _plan(kv_tokens=32)
    big_moe, big_pages, _ = _plan(kv_tokens=1_000)
    assert big_pages > small_pages
    assert big_moe < small_moe
    assert big_moe >= 8
    assert big_moe * 100 + big_pages * 10 <= 2_000


def test_full_expert_cache_does_not_inflate_the_kv_pool():
    # Experts hit the cap; leftover bytes stay free instead of becoming extra KV pages.
    moe, pages, _ = _plan(kv_tokens=20, max_slots=8, total_experts=8, moe_floor=8)
    assert moe == 8
    assert pages == 2


def test_kv_pin_below_the_moe_floor_raises():
    with pytest.raises(ValueError, match="below the floor"):
        _plan(kv_tokens=1_900, moe_floor=8)


def test_max_tokens_keeps_the_moe_floor():
    fit = max_elastic_tokens(
        budget_bytes=2_000,
        per_expert_bytes=100,
        cache_per_page=10,
        page_size=10,
        moe_floor=8,
    )
    # 2000 - 8*100 = 1200 bytes -> 120 pages -> 1200 tokens.
    assert fit == 1_200
    moe, pages, _ = _plan(kv_tokens=fit)
    assert pages * 10 == fit
    assert moe == 8


def test_elastic_state_replans_the_same_budget():
    ek = ElasticKv(
        budget_bytes=2_000,
        per_expert_bytes=100,
        cache_per_page=10,
        page_size=10,
        num_experts=4,
        total_experts=50,
        max_slots=50,
        prefill_overlap=True,
        moe_floor=8,
        initial_tokens=40,
        max_tokens=1_200,
    )
    moe, pages, _ = ek.plan(40)
    grown, grown_pages, _ = ek.plan(ek.max_tokens)
    assert pages * ek.page_size == 40
    assert grown < moe
    assert grown_pages * ek.page_size == ek.max_tokens


def test_grow_covers_several_requests_until_the_ceiling():
    # 40k + 40k fits under 100k; the pool grows once to their sum.
    assert elastic_grow_tokens(32_000, [40_000, 40_000], 100_000) == 80_000
    # The second 200k does not fit beside the first under 256k, so the pool grows to the first.
    assert elastic_grow_tokens(32_000, [200_000, 200_000], 262_144) == 200_000
    # Already covered: no rebuild.
    assert elastic_grow_tokens(80_000, [40_000, 40_000], 100_000) is None
    # A request above the ceiling is ignored; the one that fits still grows the pool.
    assert elastic_grow_tokens(32_000, [300_000, 40_000], 100_000) == 40_000
    assert elastic_grow_tokens(32_000, [], 100_000) is None
