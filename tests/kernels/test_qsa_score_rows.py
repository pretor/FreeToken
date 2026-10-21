"""The prefill row-tiled QSA score kernel (``qsa_mqa_paged_rows``) against the one-row kernel.

Both must give the same visible counts, bitwise-equal logits on every visible column, and so the
same top-k block selection, for ragged multi-request prefill chunks whose request starts are not
aligned to the row tile, and for a row slice in the middle of a forward (the ``_select`` chunking).
"""

from __future__ import annotations

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

CMP_PAGE = 16
RATIO = 4
HEADS = 4
DIM = 128
TOPK = 512


def _case(lengths, rows_per_req, seed, with_invalid=False):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    max_blocks = max(lengths) // RATIO
    width = -(-max_blocks // CMP_PAGE)
    bs = len(lengths)
    total = bs * width + 3
    block_table = torch.randperm(total, device=device, generator=gen)[: bs * width]
    block_table = block_table.reshape(bs, width).to(torch.int32)
    if with_invalid:
        block_table[0, 1] = -1
        block_table[-1, 0] = total + 5
    cache = torch.randn(total, CMP_PAGE, 1, DIM, device=device, dtype=torch.bfloat16, generator=gen)
    rows = sum(rows_per_req)
    q = torch.randn(rows, HEADS, DIM, device=device, dtype=torch.bfloat16, generator=gen)
    token_to_req = torch.repeat_interleave(
        torch.arange(bs, dtype=torch.int32), torch.tensor(rows_per_req)
    ).to(device)
    positions = torch.cat(
        [
            torch.arange(n - r, n, dtype=torch.int32, device=device)
            for n, r in zip(lengths, rows_per_req)
        ]
    )
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0] + list(rows_per_req), dtype=torch.int32).cumsum(0).to(torch.int32)
    return q, cache, block_table, token_to_req, positions, seq_lens, qo_indptr, width * CMP_PAGE


def _run_both(case, start, end, rows_per_tile):
    from freetoken.kernel.triton.qsa import (
        qsa_block_topk,
        qsa_mqa_paged,
        qsa_mqa_paged_rows,
        qsa_score_tile_map,
    )

    q, cache, table, t2r, pos, seq, qo, columns = case
    sl = slice(start, end)
    n = end - start
    out = []
    for rows_kernel in (False, True):
        logits = torch.full((n, columns), float("nan"), dtype=torch.float32, device=q.device)
        visible = torch.full((n,), -7, dtype=torch.int32, device=q.device)
        if rows_kernel:
            tiles = qsa_score_tile_map(qo, start, end, rows_per_tile, q.device)
            qsa_mqa_paged_rows(
                q[sl], cache, table, tiles, t2r[sl], pos[sl], seq, RATIO, logits, visible,
                rows_per_tile=rows_per_tile,
            )
        else:
            qsa_mqa_paged(q[sl], cache, table, t2r[sl], pos[sl], seq, RATIO, logits, visible)
        blocks = torch.empty((n, TOPK), dtype=torch.int32, device=q.device)
        qsa_block_topk(logits.clone(), visible, blocks)
        out.append((logits, visible, blocks))
    return out


def _assert_same(old, new):
    (lo, vo, bo), (ln, vn, bn) = old, new
    assert torch.equal(vo, vn)
    col = torch.arange(lo.shape[1], device=lo.device)
    live = col[None, :] < vo[:, None].to(torch.int64)
    assert live.any()
    a, b = lo[live], ln[live]
    assert not torch.isnan(a).any() and not torch.isnan(b).any()
    # bitwise, including the sign of zero and -inf on invalid pages
    assert torch.equal(a.view(torch.int32), b.view(torch.int32)), (
        f"{(a.view(torch.int32) != b.view(torch.int32)).sum().item()} of {a.numel()} scores differ"
    )
    assert torch.equal(bo, bn)


@requires_cuda
@pytest.mark.parametrize("rows_per_tile", [16, 32])
@pytest.mark.parametrize(
    "lengths, rows_per_req",
    [
        ([9000], [517]),
        ([20000, 3001], [37, 300]),
        ([700, 12345, 64, 5000], [5, 210, 33, 17]),
        ([4096], [4096]),
    ],
)
def test_rows_kernel_is_bitwise_the_one_row_kernel(lengths, rows_per_req, rows_per_tile):
    case = _case(lengths, rows_per_req, seed=sum(lengths) + rows_per_tile)
    rows = sum(rows_per_req)
    old, new = _run_both(case, 0, rows, rows_per_tile)
    _assert_same(old, new)


@requires_cuda
@pytest.mark.parametrize("start, end", [(100, 612), (0, 512), (517, 818)])
def test_rows_kernel_on_a_row_slice(start, end):
    case = _case([30000, 2000, 9000], [600, 101, 117], seed=start + end, with_invalid=True)
    old, new = _run_both(case, start, end, 16)
    _assert_same(old, new)


@requires_cuda
def test_rows_kernel_at_long_context():
    """A late 249K-context chunk: ~62K visible blocks per row, 512-row slice."""
    case = _case([249000], [512], seed=3)
    old, new = _run_both(case, 0, 512, 16)
    _assert_same(old, new)


def test_tile_map_never_crosses_a_request():
    from freetoken.kernel.triton.qsa import qsa_score_tile_map

    qo = torch.tensor([0, 5, 40, 41, 100], dtype=torch.int32)
    tiles = qsa_score_tile_map(qo, 3, 90, 16, torch.device("cpu"))
    firsts, counts = tiles.tolist()
    assert firsts == [0, 2, 18, 34, 37, 38, 54, 70, 86]
    assert counts == [2, 16, 16, 3, 1, 16, 16, 16, 1]
