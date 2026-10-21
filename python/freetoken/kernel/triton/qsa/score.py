# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM (vllm/models/qwen4_exp/nvidia/ops/qsa.py)
"""QSA block scoring over the paged compressed-key slab."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    MAX_N: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    # Top-k is bounded by visible_blocks, so columns beyond it need no value.
    if tile_start * BLOCK_N >= visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    # Pad the small head axis to a tensor-core-compatible N dimension.
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(heads[None, :] < NUM_HEADS) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(scores, 0.0), 0.0)
        score = tl.sum(scores, axis=1) / score_divisor
        tl.store(
            logits_ptr + row * stride_logits_row + columns,
            tl.where(page_valid, score, -float("inf")),
            mask=live & (columns < num_columns),
        )


@triton.jit
def _qsa_mqa_paged_rows_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    tile_map_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_tiles,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    """Prefill scoring, ROWS query rows x 4 heads per program.

    The rows of a tile all belong to one request (``tile_map`` holds each tile's first row
    and row count, cut at request boundaries), so one key tile serves every row. Each head
    is its own ``[BLOCK_N, ROWS]`` dot and the head sum keeps the one-row kernel's order
    ``(h0 + h1) + (h2 + h3)``: every score is bitwise the one-row kernel's.
    """
    tile = tl.program_id(0)
    row0 = tl.load(tile_map_ptr + tile)
    nrows = tl.load(tile_map_ptr + num_tiles + tile)
    dims = tl.arange(0, BLOCK_D)
    lanes = tl.arange(0, ROWS)
    row_live = lanes < nrows
    rows = row0 + lanes
    request = tl.load(token_to_req_ptr + row0)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + rows, mask=row_live, other=0)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    visible = tl.where(row_live, visible, 0)
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + rows, visible, mask=row_live)
    visible_max = tl.max(visible, axis=0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    # Top-k is bounded by visible_blocks, so columns beyond it need no value.
    if tile_start * BLOCK_N >= visible_max:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(visible_max, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    q_rows = q_ptr + rows[None, :].to(tl.int64) * stride_q_row + dims[:, None] * stride_q_dim
    q_mask = row_live[None, :] & (dims[:, None] < HEAD_DIM)
    query0 = tl.load(q_rows + 0 * stride_q_head, mask=q_mask, other=0.0)
    query1 = tl.load(q_rows + 1 * stride_q_head, mask=q_mask, other=0.0)
    query2 = tl.load(q_rows + 2 * stride_q_head, mask=q_mask, other=0.0)
    query3 = tl.load(q_rows + 3 * stride_q_head, mask=q_mask, other=0.0)
    column_offsets = tl.arange(0, BLOCK_N)
    out_rows = logits_ptr + rows[None, :].to(tl.int64) * stride_logits_row
    for col_tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = col_tile * BLOCK_N + column_offsets
        live = columns < visible_max
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        s0 = tl.maximum(tl.dot(keys, query0, out_dtype=tl.float32), 0.0)
        s1 = tl.maximum(tl.dot(keys, query1, out_dtype=tl.float32), 0.0)
        s2 = tl.maximum(tl.dot(keys, query2, out_dtype=tl.float32), 0.0)
        s3 = tl.maximum(tl.dot(keys, query3, out_dtype=tl.float32), 0.0)
        score = ((s0 + s1) + (s2 + s3)) / score_divisor
        tl.store(
            out_rows + columns[:, None],
            tl.where(page_valid[:, None], score, -float("inf")),
            mask=(columns[:, None] < visible[None, :]) & (columns[:, None] < num_columns),
        )


def qsa_score_tile_map(
    qo_indptr: torch.Tensor, start: int, end: int, rows_per_tile: int, device: torch.device
) -> torch.Tensor:
    """``[2, tiles]`` int32 (first row, row count) of rows ``[start, end)``, relative to
    ``start``, in tiles of at most ``rows_per_tile`` rows that never cross a request boundary
    of the host-side ``qo_indptr`` (cu_seqlens)."""

    bounds = [int(x) for x in qo_indptr.tolist()]
    firsts: list[int] = []
    counts: list[int] = []
    for lo, hi in zip(bounds, bounds[1:]):
        lo, hi = max(lo, start), min(hi, end)
        for row in range(lo, hi, rows_per_tile):
            firsts.append(row - start)
            counts.append(min(rows_per_tile, hi - row))
    host = torch.tensor([firsts, counts], dtype=torch.int32)
    if device.type == "cuda":
        host = host.pin_memory()
    return host.to(device, non_blocking=True)


SCORE_ROWS = 16


def qsa_mqa_paged_rows(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    tile_map: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    score_scale: float | None = None,
    rows_per_tile: int = SCORE_ROWS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill form of :func:`qsa_mqa_paged`: the same scores bit for bit, with
    ``rows_per_tile`` query rows of one request per program instead of one row padded to 16
    heads. ``tile_map`` comes from :func:`qsa_score_tile_map` over the same rows and must use
    the same ``rows_per_tile``. Needs exactly 4 index heads."""

    if q.ndim != 3 or q.shape[1] != 4 or q.shape[2] <= 0:
        raise ValueError("QSA row-tiled scoring needs a [rows, 4, head_dim] query")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1 or k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim] matching the query")
    if token_to_req.shape != (q.shape[0],) or query_positions.shape != (q.shape[0],):
        raise ValueError("QSA request mapping and positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    if q.dtype not in (torch.bfloat16, torch.float16) or k_cache.dtype is not q.dtype:
        raise ValueError(
            f"QSA scoring is 16-bit only, got query={q.dtype} keys={k_cache.dtype}"
        )
    if tile_map.ndim != 2 or tile_map.shape[0] != 2 or tile_map.dtype != torch.int32:
        raise ValueError("QSA tile map must be [2, tiles] int32")
    if rows_per_tile < 16 or rows_per_tile & (rows_per_tile - 1):
        raise ValueError("rows_per_tile must be a power of two >= 16 (the dot's N)")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    columns = logits.shape[1]
    tiles = tile_map.shape[1]
    if not q.shape[0] or not columns or not tiles:
        return logits, visible_blocks
    BLOCK_N = 64
    BLOCK_D = max(16, triton.next_power_of_2(q.shape[2]))
    tiles_per_program = 8
    _qsa_mqa_paged_rows_kernel[(tiles, triton.cdiv(columns, BLOCK_N * tiles_per_program))](
        q,
        k_cache,
        page_table,
        tile_map,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        tiles,
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        ROWS=rows_per_tile,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
    )
    return logits, visible_blocks


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA scores directly from a paged compressed-key cache."""

    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if token_to_req.shape != (q.shape[0],) or query_positions.shape != (q.shape[0],):
        raise ValueError("QSA request mapping and positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    if q.dtype not in (torch.bfloat16, torch.float16) or k_cache.dtype is not q.dtype:
        # The dot below is a plain 16-bit matmul: an fp8 operand is not a slow path, it
        # is a triton compile error that surfaces mid-CUDA-graph-capture. The KV cache
        # may well be e4m3 codes (--kv-cache-dtype fp8) -- what must never reach here
        # are those codes; the compressed index keys are their own 16-bit tier.
        raise ValueError(
            f"QSA scoring is 16-bit only, got query={q.dtype} keys={k_cache.dtype}"
        )
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    columns = logits.shape[1]
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    BLOCK_N = 64
    BLOCK_D = max(16, triton.next_power_of_2(q.shape[2]))
    MAX_N = max(16, triton.next_power_of_2(q.shape[1]))
    # Tuned on GB300: larger row batches provide enough parallelism to reuse Q.
    tiles_per_program = 1 if q.shape[0] <= 32 else 8
    _qsa_mqa_paged_kernel[
        (q.shape[0], triton.cdiv(columns, BLOCK_N * tiles_per_program))
    ](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
        num_warps=2,
    )
    return logits, visible_blocks


__all__ = ["qsa_mqa_paged"]
