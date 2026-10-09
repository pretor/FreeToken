"""Qwen3VLVisionModel on the CPU in FP32, as the --mm-encoder-weights cpu process runs it (no CUDA, no Triton).

Separate from test_qwen3_vl_vision.py, which skips every test without CUDA.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.qwen3_vl import Qwen3VLVisionModel, VisionConfig
from freetoken.models.qwen3_vl.vision import _apply_vision_rope
from freetoken.utils import torch_dtype

GRID = [[1, 8, 8]]  # 64 patches -> 16 merged tokens


@pytest.fixture(autouse=True)
def _single_rank():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _vc(deepstack=()):
    return VisionConfig(
        hidden_size=64, depth=3, num_heads=4, intermediate_size=128, patch_size=16, temporal_patch_size=2,
        spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=32, in_channels=3,
        deepstack_visual_indexes=deepstack,
    )


def _tower(vc, device="cpu", dtype=torch.float32, rope_dtype=None):
    """A tower with the same random weights for every device and dtype (seeded, generated on the CPU)."""
    with torch.device(device), torch_dtype(dtype):
        tower = Qwen3VLVisionModel(vc, rope_dtype=rope_dtype)
    gen = torch.Generator().manual_seed(0)
    for p in tower.state_dict().values():
        p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
    return tower


def _pixels():
    return torch.randn(64, 3 * 2 * 16 * 16, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("deepstack", [(), (0, 1)])
def test_cpu_forward_matches_the_naive_forward(deepstack):
    tower = _tower(_vc(deepstack))
    out = tower.forward(_pixels(), GRID)
    assert out.shape == (16, 32 * (1 + len(deepstack))) and out.dtype == torch.float32
    torch.testing.assert_close(out, tower.forward_naive(_pixels(), GRID))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_eager_rope_rotates_q_and_k_in_place_and_leaves_v(dtype):
    S, H, D = 64, 4, 16
    gen = torch.Generator().manual_seed(2)
    qkv = torch.randn(S, 3 * H * D, generator=gen).to(dtype)
    freqs = torch.randn(S, D // 2, generator=gen) * 10
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    # strided views into the fused projection, as VisionAttention.attend passes them
    q, k, v = qkv.view(S, 3, H, D).unbind(1)
    v_before = v.clone()

    def rotated(x):  # the rotate-half formula, written the plain way
        x = x.float()
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)[:, None]
        sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)[:, None]
        return (x * cos + torch.cat((-x[..., D // 2 :], x[..., : D // 2]), dim=-1) * sin).to(dtype)

    want_q, want_k = rotated(q), rotated(k)
    _apply_vision_rope(q, k, cache, torch.arange(S, dtype=torch.int32), D)
    tol = {} if dtype == torch.float32 else {"atol": 2e-2, "rtol": 1e-2}  # one bf16 ulp of rounding freedom
    torch.testing.assert_close(q, want_q, **tol)
    torch.testing.assert_close(k, want_k, **tol)
    assert torch.equal(v, v_before)


def test_an_fp32_tower_can_keep_the_bf16_rope_rounding():
    grid, cpu = torch.tensor(GRID), torch.device("cpu")
    bf16_rows, _ = _tower(_vc(), dtype=torch.bfloat16)._rope_table(grid, cpu)
    fp32_rows, _ = _tower(_vc())._rope_table(grid, cpu)
    pinned_rows, _ = _tower(_vc(), rope_dtype=torch.bfloat16)._rope_table(grid, cpu)
    assert torch.equal(pinned_rows, bf16_rows)
    assert not torch.equal(fp32_rows, bf16_rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_fp32_tower_tracks_the_gpu_bf16_tower():
    vc = _vc()
    cpu_out = _tower(vc, rope_dtype=torch.bfloat16).forward(_pixels(), GRID)
    gpu_tower = _tower(vc, device="cuda", dtype=torch.bfloat16)
    gpu_out = gpu_tower.forward(_pixels().to("cuda", torch.bfloat16), GRID).float().cpu()
    assert F.cosine_similarity(cpu_out, gpu_out, dim=-1).mean() > 0.99
