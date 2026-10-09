"""CPU vision encoder (--mm-encoder-weights cpu): per-image latency, RSS and FP32-vs-BF16 drift of the Qwen VL tower on the host.

No GPU is touched (CUDA is hidden before torch loads), so it can run beside a serving engine.

    python benchmarks/bench_cpu_vision.py --model /path/to/ckpt [--tokens 256,512,1024,2048] [--threads 22]
    python benchmarks/bench_cpu_vision.py --model /path/to/ckpt --loop 1024   # encode forever, for interference runs

Pin it the way the server would: numactl --cpunodebind=1 --preferred=1 python ... --threads 22
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import resource
import statistics
import time


def _rss_gib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _test_image(path: str | None):
    """The given picture, or a deterministic 2048x2048 one with gradients, shapes and text."""
    import random

    from PIL import Image, ImageDraw

    if path:
        return Image.open(path).convert("RGB")
    rng = random.Random(0)
    img = Image.linear_gradient("L").resize((2048, 2048)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(200):
        x, y = rng.randrange(2048), rng.randrange(2048)
        color = tuple(rng.randrange(256) for _ in range(3))
        draw.rectangle((x, y, x + rng.randrange(8, 300), y + rng.randrange(8, 300)), fill=color)
    for row in range(0, 2048, 64):
        draw.text((16, row), "The quick brown fox jumps over the lazy dog 0123456789 " * 3, fill=(0, 0, 0))
    return img


def _item(model: str, pil, tokens: int):
    from freetoken.mm.config import MultimodalConfig
    from freetoken.mm.processor import get_mm_processor

    proc = get_mm_processor(model, MultimodalConfig(image_min_tokens=tokens, image_max_tokens=tokens))
    return proc.process([pil])[0]


def _time(tower, item, runs: int) -> tuple[float, object]:
    out = tower.forward(item.feature, [item.grid_thw])  # first call creates the oneDNN primitives for the shapes
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = tower.forward(item.feature, [item.grid_thw])
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def main() -> None:
    from freetoken.mm.cpu_encoder import default_encoder_threads

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tokens", default="256,512,1024,2048", help="image token budgets to time")
    p.add_argument("--threads", type=int, default=default_encoder_threads())
    p.add_argument("--runs", type=int, default=3, help="timed runs per size after one warm-up (median reported)")
    p.add_argument("--bf16-max-tokens", type=int, default=1024, help="also time a BF16 tower up to this size; 0 skips")
    p.add_argument("--image", help="picture to encode (default: a synthetic 2048x2048 one)")
    p.add_argument("--loop", type=int, default=0, metavar="TOKENS", help="encode at this budget until killed")
    p.add_argument("--json", help="write the results here")
    args = p.parse_args()

    import torch
    import torch.nn.functional as F

    from freetoken.mm.cpu_encoder import load_cpu_vision_tower

    torch.set_num_threads(args.threads)
    pil = _test_image(args.image)
    t0 = time.perf_counter()
    tower = load_cpu_vision_tower(args.model, dtype=torch.float32)
    load_s = time.perf_counter() - t0
    print(f"threads {args.threads}  load {load_s:.1f} s  RSS {_rss_gib():.2f} GiB  ({torch.__version__}, "
          f"mkldnn={torch.backends.mkldnn.is_available()})", flush=True)

    if args.loop:
        item = _item(args.model, pil, args.loop)
        tower.forward(item.feature, [item.grid_thw])
        while True:
            t0 = time.perf_counter()
            out = tower.forward(item.feature, [item.grid_thw])
            print(f"{time.strftime('%H:%M:%S')}  {out.shape[0]} tokens  "
                  f"{(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)

    bf16 = None
    if args.bf16_max_tokens:
        bf16 = load_cpu_vision_tower(args.model, dtype=torch.bfloat16)
    results = []
    for budget in (int(t) for t in args.tokens.split(",")):
        item = _item(args.model, pil, budget)
        _, h, w = item.grid_thw
        fp32_s, ref = _time(tower, item, args.runs)
        row = {"budget": budget, "tokens": ref.shape[0], "patch_grid": [h, w], "fp32_s": round(fp32_s, 3),
               "peak_rss_gib": round(_peak_rss_gib(), 2)}
        if bf16 is not None and budget <= args.bf16_max_tokens:
            bf16_s, out = _time(bf16, item, max(1, args.runs - 1))
            cos = F.cosine_similarity(ref.float(), out.float(), dim=-1)
            rel = ((ref.float() - out.float()).norm() / ref.float().norm()).item()
            row.update(bf16_s=round(bf16_s, 3), bf16_cos_min=round(cos.min().item(), 5),
                       bf16_cos_mean=round(cos.mean().item(), 5), bf16_rel_err=round(rel, 5))
        results.append(row)
        print(json.dumps(row), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"threads": args.threads, "load_s": round(load_s, 1), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
