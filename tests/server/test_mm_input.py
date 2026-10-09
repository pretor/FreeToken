"""Image input plumbing: content-part rendering, ref collection, fetch, and the gate.

No GPU and no model checkpoint: everything here is pure frontend logic."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from freetoken.mm.media import collect_image_refs, fetch_image_bytes, image_reject_reason
from freetoken.server.generation import GenerationError, render_messages
from freetoken.server.stats import derive_model_card

PNG = base64.b64encode(b"fakepng").decode()


def _config(**overrides):
    fields = dict(
        model_path="/nonexistent", allowed_media_domains="", allowed_local_media_path="",
        served_model_name="unit-model", max_seq_len=8192, model_config=SimpleNamespace(),
    )
    text_model_only = overrides.pop("text_model_only", False)
    serves_images = overrides.pop("vision_enabled", False)
    mm = SimpleNamespace(
        text_model_only=text_model_only,
        disabled_encoders=frozenset({"vision", "audio"}) if text_model_only else frozenset(),
    )
    return SimpleNamespace(mm=mm, served_modalities=frozenset({"image"}) if serves_images else frozenset(), **{**fields, **overrides})


def test_image_url_part_becomes_template_image_part():
    msgs = render_messages(
        [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]}]
    )
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image"
    refs = collect_image_refs(msgs)
    assert refs == [{"kind": "url", "data": "https://x/y.png"}]
    # refs are popped; the template-facing part stays
    assert msgs[0]["content"][1] == {"type": "image"}
    assert collect_image_refs(msgs) == []


def test_collect_refs_preserves_prompt_order():
    msgs = render_messages(
        [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u1"}}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "u2"}},
                {"type": "image_url", "image_url": {"url": "u3"}},
            ]},
        ]
    )
    assert [r["data"] for r in collect_image_refs(msgs)] == ["u1", "u2", "u3"]


def test_fetch_decodes_data_uri_and_raw_b64():
    refs = [
        {"kind": "url", "data": f"data:image/png;base64,{PNG}"},
        {"kind": "b64", "data": PNG},
    ]
    assert asyncio.run(fetch_image_bytes(refs, _config())) == [b"fakepng", b"fakepng"]


def test_fetch_failures_surface_as_generation_error(monkeypatch):
    from freetoken.server import generation as gen

    # bypass the capability gate; the fetch failure itself must become a GenerationError
    monkeypatch.setattr(gen, "image_reject_reason", lambda config: None)
    state = SimpleNamespace(config=_config())
    with pytest.raises(GenerationError):
        asyncio.run(gen._resolve_images([{"kind": "url", "data": "ftp://nope"}], state))


def test_image_gate_reasons():
    assert "text-model-only" in image_reject_reason(_config(text_model_only=True))
    assert "vision" in image_reject_reason(_config(model_path="/nonexistent"))


def test_stats_model_card_lists_the_accepted_input_modalities():
    assert derive_model_card(_config())["input_modalities"] == ["text"]
    assert derive_model_card(_config(vision_enabled=True))["input_modalities"] == ["text", "image"]


def test_media_domain_allowlist():
    from freetoken.mm.media import _check_media_domain

    config = _config(allowed_media_domains="cdn.example.com, Other.COM.")
    _check_media_domain("https://cdn.example.com/a.png", config)  # allowed: no raise
    _check_media_domain("https://OTHER.com./b.png", config)  # case/root-dot normalized
    with pytest.raises(ValueError, match="allowed domains"):
        _check_media_domain("https://evil.com/a.png", config)
    # empty allowlist admits any domain
    _check_media_domain("https://evil.com/a.png", _config())

    with pytest.raises(ValueError, match="allowed domains"):
        asyncio.run(
            fetch_image_bytes([{"kind": "url", "data": "https://evil.com/a.png"}], config)
        )


def test_local_media_requires_allowlisted_root(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"fakepng")
    url = f"file://{img}"

    # gate off (default): rejected
    with pytest.raises(ValueError, match="allowed-local-media-path"):
        asyncio.run(fetch_image_bytes([{"kind": "url", "data": url}], _config()))

    # gate on, file under the root: served
    config = _config(allowed_local_media_path=str(tmp_path))
    assert asyncio.run(fetch_image_bytes([{"kind": "url", "data": url}], config)) == [b"fakepng"]

    # a path outside the root is rejected even with the gate on
    with pytest.raises(ValueError, match="subpath"):
        asyncio.run(fetch_image_bytes([{"kind": "url", "data": "file:///etc/hostname"}], config))


def test_image_token_budget_flags_land_in_the_multimodal_config():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3VLForConditionalGeneration"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        args, _ = parse_args([
            "--model", "/models/anon", "--image-min-tokens", "64", "--image-max-tokens", "1024",
            "--mm-processor-kwargs", '{"size": {"longest_edge": 4096}}',
        ])
        assert (args.mm.image_min_tokens, args.mm.image_max_tokens) == (64, 1024)
        assert args.mm.processor_kwargs == {"size": {"longest_edge": 4096}}
        assert parse_args(["--model", "/models/anon"])[0].mm.processor_kwargs == {}
        with pytest.raises(SystemExit):  # argparse reports the bad pair and exits
            parse_args(["--model", "/models/anon", "--image-min-tokens", "2048", "--image-max-tokens", "1024"])


def _hf_with_vision(arch):
    return SimpleNamespace(
        architectures=[arch], vision_config=SimpleNamespace(),
        to_dict=lambda: {"architectures": [arch], "torch_dtype": "bfloat16"},
    )


def test_cpu_encoder_flags_land_in_the_multimodal_config():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    hf = _hf_with_vision("Qwen4ExpForConditionalGeneration")
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf), patch(
        "freetoken.engine.config.cached_load_hf_config", lambda _path, _overrides=None: hf
    ):
        cpu = ["--model", "/models/anon", "--mm-encoder-weights", "cpu"]
        mm = parse_args(cpu)[0].mm
        assert mm.encoder_out_of_process and mm.image_max_tokens == 1024  # the CPU default cap
        assert (mm.encoder_threads, mm.encoder_cpus, mm.encoder_cache_mb) == (None, None, 256)
        mm = parse_args(cpu + [
            "--image-max-tokens", "2048", "--mm-encoder-threads", "8", "--mm-encoder-cpus", "22-25",
            "--mm-encoder-cache-mb", "0",
        ])[0].mm
        assert (mm.image_max_tokens, mm.encoder_threads, mm.encoder_cpus, mm.encoder_cache_mb) == (
            2048, 8, (22, 23, 24, 25), 0,
        )
        # the default cap never undercuts an explicit floor; the other placements keep the checkpoint limits
        assert parse_args(cpu + ["--image-min-tokens", "1500"])[0].mm.image_max_tokens == 1500
        assert parse_args(["--model", "/models/anon"])[0].mm.image_max_tokens is None
        with pytest.raises(SystemExit):
            parse_args(cpu + ["--mm-encoder-cpus", "3-1"])


def test_cpu_encoder_refuses_a_family_without_the_qwen_vl_tower():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    hf = _hf_with_vision("Gemma4ForConditionalGeneration")
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf), patch(
        "freetoken.engine.config.cached_load_hf_config", lambda _path, _overrides=None: hf
    ):
        with pytest.raises(SystemExit):
            parse_args(["--model", "/models/anon", "--mm-encoder-weights", "cpu"])
        assert parse_args(["--model", "/models/anon", "--mm-encoder-weights", "host"])[0].mm.encoder_weights == "host"
