import time
import requests
import json
import uuid
from transformers import AutoTokenizer

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Next-NVFP4-QAD"
TOKENIZER_PATH = "/models/local-inference-lab-Qwen3.8-Flash-Next-NVFP4-FTW"

print("Inicijalizacija tokenizera...")
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

SAMPLE_PARAGRAPHS = [
    "In deep learning architectures, mixture of experts divides neural networks into specialized subnetworks. "
    "A gating or routing network assigns tokens to the top-k relevant experts dynamically during the forward pass. "
    "Tensor parallelism shards weight matrices across multiple graphic processing units to scale memory bandwidth. ",
    "Quantum mechanics underpins emerging computing paradigms utilizing cryogenic processors and superconducting qubits. "
    "Entanglement and superposition enable simultaneous evaluation of exponential state spaces compared to classical systems. ",
    "Modern GPU clusters employ high-speed interconnects like NVLink and PCIe fabrics to minimize all-reduce latency. "
    "Activation checkpointing, kernel fusion, and FP8 precision arithmetic maximize throughput during deep transformer training. ",
    "The thermodynamics of computation dictates limits on energy dissipation. The Landauer limit states that irreversible "
    "erasure of information dissipates thermal entropy proportional to temperature. "
]

def build_prompt_exact_tokens(target_tokens):
    salt = f"TestRun-{uuid.uuid4().hex[:8]}: "
    text = salt
    idx = 0
    while True:
        text += SAMPLE_PARAGRAPHS[idx % len(SAMPLE_PARAGRAPHS)]
        enc = tok.encode(text)
        if len(enc) >= target_tokens:
            return tok.decode(enc[:target_tokens])
        idx += 1

def run_test(name, target_prompt_tokens, max_decode_tokens):
    print(f"\n{'=' * 70}")
    print(f"POKREĆEM: {name} (Cilj: {target_prompt_tokens} prompt tok, {max_decode_tokens} decode tok)")
    print(f"{'=' * 70}")
    
    prompt = build_prompt_exact_tokens(target_prompt_tokens)
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt + "\n\nSumiraj ključne teze."}
        ],
        "max_tokens": max_decode_tokens,
        "temperature": 0.6,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    t0 = time.time()
    try:
        resp = requests.post(URL, json=payload, stream=True, timeout=600)
    except Exception as e:
        print(f"Greška pri povezivanju: {e}")
        return None
        
    if resp.status_code != 200:
        print(f"Greška {resp.status_code}: {resp.text}")
        return None

    first_token_time = None
    chunks = 0
    reasoning_tokens = 0
    content_tokens = 0
    reasoning_text = ""
    content_text = ""
    usage_data = None
    
    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            data_str = line_str[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                if "usage" in data and data["usage"]:
                    usage_data = data["usage"]
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    token_text = delta.get("reasoning_content") or delta.get("content")
                    if token_text:
                        if first_token_time is None:
                            first_token_time = time.time()
                        if delta.get("reasoning_content"):
                            reasoning_text += delta["reasoning_content"]
                            reasoning_tokens += 1
                            chunks += 1
                        if delta.get("content"):
                            content_text += delta["content"]
                            content_tokens += 1
                            chunks += 1
            except Exception:
                pass
                
    t_end = time.time()
    total_time = t_end - t0
    ttft = (first_token_time - t0) if first_token_time else total_time
    decode_time = t_end - (first_token_time if first_token_time else t_end)
    decode_tps = chunks / decode_time if decode_time > 0 else 0
    
    prompt_tokens = usage_data.get("prompt_tokens", target_prompt_tokens) if usage_data else target_prompt_tokens
    prefill_tps = prompt_tokens / ttft if ttft > 0 else 0

    print(f"Prompt tokena (stvarni):   {prompt_tokens}")
    print(f"Generisano tokena:        {chunks} (Reasoning: {reasoning_tokens}, Odgovor: {content_tokens})")
    print(f"TTFT (Prefill vreme):     {ttft:.3f} s")
    print(f"Prefill brzina:           {prefill_tps:.1f} tokens/s")
    print(f"Decode vreme:             {decode_time:.3f} s")
    print(f"Decode brzina:            {decode_tps:.2f} tokens/s")
    print(f"Ukupna latencija:         {total_time:.3f} s")
    
    if reasoning_text:
        preview = reasoning_text.strip()[:100].replace('\n', ' ')
        print(f"Preview (thinking):       {preview}...")
    elif content_text:
        preview = content_text.strip()[:100].replace('\n', ' ')
        print(f"Preview (odgovor):        {preview}...")
        
    return {
        "name": name,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": chunks,
        "ttft_s": round(ttft, 3),
        "prefill_tps": round(prefill_tps, 1),
        "decode_time_s": round(decode_time, 3),
        "decode_tps": round(decode_tps, 2),
        "total_time_s": round(total_time, 3)
    }

def main():
    test_cases = [
        ("1K Prefill", 1024, 32),
        ("2K Prefill", 2048, 32),
        ("4K Prefill", 4096, 32),
        ("8K Prefill", 8192, 32),
        ("16K Prefill + 500 Decode", 16384, 500)
    ]
    
    results = []
    for name, p_tok, d_tok in test_cases:
        res = run_test(name, p_tok, d_tok)
        if res:
            results.append(res)
        time.sleep(1)
        
    print("\n" + "=" * 80)
    print(f"{'Test':<26} | {'Prompt':<7} | {'Decode':<7} | {'TTFT(s)':<8} | {'Prefill tok/s':<14} | {'Dec tok/s':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<26} | {r['prompt_tokens']:<7} | {r['decode_tokens']:<7} | {r['ttft_s']:<8.3f} | {r['prefill_tps']:<14.1f} | {r['decode_tps']:<10.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
