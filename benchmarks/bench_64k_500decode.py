import time
import requests
import json
import uuid
import sys
from transformers import AutoTokenizer

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Next-NVFP4-QAD"
TOKENIZER_PATH = "/models/local-inference-lab-Qwen3.8-Flash-Next-NVFP4-FTW"

TARGET_PROMPT_TOKENS = 65536  # 64K
TARGET_DECODE_TOKENS = 500

print(f"1. Inicijalizacija tokenizera sa: {TOKENIZER_PATH}...")
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

print(f"2. Generisanje unikatnog 64K konteksta (~{TARGET_PROMPT_TOKENS} tokena)...")
t_prep = time.time()
base_chunk = (
    "In large-scale distributed systems and high-performance computing, mixture-of-experts architectures "
    "partition neural network parameters across multiple sparse feed-forward expert banks. Specialized "
    "routing algorithms dynamically evaluate gating logits to select the top-k expert subnetworks for each "
    "token, achieving vast parameter capacities while constraining runtime FLOPs. Tensor parallelism "
    "further shards matrix multiplications across high-bandwidth GPU clusters interconnected via NVLink. "
    "Advanced memory architectures employ hybrid linear attention, gated delta networks, and recurrent state "
    "snapshots to bound context caching overhead across extensive sequence lengths. "
)

base_tokens = tok.encode(base_chunk)
needed_reps = (TARGET_PROMPT_TOKENS // len(base_tokens)) + 2
salt = f"Test64k-Nonce-{uuid.uuid4().hex}: "

# Kreiramo tekst ponavljanjem i odsecamo na tačno TARGET_PROMPT_TOKENS
raw_text = salt + (base_chunk * needed_reps)
token_ids = tok.encode(raw_text)[:TARGET_PROMPT_TOKENS]
final_prompt = tok.decode(token_ids)
actual_prompt_len = len(token_ids)

print(f"   -> Pripremljen prompt: tačno {actual_prompt_len} tokena (priprema trajala {time.time()-t_prep:.2f}s).")

instruction = (
    "\n\nMolim te napiši detaljnu, duboku i sveobuhvatnu stručnu analizu na srpskom jeziku o "
    "prednostima i izazovima MoE arhitektura, distribuiranog paralelizma i efikasnog skaliranja "
    "velikih jezičkih modela na osnovu prethodnog konteksta. Objasni ključne aspekte u detalje."
)

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": final_prompt + instruction}
    ],
    "max_tokens": TARGET_DECODE_TOKENS,
    "temperature": 0.6,
    "stream": True,
    "stream_options": {"include_usage": True}
}

print(f"\n3. Slanje 64K zahteva ka {URL}...")
print(f"   - Target Prompt: {actual_prompt_len} tokena (16 chunkova po 4096)")
print(f"   - Target Decode: {TARGET_DECODE_TOKENS} tokena")
print(f"   - Očekivano vreme prefill-a: ~45-50s, decode-a: ~9-10s...")

t0 = time.time()
try:
    resp = requests.post(URL, json=payload, stream=True, timeout=600)
except Exception as e:
    print(f"Greška u konekciji: {e}")
    sys.exit(1)

if resp.status_code != 200:
    print(f"HTTP Greška {resp.status_code}: {resp.text}")
    sys.exit(1)

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
                piece = delta.get("reasoning_content") or delta.get("content")
                if piece:
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft_now = first_token_time - t0
                        print(f"   --> [PRVI TOKEN STIGAO!] TTFT: {ttft_now:.2f} s (~{actual_prompt_len/ttft_now:.1f} tok/s)")
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

reported_prompt = usage_data.get("prompt_tokens", actual_prompt_len) if usage_data else actual_prompt_len
reported_completion = usage_data.get("completion_tokens", chunks) if usage_data else chunks
prefill_tps = reported_prompt / ttft if ttft > 0 else 0

print("\n" + "=" * 75)
print("REZULTATI BENČMARKA: 64K PREFILL + 500 DECODE")
print("=" * 75)
print(f"Stvarni Prompt Tokeni (Prefill):   {reported_prompt}")
print(f"Generisano Tokena (Decode):        {chunks} (Reasoning: {reasoning_tokens}, Odgovor: {content_tokens})")
print(f"TTFT (Prefill Vreme):              {ttft:.3f} s")
print(f"Prefill Brzina:                    {prefill_tps:.2f} tokens/s")
print(f"Decode Vreme:                      {decode_time:.3f} s")
print(f"Decode Brzina:                     {decode_tps:.2f} tokens/s")
print(f"Ukupno Vreme (End-to-End Latency): {total_time:.3f} s")
print("=" * 75)

if reasoning_text:
    preview_r = reasoning_text.strip()[:200].replace('\n', ' ')
    print(f"\n[Razmišljanje modela preview - {reasoning_tokens} tokena]:\n{preview_r}...")
if content_text:
    preview_c = content_text.strip()[:200].replace('\n', ' ')
    print(f"\n[Odgovor modela preview - {content_tokens} tokena]:\n{preview_c}...")
print("=" * 75)
