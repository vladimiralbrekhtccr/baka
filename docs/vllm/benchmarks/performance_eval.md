# Performnace evaluation TTFT, etc.

If you want to evaluatin **Image** or **Audio** make sure to download Assets first.

## Assets can be download with:
```python
from huggingface_hub import hf_hub_download, login, HfApi
import dotenv, os, shutil

dotenv.load_dotenv()
login(token=os.environ["HF_TOKEN"])

REPO_ID = "CCRss/kv_brain"
PATH_IN_REPO = "performance_eval_vllm/assets"

LOCAL_DIR = "/scratch/vladimir_albrekht/projects/10_09_2025_MOe/debug_inf/assets"

def download_all_files(repo_id, path_in_repo, local_dir):
    print(f"Downloading files from {repo_id}/{path_in_repo} ...")
    os.makedirs(local_dir, exist_ok=True)

    api = HfApi()
    repo_type = "dataset"
    files = api.list_repo_files(repo_id, repo_type=repo_type)

    # only files inside PATH_IN_REPO
    data_files = [f for f in files if f.startswith(path_in_repo)]

    for filename in data_files:
        try:
            # download to cache (don’t enforce local_dir mirroring!)
            cached_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=None,
                local_dir_use_symlinks=False
            )

            # strip "performance_eval_vllm/assets/" prefix
            short_name = filename.replace(path_in_repo + "/", "")
            target_path = os.path.join(local_dir, short_name)

            # ensure parent dirs exist if short_name had subfolders
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(cached_file, target_path)

            print(f"Saved {short_name} → {target_path}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

download_all_files(REPO_ID, PATH_IN_REPO, LOCAL_DIR)
```



## Audio evaluation
```python
import json
import time
import math
import base64
import io
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torchaudio
from tqdm import tqdm

# ----- Config -----
SERVED_NAME = "kita"
BASE_URL = "http://localhost:6655/v1"
API_URL = f"{BASE_URL}/chat/completions"
HEADERS = {"Authorization": "Bearer empty", "Content-Type": "application/json"}

# Текстовая часть запроса (то, что будем токенизировать)
SYSTEM_PROMPT = (
    "Your name is Oylan, you are a useful multi-modal large language model "
    "developed by ISSAI, Kazakhstan."
)
USER_TEXT = "<audio>\nWhat was said in this audio"

# Параметры генерации
MAX_TOKENS = 512
TEMPERATURE = 0.1

# ----- Tokenizer API (/tokenize) -----
def _api_root_from_base(base_url: str) -> str:
    return base_url.split("/v1")[0] if "/v1" in base_url else base_url

_API_ROOT = _api_root_from_base(BASE_URL)
_TOKENIZE_ENDPOINT = f"{_API_ROOT}/tokenize"

def _token_count(text: str) -> int:
    try:
        r = requests.post(_TOKENIZE_ENDPOINT, json={"prompt": text}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data.get("count"), int):
            return data["count"]
        if isinstance(data.get("tokens"), list):
            return len(data["tokens"])
    except Exception:
        pass
    return 0

# ----- Аудио helpers -----
def _to_wav_bytes(path: Path) -> bytes:
    ext = path.suffix.lower()
    if ext == ".wav":
        return path.read_bytes()
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    buf = io.BytesIO()
    torchaudio.save(buf, waveform, 16000, format="wav")
    return buf.getvalue()

def load_audio_base64_wav(path: Path) -> str:
    wav_bytes = _to_wav_bytes(path)
    return base64.b64encode(wav_bytes).decode("utf-8")

# ----- Построение payload (без OpenAI SDK, чтобы свободно класть audio_url) -----
def _build_payload(audio_b64: str) -> dict:
    return {
        "model": SERVED_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_TEXT},
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
                ],
            },
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

# ---------- Метрики ----------
def _percentile(sorted_vals: List[float], p: float):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

# ---------- Запрос к серверу ----------
def _one_request(audio_name: str, audio_b64: str) -> dict:
    start = time.perf_counter()
    try:
        payload = _build_payload(audio_b64)
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        latency_ms = (time.perf_counter() - start) * 1000.0

        if resp.status_code != 200:
            preview = (resp.text or "")[:160].replace("\n", " ")
            return {
                "audio_name": audio_name, "ok": False, "status_code": resp.status_code,
                "latency_ms": latency_ms, "error": f"http {resp.status_code}", "preview": preview,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            }

        data = resp.json()
        text = None
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            text = None
        usage = data.get("usage") or {}
        pt = usage.get("prompt_tokens") or 0
        ct = usage.get("completion_tokens") or 0
        tt = usage.get("total_tokens") or (pt + ct)

        return {
            "audio_name": audio_name, "ok": True, "status_code": resp.status_code,
            "latency_ms": latency_ms, "gemma_translation": text,
            "preview": (text or "")[:160].replace("\n", " "),
            "prompt_tokens": int(pt), "completion_tokens": int(ct), "total_tokens": int(tt),
        }

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "audio_name": audio_name, "ok": False, "status_code": 0,
            "latency_ms": latency_ms, "error": str(e), "preview": None,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }

# ---------- Основная функция: список аудио -> JSONL + метрики ----------
def evaluate_audio_list_to_jsonl(
    audio_paths: List[Path],
    out_jsonl_path: str,
    total_requests: int = 100,
    max_workers: int = 32,
):
    """
    Берёт список путей к аудио, повторяет их по кругу до total_requests,
    шлёт параллельно, пишет JSONL и печатает метрики (RPS, перцентили).
    """
    # Предзагрузка аудио в память, чтобы CPU/IO не мешали метрикам
    preloaded: List[Tuple[str, str]] = []
    for p in audio_paths:
        if not p.exists():
            raise FileNotFoundError(f"Audio not found: {p}")
        b64 = load_audio_base64_wav(p)
        preloaded.append((p.name, b64))
    if not preloaded:
        raise RuntimeError("Empty audio list.")

    seq = [preloaded[i % len(preloaded)] for i in range(total_requests)]

    t0 = time.perf_counter()
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex, \
         open(out_jsonl_path, "w", encoding="utf-8") as fout:

        for idx, (name, b64) in enumerate(seq):
            futures.append(ex.submit(_one_request, name, b64))

        latencies = []
        ok_cnt = 0
        total_pt = 0
        total_ct = 0

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            rec = fut.result()
            # добавим текстовую часть запроса для токен-агрегации (как у текстового скрипта)
            rec["original_text"] = USER_TEXT  # сохраним поле, как в "первом файле"
            # потоковая запись
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if rec.get("ok"):
                ok_cnt += 1
                lat = rec.get("latency_ms")
                if isinstance(lat, (int, float)):
                    latencies.append(lat)
                total_pt += int(rec.get("prompt_tokens", 0))
                total_ct += int(rec.get("completion_tokens", 0))

    wall = time.perf_counter() - t0
    latencies.sort()
    eff_rps = (ok_cnt / wall) if wall > 0 else float("inf")

    metrics = {
        "total_requests": total_requests,
        "successful": ok_cnt,
        "failed": total_requests - ok_cnt,
        "total_time_s": wall,
        "effective_rps": eff_rps,
        "latency_ms": {
            "avg": (sum(latencies) / len(latencies)) if latencies else float("nan"),
            "min": latencies[0] if latencies else float("nan"),
            "max": latencies[-1] if latencies else float("nan"),
            "p50": _percentile(latencies, 50) if latencies else float("nan"),
            "p90": _percentile(latencies, 90) if latencies else float("nan"),
            "p99": _percentile(latencies, 99) if latencies else float("nan"),
            "count": len(latencies),
        },
        "max_workers": max_workers,
        # Если сервер вернул usage — уже агрегировано
        "usage_sum": {
            "total_input_tokens": total_pt,
            "total_generated_tokens": total_ct,
            "total_tokens": total_pt + total_ct,
            "input_tokens_per_sec": (total_pt / wall) if wall > 0 else float("inf"),
            "output_tokens_per_sec": (total_ct / wall) if wall > 0 else float("inf"),
            "total_throughput_tokens_per_sec": ((total_pt + total_ct) / wall) if wall > 0 else float("inf"),
        },
    }

    print(f"\nDone. Wrote JSONL to: {out_jsonl_path}")
    print(f"Total Time: {metrics['total_time_s']:.2f} seconds")
    print(f"Effective RPS: {metrics['effective_rps']:.2f} req/s\n")
    print("Request Latency (ms):")
    lm = metrics["latency_ms"]
    print(f"  Avg: {lm['avg']:.1f}, Min: {lm['min']:.1f}, Max: {lm['max']:.1f}")
    print(f"  P50: {lm['p50']:.1f}, P90: {lm['p90']:.1f}, P99: {lm['p99']:.1f}")

    # Если usage не пришёл, можно добить токены через /tokenize по текстовой части (system+user)
    if metrics["usage_sum"]["total_tokens"] == 0:
        summary = summarize_tokens_with_tokenizer_api(
            jsonl_path=out_jsonl_path,
            total_time_s=wall,
        )
        metrics["tokenize_sum"] = summary

    # Печать токенов (usage если есть, иначе tokenize)
    use = metrics["usage_sum"] if metrics["usage_sum"]["total_tokens"] > 0 else metrics.get("tokenize_sum", {})
    if use:
        print("\nTokens:")
        print(f"  Total Input Tokens      : {use.get('total_input_tokens')}")
        print(f"  Total Generated Tokens  : {use.get('total_generated_tokens')}")
        print(f"  Input (tokens/s)        : {use.get('input_tokens_per_sec'):.2f}")
        print(f"  Output (tokens/s)       : {use.get('output_tokens_per_sec'):.2f}")
        print(f"  Total Throughput        : {use.get('total_throughput_tokens_per_sec'):.2f} tokens/s")

    return metrics

# --- Суммирование токенов через /tokenize по JSONL (только текстовая часть запроса и ответ) ---
def summarize_tokens_with_tokenizer_api(jsonl_path: str, total_time_s: float):
    total_input_tokens = 0
    total_output_tokens = 0
    ok = 0
    for line in open(jsonl_path, "r", encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("ok"):
            continue
        ok += 1
        # считаем токены только для текстовой части (без audio_url)
        # system + user_text
        text_prompt = f"{SYSTEM_PROMPT}\n\n{rec.get('original_text','')}"
        total_input_tokens += _token_count(text_prompt)
        total_output_tokens += _token_count(rec.get("gemma_translation") or "")

    input_tps = (total_input_tokens / total_time_s) if total_time_s > 0 else float("inf")
    output_tps = (total_output_tokens / total_time_s) if total_time_s > 0 else float("inf")
    total_tps = ((total_input_tokens + total_output_tokens) / total_time_s) if total_time_s > 0 else float("inf")

    return {
        "successful": ok,
        "total_input_tokens": total_input_tokens,
        "total_generated_tokens": total_output_tokens,
        "input_tokens_per_sec": input_tps,
        "output_tokens_per_sec": output_tps,
        "total_throughput_tokens_per_sec": total_tps,
    }

# --------- Example run with AUDIO LIST ---------
if __name__ == "__main__":
    AUDIO_FILES = [
        Path("assets/test_1.mp3"),
        Path("assets/test_3.mp3"),
        Path("assets/rustem_1.wav"),
        Path("assets/rustem_2.wav"),
    ]

    N = 100          # ровно столько запросов отправим (циклом по списку)
    OUT = "audio_eval_results.jsonl"

    evaluate_audio_list_to_jsonl(
        audio_paths=AUDIO_FILES,
        out_jsonl_path=OUT,
        total_requests=N,
        max_workers=32,
    )
```

## Image evaluation

```python
import json
import time
import math
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import requests

# ----- Your existing config -----
MODELS = ["ki_l", "ki_oylan_a_v_t_2_5"]
BASE_URL = "http://localhost:6655/v1"

IMAGES_DIR = Path("assets")
IMAGE_NAMES = ["cute_girl.jpeg"] + [f"{i}.jpg" for i in range(1, 10)]  # total 10 names

SYSTEM_PROMPT = "Ты — наблюдательный ассистент. Отвечай кратко и по делу."
# All prompts MUST include "<image>" first (your requirement)
IMAGE_TASKS = [
    "<image> Describe this image in detail.",
    "<image> List the main objects and their colors.",
    "<image> Summarize the scene (who/what/where).",
    "<image> What is the mood and style? Explain briefly.",
    "<image> Identify notable attributes (clothes, pose, background).",
    "<image> Give 3 tags that describe this image.",
    "<image> Describe composition (foreground/background, framing).",
    "<image> What stands out first to the viewer? Why?",
    "<image> Describe lighting and color palette.",
    "<image> One-sentence caption, then 3 bullet facts.",
]

def _api_root_from_base(base_url: str) -> str:
    return base_url.split("/v1")[0] if "/v1" in base_url else base_url

_API_ROOT = _api_root_from_base(BASE_URL)
_TOKENIZE_ENDPOINT = f"{_API_ROOT}/tokenize"

def _token_count(text: str) -> int:
    try:
        r = requests.post(_TOKENIZE_ENDPOINT, json={"prompt": text}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "count" in data and isinstance(data["count"], int):
            return data["count"]
        if "tokens" in data and isinstance(data["tokens"], list):
            return len(data["tokens"])
    except Exception:
        pass
    return 0

def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    # Fallback to JPEG
    return "image/jpeg"

def encode_image_b64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")

def get_image_prediction(image_path: Path, prompt_text: str, model_name: str):
    client = OpenAI(api_key="empty", base_url=BASE_URL)  # local client per call (thread-safe)
    
    mime = guess_mime(image_path)
    b64_image = encode_image_b64(image_path)
    
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=50,
    )
    return resp.choices[0].message.content, resp.usage

# ---------- NEW: оценка скорости по списку ----------
def _percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

def evaluate_images_to_jsonl(
    image_paths_list,
    prompts_list,
    out_jsonl_path,
    total_requests=100,
    max_workers=32,
    max_retries=3,
    retry_backoff=0.7,
):
    """
    Берёт СПИСОК путей к изображениям и промптов, отправляет до `max_workers` параллельных запросов,
    и пишет результаты (и время обработки запроса) в JSONL:
        {"index": ..., "image_name": ..., "prompt_text": ..., "response": ..., "latency_ms": ..., "ok": ...}
    Возвращает метрики со стендовым временем (wall time).
    """

    def worker(idx, image_path, prompt, model_name):
        last_err = None
        for attempt in range(1, max_retries + 1):
            start = time.perf_counter()
            try:
                response, usage = get_image_prediction(image_path, prompt, model_name)
                latency_ms = (time.perf_counter() - start) * 1000.0
                return {
                    "index": idx, 
                    "image_name": image_path.name,
                    "prompt_text": prompt,
                    "model_name": model_name,
                    "gemma_translation": response, 
                    "latency_ms": latency_ms, 
                    "ok": True,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                }
            except Exception as e:
                last_err = e
                time.sleep(retry_backoff * attempt)
        latency_ms = (time.perf_counter() - start) * 1000.0  # время последней попытки
        return {
            "index": idx, 
            "image_name": image_path.name,
            "prompt_text": prompt,
            "model_name": model_name,
            "error": str(last_err),
            "latency_ms": latency_ms, 
            "ok": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # Build sequence for total_requests (round-robin images & prompts, alternating models)
    sequence = []
    for i in range(total_requests):
        image_path = image_paths_list[i % len(image_paths_list)]
        prompt_text = prompts_list[i % len(prompts_list)]
        model_name = MODELS[i % len(MODELS)]  # Alternate between models
        sequence.append((image_path, prompt_text, model_name))

    futures = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex, \
         open(out_jsonl_path, "w", encoding="utf-8") as fout:

        for idx, (image_path, prompt_text, model_name) in enumerate(sequence):
            futures.append(ex.submit(worker, idx, image_path, prompt_text, model_name))

        latencies = []
        ok_cnt = 0
        total_pt = 0
        total_ct = 0

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            rec = fut.result()
            if rec.get("ok"):
                ok_cnt += 1
                if rec.get("latency_ms") is not None:
                    latencies.append(rec["latency_ms"])
                total_pt += int(rec.get("prompt_tokens", 0))
                total_ct += int(rec.get("completion_tokens", 0))
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_time_s = time.perf_counter() - t0
    eff_rps = (ok_cnt / total_time_s) if total_time_s > 0 else float("inf")
    latencies.sort()

    metrics = {
        "total_requests": total_requests,
        "successful": ok_cnt,
        "failed": total_requests - ok_cnt,
        "total_time_s": total_time_s,
        "effective_rps": eff_rps,
        "latency_ms": {
            "avg": (sum(latencies) / len(latencies)) if latencies else float("nan"),
            "min": latencies[0] if latencies else float("nan"),
            "max": latencies[-1] if latencies else float("nan"),
            "p50": _percentile(latencies, 50) if latencies else float("nan"),
            "p90": _percentile(latencies, 90) if latencies else float("nan"),
            "p99": _percentile(latencies, 99) if latencies else float("nan"),
            "count": len(latencies),
        },
        "max_workers": max_workers,
        # Actual usage from server responses
        "usage_sum": {
            "total_input_tokens": total_pt,
            "total_generated_tokens": total_ct,
            "total_tokens": total_pt + total_ct,
            "input_tokens_per_sec": (total_pt / total_time_s) if total_time_s > 0 else float("inf"),
            "output_tokens_per_sec": (total_ct / total_time_s) if total_time_s > 0 else float("inf"),
            "total_throughput_tokens_per_sec": ((total_pt + total_ct) / total_time_s) if total_time_s > 0 else float("inf"),
        },
    }

    print(f"\nDone. Wrote JSONL to: {out_jsonl_path}")
    print(f"Total Time: {metrics['total_time_s']:.2f} seconds")
    print(f"Effective RPS: {metrics['effective_rps']:.2f} req/s\n")
    print("Request Latency (ms):")
    lm = metrics["latency_ms"]
    print(f"  Avg: {lm['avg']:.1f}, Min: {lm['min']:.1f}, Max: {lm['max']:.1f}")
    print(f"  P50: {lm['p50']:.1f}, P90: {lm['p90']:.1f}, P99: {lm['p99']:.1f}")

    # Print actual token usage from server responses (includes image tokens!)
    if metrics["usage_sum"]["total_tokens"] > 0:
        use = metrics["usage_sum"]
        print("\nTokens (from API usage):")
        print(f"  Total Input Tokens      : {use['total_input_tokens']}")
        print(f"  Total Generated Tokens  : {use['total_generated_tokens']}")
        print(f"  Input (tokens/s)        : {use['input_tokens_per_sec']:.2f}")
        print(f"  Output (tokens/s)       : {use['output_tokens_per_sec']:.2f}")
        print(f"  Total Throughput        : {use['total_throughput_tokens_per_sec']:.2f} tokens/s")

    return metrics

# --------- NEW: после генерации — считаем токены через /tokenize ---------
def summarize_tokens_with_tokenizer_api(
    jsonl_path: str,
    total_time_s: float,
):
    total_input_tokens = 0
    total_output_tokens = 0
    ok = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            ok += 1
            prompt_text = rec.get("prompt_text") or ""
            output = rec.get("gemma_translation") or ""
            # ВОССТАНАВЛИВАЕМ ровно тот текст-промпт, что отправляли:
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt_text}"
            total_input_tokens += _token_count(full_prompt)
            total_output_tokens += _token_count(output)

    input_tps = (total_input_tokens / total_time_s) if total_time_s > 0 else float("inf")
    output_tps = (total_output_tokens / total_time_s) if total_time_s > 0 else float("inf")
    total_tps = ((total_input_tokens + total_output_tokens) / total_time_s) if total_time_s > 0 else float("inf")

    print("\nTokens (via /tokenize):")
    print(f"  Successful requests     : {ok}")
    print(f"  Total Input Tokens      : {total_input_tokens}")
    print(f"  Total Generated Tokens  : {total_output_tokens}")
    print(f"  Input (tokens/s)        : {input_tps:.2f}")
    print(f"  Output (tokens/s)       : {output_tps:.2f}")
    print(f"  Total Throughput        : {total_tps:.2f} tokens/s")

    return {
        "successful": ok,
        "total_input_tokens": total_input_tokens,
        "total_generated_tokens": total_output_tokens,
        "input_tokens_per_sec": input_tps,
        "output_tokens_per_sec": output_tps,
        "total_throughput_tokens_per_sec": total_tps,
    }

# --------- Example run with IMAGE LIST ---------
if __name__ == "__main__":
    # Preload available images
    image_files = []
    for name in IMAGE_NAMES:
        p = IMAGES_DIR / name
        if p.exists():
            image_files.append(p)
        else:
            print(f"⚠ Skipping missing image: {p}")
    
    if not image_files:
        raise RuntimeError("No images found to benchmark.")

    N = 100
    OUT = "image_eval_results.jsonl"

    metrics = evaluate_images_to_jsonl(
        image_files, 
        IMAGE_TASKS, 
        OUT, 
        total_requests=N, 
        max_workers=128
    )

    # If server didn't provide usage tokens, fall back to tokenizer API for text-only estimation
    if metrics["usage_sum"]["total_tokens"] == 0:
        print("\nNo usage tokens from server, falling back to text-only tokenizer estimation:")
        summarize_tokens_with_tokenizer_api(
            jsonl_path=OUT,
            total_time_s=metrics["total_time_s"],
        )
```

## Text evaluation

```python
import json
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import requests

# ----- Your existing config -----
SERVED_NAME = "kita"
BASE_URL = "http://localhost:6655/v1"

all_langs = {
    "kk": "Kazakh",
    "en": "English",
    "ru": "Russian",
}

def _api_root_from_base(base_url: str) -> str:
    return base_url.split("/v1")[0] if "/v1" in base_url else base_url

_API_ROOT = _api_root_from_base(BASE_URL)
_TOKENIZE_ENDPOINT = f"{_API_ROOT}/tokenize"

def _token_count(text: str) -> int:
    try:
        r = requests.post(_TOKENIZE_ENDPOINT, json={"prompt": text}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "count" in data and isinstance(data["count"], int):
            return data["count"]
        if "tokens" in data and isinstance(data["tokens"], list):
            return len(data["tokens"])
    except Exception:
        pass
    return 0

def _make_user_prompt(input_text: str, tgt_lang: str, src_lang=None, tgt_mode='text') -> str:
    instruction = f'Translate the following text into {all_langs[tgt_lang]}.'
    NEWLINE = '\n'
    if NEWLINE in input_text:
        instruction += f' Preserve every {NEWLINE} token—same count.'
    if tgt_mode == 'speech':
        instruction = instruction + " Transcribe all numbers as read."
    if tgt_mode == 'speech' and src_lang and src_lang == tgt_lang:
        instruction = f"Do not translate or change this {all_langs[src_lang]} text, only transcribe all numbers as read."
    return f"{instruction}\n\n{input_text}"

def get_prediction(input_text, tgt_lang, src_lang=None, tgt_mode='text'):
    client = OpenAI(api_key="empty", base_url=BASE_URL)  # local client per call (thread-safe)
    _input = _make_user_prompt(input_text, tgt_lang, src_lang, tgt_mode)
    resp = client.chat.completions.create(
        model=SERVED_NAME,
        messages=[{"role": "user", "content": _input}],
        temperature=0.05,
        top_p=0.95,
        max_tokens=2048,
        frequency_penalty=0.3,
    )
    return resp.choices[0].message.content

# ---------- NEW: оценка скорости по списку ----------
def _percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

def evaluate_list_to_jsonl(
    texts_list,
    out_jsonl_path,
    tgt_lang="kk",
    max_workers=100,
    max_retries=3,
    retry_backoff=0.7,
):
    """
    Берёт СПИСОК строк `texts_list`, отправляет до `max_workers` параллельных запросов,
    и пишет результаты (и время обработки запроса) в JSONL:
        {"index": ..., "original_text": ..., "gemma_translation": ..., "latency_ms": ..., "ok": ...}
    Возвращает метрики со стендовым временем (wall time).
    """

    def worker(idx, src_text):
        last_err = None
        for attempt in range(1, max_retries + 1):
            start = time.perf_counter()
            try:
                tr = get_prediction(src_text, tgt_lang)
                latency_ms = (time.perf_counter() - start) * 1000.0
                return {"index": idx, "original_text": src_text,
                        "gemma_translation": tr, "latency_ms": latency_ms, "ok": True}
            except Exception as e:
                last_err = e
                time.sleep(retry_backoff * attempt)
        latency_ms = (time.perf_counter() - start) * 1000.0  # время последней попытки
        return {"index": idx, "original_text": src_text, "error": str(last_err),
                "latency_ms": latency_ms, "ok": False}

    futures = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex, \
         open(out_jsonl_path, "w", encoding="utf-8") as fout:

        for idx, src_text in enumerate(texts_list):
            futures.append(ex.submit(worker, idx, src_text))

        latencies = []
        ok_cnt = 0

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            rec = fut.result()
            if rec.get("ok"):
                ok_cnt += 1
                if rec.get("latency_ms") is not None:
                    latencies.append(rec["latency_ms"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_time_s = time.perf_counter() - t0
    eff_rps = (ok_cnt / total_time_s) if total_time_s > 0 else float("inf")
    latencies.sort()

    metrics = {
        "total_requests": len(texts_list),
        "successful": ok_cnt,
        "failed": len(texts_list) - ok_cnt,
        "total_time_s": total_time_s,
        "effective_rps": eff_rps,
        "latency_ms": {
            "avg": (sum(latencies) / len(latencies)) if latencies else float("nan"),
            "min": latencies[0] if latencies else float("nan"),
            "max": latencies[-1] if latencies else float("nan"),
            "p50": _percentile(latencies, 50) if latencies else float("nan"),
            "p90": _percentile(latencies, 90) if latencies else float("nan"),
            "p99": _percentile(latencies, 99) if latencies else float("nan"),
            "count": len(latencies),
        },
        "max_workers": max_workers,
    }

    print(f"\nDone. Wrote JSONL to: {out_jsonl_path}")
    print(f"Total Time: {metrics['total_time_s']:.2f} seconds")
    print(f"Effective RPS: {metrics['effective_rps']:.2f} req/s\n")
    print("Request Latency (ms):")
    lm = metrics["latency_ms"]
    print(f"  Avg: {lm['avg']:.1f}, Min: {lm['min']:.1f}, Max: {lm['max']:.1f}")
    print(f"  P50: {lm['p50']:.1f}, P90: {lm['p90']:.1f}, P99: {lm['p99']:.1f}")

    return metrics

# --------- NEW: после генерации — считаем токены через /tokenize ---------
def summarize_tokens_with_tokenizer_api(
    jsonl_path: str,
    tgt_lang: str,
    total_time_s: float,
    src_lang=None,
    tgt_mode='text',
):
    total_input_tokens = 0
    total_output_tokens = 0
    ok = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            ok += 1
            original = rec.get("original_text") or ""
            output = rec.get("gemma_translation") or ""
            # ВОССТАНАВЛИВАЕМ ровно тот user-подсказ, что отправляли:
            prompt_text = _make_user_prompt(original, tgt_lang, src_lang, tgt_mode)
            total_input_tokens += _token_count(prompt_text)
            total_output_tokens += _token_count(output)

    input_tps = (total_input_tokens / total_time_s) if total_time_s > 0 else float("inf")
    output_tps = (total_output_tokens / total_time_s) if total_time_s > 0 else float("inf")
    total_tps = ((total_input_tokens + total_output_tokens) / total_time_s) if total_time_s > 0 else float("inf")

    print("\nTokens (via /tokenize):")
    print(f"  Successful requests     : {ok}")
    print(f"  Total Input Tokens      : {total_input_tokens}")
    print(f"  Total Generated Tokens  : {total_output_tokens}")
    print(f"  Input (tokens/s)        : {input_tps:.2f}")
    print(f"  Output (tokens/s)       : {output_tps:.2f}")
    print(f"  Total Throughput        : {total_tps:.2f} tokens/s")

    return {
        "successful": ok,
        "total_input_tokens": total_input_tokens,
        "total_generated_tokens": total_output_tokens,
        "input_tokens_per_sec": input_tps,
        "output_tokens_per_sec": output_tps,
        "total_throughput_tokens_per_sec": total_tps,
    }

# --------- Example run with LIST ---------
if __name__ == "__main__":
    QUESTIONS = [
        "Какое у тебя любимое аниме? /no_think",
        "Что важнее в аниме: сюжет или визуал? /no_think",
        "Какое аниме ты бы порекомендовал новичку? /no_think",
        "Какая опенинг-песня у тебя в топе? /no_think",
        "Какой персонаж тебя вдохновляет? /no_think",
    ]

    N = 100
    TEXTS = (QUESTIONS * math.ceil(N / len(QUESTIONS)))[:N]
    OUT = "translations_from_list.jsonl"

    metrics = evaluate_list_to_jsonl(TEXTS, OUT, tgt_lang="kk", max_workers=128)

    # После генерации считаем токены через /tokenize и печатаем сводку
    summarize_tokens_with_tokenizer_api(
        jsonl_path=OUT,
        tgt_lang="kk",
        total_time_s=metrics["total_time_s"],
)
```