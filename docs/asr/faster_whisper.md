```bash
conda activate gemma_translator
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ctranslate2 faster-whisper
LD_LIBRARY_PATH=/home/vladimir_albrekht/miniconda3/envs/gemma_translator/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```


```python
# working conda is gemma_translator

# convert_fp16_minimal.py
# ВВЕДИ ТОЛЬКО ЭТИ ДВА ПУТИ:
MODEL = r"whisper"  # локальная папка или HF repo id
OUT = r"whisper-fp16"  # куда сохранить CT2

# -------------------------------------------------------------
# Ничего больше менять не нужно
import os
from pathlib import Path


def ensure_tokenizer_json(model_dir: Path):
    tj = model_dir / "tokenizer.json"
    if tj.exists():
        return
    # Создаём токенайзер явно (WhisperTokenizerFast -> надёжнее для Whisper)
    try:
        from transformers import WhisperTokenizerFast
        tok = WhisperTokenizerFast.from_pretrained(str(model_dir))
    except Exception:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    tok.save_pretrained(str(model_dir))
    if not tj.exists():
        raise RuntimeError("Не удалось создать tokenizer.json — проверь исходную модель.")


def ensure_preprocessor(model_dir: Path):
    pj = model_dir / "preprocessor_config.json"
    if not pj.exists():
        raise FileNotFoundError(
            "Нет preprocessor_config.json в исходной модели. "
            "Для Whisper он обязателен."
        )


def convert_fp16(model_ref: str, out_dir: Path):
    from ctranslate2.converters import TransformersConverter
    conv = TransformersConverter(
        model_name_or_path=model_ref,
        copy_files=["tokenizer.json", "preprocessor_config.json", "added_tokens.json",
                    "generation_config.json", "normalizer.json", "special_tokens_map.json", "tokenizer_config.json",
                    "vocab.json"],
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(conv.convert(output_dir=str(out_dir), quantization="float16", force=True))


def smoke_test(ct2_dir: Path, lang: str = "kk"):
    from faster_whisper import WhisperModel
    import numpy as np
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    device = "cuda" if has_cuda else "cpu"
    compute_type = "float16" if has_cuda else "int8_float16"

    model = WhisperModel(str(ct2_dir), device=device, compute_type=compute_type)
    # секунда тишины, просто чтобы убедиться, что декодер живой
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    segments, info = model.transcribe(audio, language=lang, vad_filter=False)
    _ = list(segments)
    print(f"[SMOKE] OK | device={device}/{compute_type}, language={info.language}, p={info.language_probability:.2f}")


def main():
    assert MODEL and OUT, "Укажи MODEL и OUT вверху файла."
    model_ref = MODEL
    out_dir = Path(OUT)

    # Если модель локальная — подготовим обязательные файлы
    model_path = Path(model_ref)
    if model_path.exists():
        ensure_tokenizer_json(model_path)
        ensure_preprocessor(model_path)

    ct2_dir = convert_fp16(model_ref, out_dir)

    # Быстрая проверка наличия ключевых файлов
    for f in ("model.bin", "tokenizer.json", "preprocessor_config.json"):
        if not (ct2_dir / f).exists():
            raise RuntimeError(f"После конвертации отсутствует: {f}")

    smoke_test(ct2_dir)
    print(f"[DONE] CT2 FP16 готово: {ct2_dir}")


if __name__ == "__main__":
    # Если репо приватное на HF — можешь выставить токен:
    # os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", "<hf_token>")
    main()


### INFERENCE:
# conda activate gemma_translator
# cd /home/vladimir_albrekht/projects/digital_bridge/damumed/whisper_q
# conda activate gemma_translator
# cd /home/vladimir_albrekht/projects/digital_bridge/damumed/whisper_q
from faster_whisper import WhisperModel

model_size = "/home/vladimir_albrekht/projects/digital_bridge/damumed/whisper_model/whisper-mangisoz-best-10july2025-fp16"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio_damumed_sample.wav", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```