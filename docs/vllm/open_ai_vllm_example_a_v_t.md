# OpenAI example inference for vLLM Online serve

## Text inference 

```python
import time
import openai

client = openai.Client(
    base_url="http://localhost:6655/v1", api_key="EMPTY"
)
MODEL = "ki_oylan_a_v_t_2_5"


start_time = time.perf_counter()
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Ты любишь смотреть аниме. /no_think"},
        {"role": "user", "content": "Какое у тебя любимое аниме?"},
    ],
    temperature=0,
    max_tokens=256,
    stream=True
)

first_token_time = None
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        if first_token_time is None:  # first token arrived
            first_token_time = time.perf_counter()
            ttft = first_token_time - start_time
            print(f"\n\nTTFT: {ttft:.3f} seconds\n")
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Image inference
```python
# Image inference
import time
from pathlib import Path
import openai
import base64
client = openai.Client(
    base_url="http://localhost:6655/v1", api_key="EMPTY"
)
MODEL = "ki_l"

# ########################################################################
# ################ if you want to use local file #########################
# ########################################################################
# IMAGE_PATH = Path("/home/vladimir_albrekht/projects/digital_bridge/vllm/1_vladimir_utils/utils/benchs_perf/assets/cute_girl.jpg")


# def guess_mime(path: Path) -> str:
#     """Guess MIME type from file extension"""
#     ext = path.suffix.lower()
#     if ext in [".jpg", ".jpeg"]:
#         return "image/jpeg"
#     elif ext == ".png":
#         return "image/png"
#     elif ext == ".webp":
#         return "image/webp"
#     elif ext == ".gif":
#         return "image/gif"
#     else:
#         return "image/jpeg"  # Fallback

# def encode_image(image_path: Path) -> str:
#     """Encode image file to base64 string"""
#     return base64.b64encode(image_path.read_bytes()).decode("utf-8")

# base64_image = encode_image(IMAGE_PATH)
# mime_type = guess_mime(IMAGE_PATH)
# # data:{mime_type};base64,{base64_image}
# ########################################################################



start_time = time.perf_counter()
response = client.chat.completions.create(
  model=MODEL,
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe this image in detail", # make sure to inclue <image> otherwise it will crush.
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  f"https://huggingface.co/datasets/CCRss/kv_brain/resolve/main/Xnip2025-08-24_15-02-37.jpg" # for local path `data:{mime_type};base64,{base64_image}`
          },
        },
      ],
    }
  ],
  max_tokens=256,
  stream=True,
  temperature=0.1
)


first_token_time = None
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        if first_token_time is None:  # first token arrived
            first_token_time = time.perf_counter()
            ttft = first_token_time - start_time
            print(f"\n\nTTFT: {ttft:.3f} seconds\n")
        print(chunk.choices[0].delta.content, end="", flush=True)
```
    
## Audio inference

```python
# Audio request
import base64
import io
import time
from openai import OpenAI
import torch
from pathlib import Path
import requests
import soundfile as sf
import numpy as np
import torchaudio


client = OpenAI(base_url="http://localhost:6655/v1", api_key="EMPTY")
MODEL = "ki_oylan_a_v_t_2_5"


def encode_audio_from_url(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    data, sr = sf.read(io.BytesIO(resp.content))  # reads MP3/WAV/etc.
    if data.ndim > 1:
        data = np.mean(data, axis=1, keepdims=True)  # mono
    if sr != 16000:
        import torchaudio
        tensor = torchaudio.functional.resample(torch.from_numpy(data.T), sr, 16000)
        sr = 16000
        data = tensor.numpy().T
    with io.BytesIO() as buffer:
        sf.write(buffer, data, sr, format="WAV")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

########################################################################
# ################ if you want to use local file #########################
# ########################################################################
# AUDIO_PATH = Path("assets/test_3.mp3")
# def convert_to_wav_bytes(path: Path) -> bytes:
#     ext = path.suffix.lower()
#     if ext == ".wav":
#         return path.read_bytes()  # Already in WAV format

#     elif ext == ".mp3":
#         # Load MP3, convert to WAV format (16kHz mono float32)
#         waveform, sr = torchaudio.load(path)
#         if waveform.shape[0] > 1:
#             waveform = waveform.mean(dim=0, keepdim=True)  # Make mono
#         if sr != 16000:
#             resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
#             waveform = resampler(waveform)

#         # Save to in-memory WAV
#         with io.BytesIO() as buffer:
#             torchaudio.save(buffer, waveform, 16000, format="wav")
#             return buffer.getvalue()

#     else:
#         raise ValueError(f"Unsupported audio format: {ext}")

# def encode_audio(audio_path: Path) -> str:
#     """Encode audio file to base64 string"""
#     wav_bytes = convert_to_wav_bytes(audio_path)
#     return base64.b64encode(wav_bytes).decode("utf-8")

# base64_audio = encode_audio(AUDIO_PATH)

#######################################################################

start_time = time.perf_counter()
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "Your name is Oylan, you are a useful multi-modal large language model developed by ISSAI, Kazakhstan. /no_think"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Transcribe the audio"
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/wav;base64,{base64_audio}"
                        # or 
                        # "url": f"data:audio/wav;base64,{encode_audio_from_url('https://huggingface.co/datasets/CCRss/kv_brain/resolve/main/yes_my_lord.mp3')}"
                    }
                }
            ]
        }
    ],
    max_tokens=512,
    temperature=0.1,
    stream=True
)

# if stream=false
# print("✅ Response:", response.choices[0].message.content)
first_token_time = None
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        if first_token_time is None:  # first token arrived
            first_token_time = time.perf_counter()
            ttft = first_token_time - start_time
            print(f"\n\nTTFT: {ttft:.3f} seconds\n")
        print(chunk.choices[0].delta.content, end="", flush=True)


```