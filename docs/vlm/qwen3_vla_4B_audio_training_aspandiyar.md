# Qwen3VLA on ASR.

## Inference 

Setup your env like that

```bash
uv venv --python 3.12 .venv_vllm
source .venv_vllm/bin/activate
uv pip install vllm==0.11.2
```

You can deploy model like that in terminal preferable `tmux` such that if you will close laptop process will still be running on the server:

1. Create a new file `run_qwen3vl_server.sh`

```bash
# Model and environment settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="Qwen/Qwen3-VL-4B-Instruct"


MODEL_SERVED_NAME="qwen_deployed_model" 
PORT=6655
HOST="0.0.0.0"
SEED=0

# vLLM configuration parameters
GPU_MEMORY_UTILIZATION=0.90
TENSOR_PARALLEL_SIZE=4 # amount of gpus to use for splitting
DTYPE="auto"
MAX_NUM_BATCHED_TOKENS=32768
MAX_MODEL_LEN=8196
KV_CACHE_DTYPE="auto"
MAX_NUM_SEQS=50


CMD="vllm serve $MODEL \
  --tokenizer "$MODEL" \
  --host $HOST \
  --port $PORT \
  --served-model-name $MODEL_SERVED_NAME \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --max-model-len $MAX_MODEL_LEN \
  --trust-remote-code \
  --dtype $DTYPE \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --kv-cache-dtype $KV_CACHE_DTYPE \
  --max-num-seqs $MAX_NUM_SEQS \
  --seed $SEED"

# Execute the command
eval $CMD 2>&1 | tee full_output.log
```

2. Next run the server `bash run_qwen3vl_server.sh`

And send a requests to check the inference like that


```python
# Image inference
import time
from pathlib import Path
import openai
import base64
client = openai.Client(
    base_url="http://localhost:6655/v1", api_key="EMPTY"
)
MODEL = "qwen_deployed_model"

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
  # params from here https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct#vl
  max_tokens=256,
  top_p=0.8,
  extra_body={
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5
  },
  stream=True,
  temperature=0.7,

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

### Eval

Few scripts for evaluation attached, they are inside LLaMA-Factory folder. `./eval` len obisnyat, but there is readme and it's straightforward you can do it!

## Training 

Setup trahing.

### 0. Setup the env 

```bash
uv venv --python 3.11 --seed .venv
uva

git clone git@github.com:vladimiralbrekhtccr/oylan_3_llama_factory.git
cd oylan_3_llama_factory
uv pip install -e ".[torch,metrics]" --no-build-isolation
uv pip install deepspeed==0.16.9
uv pip install wandb
uv pip install liger-kernel
uv pip install git+https://github.com/vladimiralbrekhtccr/oylan_3_transformers.git
```

>> Setup the `.env_example` file change values and rename it to `.env`.


### 1. Run a test training using.

Open main repository folder and run it will run on 1 GPU training on test dataset.

```bash
llamafactory-cli train ./examples/train_full/qwen3vla_4b/example_of_training.sh
```



### 2. Run actual train on your data.

Data part:

- You will only need to change: `dataset: test_audio_asr` to your data. 

Your data should be in such structure (1 sample) sharegpt format. 

```json
{
    "audios": [
      "./data/test_audio_data/audios/commonvoice_ru_clips_16k_wav_common_voice_ru_38136195.wav"
    ],
    "messages": [
      {
        "role": "user",
        "content": "<audio>\nПереведите услышанную речь в текст."
      },
      {
        "role": "assistant",
        "content": "В мае этого года Фиджи была оказана честь быть принятой в члены Движения неприсоединения."
      }
    ]
}
```

Please check the meta file where we write all our datasets it's `./data/dataset_info.json`.

For example test case is written this way.

where we specify our file_nam path and formatting.

[!IMPORTANT] Make sure to use `<audio>\n` in the user message. Like that |"content": "<audio>\nПереведите услышанную речь в текст."|

```json
  "test_audio_asr": {
    "file_name": "test_audio_data/test_audio_asr.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "audios": "audios"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
```

- Create a new training config based on `./examples/train_full/qwen3vla_4b/train_qwen3_vla_4B_audio.yaml`

```yaml
# only change the 2 values about path and wandb run name:
# output_dir: ./output/qwen3vla_mlp_tests/qwen3vla_4B_audio_ASR_original_whisper_packing # @CHANGABLE
# run_name: qwen3vla_4B_audio_ASR_original_whisper_packing  # Add this


model_name_or_path: issai/Qwen3_VLA_4B_whisper_original_init_thinking
image_max_pixels: 50176
image_min_pixels: 784
video_max_pixels: 16384
trust_remote_code: true
flash_attn: fa2


### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true # vision encoder 
freeze_audio_tower: true # audio encoder 
freeze_audio_projector: false # audio mlp
freeze_multi_modal_projector: true # vision mlp
freeze_language_model: true # llm
deepspeed: examples/deepspeed/ds_z3_config.json


## all datasets
dataset: test_audio_asr 
cache_dir: ./cache/audios_data_full_test


# with thinking data use `qwen3_vl_audio`
# with no_think data use `qwen3_vl_audio_nothink`
template: qwen3_vl_audio # <<!>>
cutoff_len: 4096 # Cutoff length for the dataset
max_samples: 100000000
overwrite_cache: false
preprocessing_num_workers: 16
dataloader_num_workers: 4
data_seed: 42
mask_history: false # when you want to train on the last_turn only you can set mask_history==true.


group_by_length: true # useful to use when packing is false, will speed up the training
packing: false # TODO: calcualte the amount of tokens in text/vision/audio datasets. FIX: seems like for audio training is not stable with packing==true

### output
output_dir: ./output/qwen3vla_mlp_tests/qwen3vla_4B_audio_ASR_original_whisper_thinking
logging_steps: 1
save_steps: 500
save_strategy: steps
save_total_limit: 10
overwrite_output_dir: true
report_to: wandb
run_name: qwen3vla_4B_audio_ASR_original_whisper_thinking


### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 2.0e-4 # !NOTE: worked well with Qwen3VL_30B_A3 but not tested for 4B
num_train_epochs: 1.0
lr_scheduler_type: cosine
enable_liger_kernel: true
gradient_checkpointing: true

resume_from_checkpoint: null

max_grad_norm: 1.0
bf16: true
weight_decay: 0.05
warmup_ratio: 0.03
ddp_timeout: 180000000
```

- [!NOTE]: You can take model from here `issai/Qwen3_VLA_4B_whisper_original_init_thinking` it's already with original_whisper as audio encoder. In case you want to download it to the local folder, otherwise it will automatically downloaded when you run training.


Create a train bash script `<your_training_bash_script>.sh`

```bash
# export HF_DATASETS_DISABLE_LOCKING=1
unset HF_DATASETS_DISABLE_LOCKING
export WANDB_PROJECT="qwen3VL_finetune_llama_factory"

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$dir" != "/" ]]; do
  if [ -f "$dir/.env" ]; then
    set -a; source "$dir/.env"; set +a
    echo "✓ Loaded .env from: $dir"
    break
  fi
  dir="$(dirname "$dir")"
done



llamafactory-cli train ./examples/train_full/qwen3vla_4b/<your_train_config>.yaml

### FOR multi-node
# export HF_DATASETS_DISABLE_LOCKING=1
# export NCCL_TIMEOUT=72000              
# export NCCL_SOCKET_TIMEOUT=72000       # Socket timeout
# export INIT_PROCESS_GROUP_TIMEOUT=72000 
# export NCCL_BLOCKING_WAIT=1           # Block until completion
# # --- Load Environment ---
# export WANDB_PROJECT="qwen3VL_finetune_llama_factory"

# dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# while [[ "$dir" != "/" ]]; do
#   if [ -f "$dir/.env" ]; then
#     set -a; source "$dir/.env"; set +a
#     echo "✓ Loaded .env from: $dir"
#     break
#   fi
#   dir="$(dirname "$dir")"
# done

# FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=0 MASTER_ADDR=10.141.0.1 MASTER_PORT=29777 llamafactory-cli train /scratch/vladimir_albrekht/qwen3_vl_moe/llama_factory/<your_train_config>.yaml
```

And start the training with `bash ./examples/train_full/qwen3vla_4b/<your_training_bash_script>.sh`

### 4. Now you can start the training and receive some great result yria yria yria. hehe

If will log everyhting to wandb so you will be able to see something like that

![alt text](511.png)



Now try to bit my super Kita4B_VLA model with your dirty thinking data.