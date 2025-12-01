# Qwen3VL training for object detection.

Simple [demo](https://huggingface.co/spaces/dmorawiec/Qwen-VL-Object-Detection) on HF where you can check how Current Qwen3VL models works for object_detection and classification. 

Their codebase is open-source so you can use it as a reference.

Example:
![alt text](5214.png)


## Inference 

For the inference of the model I suggest to use the vLLM framework, it's relatively easy to start with and State-of-the-art for inference. 

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

## Training 

For the training you can use [Llama-factory](https://github.com/hiyouga/LLaMA-Factory) framework. 

Setup for training might be something like that:

0. Setup the env 

```bash
uv venv --python 3.11 --seed .venv
uva

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
uv pip install -e ".[torch,metrics]" --no-build-isolation
uv pip install deepspeed==0.16.9
uv pip install wandb
uv pip install liger-kernel
uv pip install transformers==4.57.1
```


1. Create a file `start_train_qwen3vl_4b_v1.sh`

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



llamafactory-cli train ./llama_factory/train_qwen3_vl_30B_test.yaml

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

# FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=0 MASTER_ADDR=10.141.0.1 MASTER_PORT=29777 llamafactory-cli train /scratch/vladimir_albrekht/qwen3_vl_moe/llama_factory/train_qwen3_vl_30B.yaml
```

2. Create a config for training `qwen3vl_4b_object_class_v1.yaml`


```yaml
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
image_max_pixels: 50176
image_min_pixels: 784
video_max_pixels: 16384
trust_remote_code: true
flash_attn: fa2


### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: false
freeze_language_model: false
deepspeed: examples/deepspeed/ds_z3_config.json 

### dataset
dataset: captions_dataset_1,captions_dataset_2 # .etc you can check inside /LLaMA-Factory/data/dataset_info.json
cache_dir: ./cache/images_data_test


template: qwen3_vl_nothink # [qwen3_vl](by default it's with thinking) or [qwen3_vl_nothink] --> {./LLaMA-Factory/src/llamafactory/data/template.py line 1933} make sure to specify this one correct #IMPORTANT
cutoff_len: 8192 # Cutoff length for the dataset
max_samples: 100000000
overwrite_cache: false
preprocessing_num_workers: 16
dataloader_num_workers: 4
data_seed: 42

group_by_length: true
packing: true

### output
output_dir: ./output/qwen3vl_4b_adema_run # @CHANGABLE
logging_steps: 1
save_steps: 100 # @CHANGABLE
save_strategy: steps
save_total_limit: 10
overwrite_output_dir: true
report_to: wandb  # or tensorboard
run_name: qwen3vl_4b_adema_run


### train
per_device_train_batch_size: 1 # @CHANGABLE
gradient_accumulation_steps: 4 # @CHANGABLE
learning_rate: 8.0e-5 # @CHANGABLE
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

3. Inside LLaMA-Factory folder create `.env` with a HF_TOKEN like that and specify your WANDB_API_KEy

```bash
WANDB_API_KEY=1630oqkweoqkeoqwkeqwpelqwpelqwpelqwelp
```

4. Now you can start the training and receive some great result yria yria yria. hehe

If will log everyhting to wandb so you will be able to see something like that

![alt text](511.png)

## Some possible considerations for your case

You can use `instruct or thinking` model. 

Qwen has 2 of them:

1. Qwen/Qwen3-VL-4B-Instruct
2. Qwen/Qwen3-VL-4B-Thinking

And thinking usually provide better performance but output will be not that fast as if you were using instruct model and also dataset might be the problem because thinking_captioning dataset is something new.