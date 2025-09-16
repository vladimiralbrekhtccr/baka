
## Default


> ![Note] newgrp docker
```bash

export VLLM_USE_V1=1
# Model and environment settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="huggingface-org/Oylan2_5-MLP-sft-10k"

export WHISPER_MODEL_PATH="$MODEL"
MODEL_SERVED_NAME="kita" 
PORT=6655
HOST="0.0.0.0"
SEED=0

# vLLM configuration parameters
GPU_MEMORY_UTILIZATION=0.90 # 80 is fine
TENSOR_PARALLEL_SIZE=4 # changable
DTYPE="bfloat16"
MAX_NUM_BATCHED_TOKENS=32768 # 32768 vs 4096
MAX_MODEL_LEN=4096
KV_CACHE_DTYPE="auto"
BLOCK_SIZE=32 
SWAP_SPACE=0
MAX_NUM_SEQS=5


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
  --swap-space $SWAP_SPACE \
  --block-size $BLOCK_SIZE \
  --kv-cache-dtype $KV_CACHE_DTYPE \
  --max-num-seqs $MAX_NUM_SEQS \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --seed $SEED"

# Execute the command
eval $CMD 2>&1 | grep -v -E "'WhisperTokenizer'|'Qwen2Tokenizer'|unexpected tokenization|'WhisperTokenizerFast'"
```


## Docker run

```bash

### Model and environment settings
MODEL="/home/vladimir_albrekht/projects/digital_bridge/models/Oylan2_5-MLP-sft-10k"

MODEL_SERVED_NAME="oylan_a_v_t_2_5" 
PORT=6655
HOST="0.0.0.0"
SEED=0

# vLLM configuration parameters
GPU_MEMORY_UTILIZATION=0.90 # 80 is fine
TENSOR_PARALLEL_SIZE=4 # changable
DTYPE="bfloat16"
MAX_NUM_BATCHED_TOKENS=32768 # 32768 vs 4096
MAX_MODEL_LEN=4096
KV_CACHE_DTYPE="auto"
BLOCK_SIZE=32 
SWAP_SPACE=0
MAX_NUM_SEQS=5


IMAGE_NAME="vllm-oylan-2_5:0.9.2-cu124"
docker run --gpus all \
  -v "$MODEL":"$MODEL":ro \
  --env WHISPER_MODEL_PATH="$MODEL" \
  -e VLLM_USE_V1=1 \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
    -p ${PORT}:${PORT} \
  --name $MODEL_SERVED_NAME \
  --ipc=host \
  $IMAGE_NAME \
  --port "$PORT" \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --swap-space $SWAP_SPACE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --max-model-len $MAX_MODEL_LEN \
  --block-size $BLOCK_SIZE \
  --seed 0 \
  --dtype $DTYPE \
  --max-num-seqs $MAX_NUM_SEQS \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --served-model-name $MODEL_SERVED_NAME

# -d if you want to run in the background
```

## With lora adapter Docker run

```bash

### Model and environment settings
MODEL="/home/vladimir_albrekht/projects/digital_bridge/models/Oylan2_5-MLP-sft-10k"

MODEL_SERVED_NAME="oylan_a_v_t_2_5" 
PORT=6655
HOST="0.0.0.0"
SEED=0

# vLLM configuration parameters
GPU_MEMORY_UTILIZATION=0.90 # 80 is fine
TENSOR_PARALLEL_SIZE=4 # changable
DTYPE="bfloat16"
MAX_NUM_BATCHED_TOKENS=32768 # 32768 vs 4096
MAX_MODEL_LEN=4096
KV_CACHE_DTYPE="auto"
BLOCK_SIZE=32 
SWAP_SPACE=0
MAX_NUM_SEQS=5

# LORA
ADAPTER_1="/home/vladimir_albrekht/projects/digital_bridge/models/adapters/remaped/vision_lora_3k_Oylan2-5_MLP_SFT_10k_remap"
# --enable-lora \
LORA_DTYPE="bfloat16"
MAX_LORA_RANK=256
MAX_LORAS=2


IMAGE_NAME="vllm-oylan-2_5:0.9.2-cu124"
docker run --gpus all \
  -v "$MODEL":"$MODEL":ro \
  -v "$ADAPTER_1":"$ADAPTER_1":ro \
  --env WHISPER_MODEL_PATH="$MODEL" \
  -e VLLM_USE_V1=1 \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
    -p ${PORT}:${PORT} \
  --name $MODEL_SERVED_NAME \
  --ipc=host \
  $IMAGE_NAME \
  --port "$PORT" \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --enable-lora \
  --lora-modules oylan_2_5_vision_lora=$ADAPTER_1 \
  --lora-dtype $LORA_DTYPE \
  --max_lora_rank $MAX_LORA_RANK \
  --max-loras $MAX_LORAS \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --swap-space $SWAP_SPACE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --max-model-len $MAX_MODEL_LEN \
  --block-size $BLOCK_SIZE \
  --seed 0 \
  --dtype $DTYPE \
  --max-num-seqs $MAX_NUM_SEQS \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --served-model-name $MODEL_SERVED_NAME

# -d if you want to run in the background
```
