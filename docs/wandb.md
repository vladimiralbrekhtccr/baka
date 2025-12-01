
# How to upload a file to wandb

```python
import os 
os.environ['WANDB_API_KEY']=''
import wandb

# 1. Initialize the API
api = wandb.Api()
run = api.run("mamubieke-parehati-ISSAI/qwen3VL_finetune_llama_factory/runs/ixvgv2fr")

# Define your full path
full_path = "/scratch/vladimir_albrekht/qwen3_vl_moe/llama_factory/LLaMA-Factory/examples/train_full/oylan_3/qwen3vl_vision/train_qwen3_vl_30B_vision.yaml"

# FIX: Set 'root' to the directory containing the file. 
# This tells W&B: "Only upload the filename, don't create the folders /scratch/vladimir/..."
run.upload_file(full_path, root=os.path.dirname(full_path))
```