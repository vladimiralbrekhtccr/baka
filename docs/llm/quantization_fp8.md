How to quantizee a Qwen3-Next-80B-A3B-Instruct model

```python
import os
os.environ["CUDA_HOME"] = "/home/vladimir_albrekht/miniconda3/envs/cuda12"
os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CUDA_HOME']}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:{os.environ.get('PATH', '')}"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "/home/vladimir_albrekht/projects/10_09_2025_MOe/models/Qwen3-Next-80B-A3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="cpu", torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC",ignore=['lm_head', 're:.*mlp.gate$', 're:.*shared_expert_gate$', 're:.*router$'])

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)
```