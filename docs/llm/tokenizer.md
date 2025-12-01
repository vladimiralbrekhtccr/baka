# Simple example how to tokenizer text

```python
from transformers import AutoTokenizer
MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

TEXT = """
What is the meaning of life, when you can't watch anime?
"""

encoded = tokenizer.encode(TEXT)
token_count = len(encoded)
print(f"Tokens: {token_count}")

for i in encoded:
    print(f"{i}: {repr(tokenizer.decode([i]))}")
```