# Mixture of experts

## Components of Moe

1. Sparse MoE Layers instead of Feed-forward (FFN)

2. Gate network or router - determines which tokens are sent to which expert.


Qwen3 pipeline closely mirrors DeepSeek R1
That means DeepSeek R1 paper is all you need

Question:
- In training what is verifiable rewards ?




## Qwen3Moe-Next
[qwen blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)

> [!Note]
> **Switch to seperate think/no_think models** why? It's some kind of new trend hmm.
> ** **


Qwen3-Next innovations:
- hybrid attention mechanism
- highly sparse (MoE)
- training-stability-friendly optimizations
- multi-token prediction mechanism


Model architecture:

```
amount of layers = 48 [ 32 -delta net, 
                        12 std attention]
                        
Qwen3NextForCausalLM(
  (model): Qwen3NextModel(
    (embed_tokens): Embedding(151936, 2048)
    (layers): ModuleList(
      (0-2): 3 x Qwen3NextDecoderLayer(
        (linear_attn): Qwen3NextGatedDeltaNet(
          (act): SiLU()
          (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
          (in_proj_qkvz): Linear(in_features=2048, out_features=12288, bias=False)
          (in_proj_ba): Linear(in_features=2048, out_features=64, bias=False)
          (norm): Qwen3NextRMSNormGated()
          (out_proj): Linear(in_features=4096, out_features=2048, bias=False)
        )
        (mlp): Qwen3NextSparseMoeBlock(
          (gate): Linear(in_features=2048, out_features=512, bias=False)
          (experts): ModuleList(
            (0-511): 512 x Qwen3NextMLP(
              (gate_proj): Linear(in_features=2048, out_features=512, bias=False)
              (up_proj): Linear(in_features=2048, out_features=512, bias=False)
              (down_proj): Linear(in_features=512, out_features=2048, bias=False)
              (act_fn): SiLU()
            )
          )
          (shared_expert): Qwen3NextMLP(
            (gate_proj): Linear(in_features=2048, out_features=512, bias=False)
            (up_proj): Linear(in_features=2048, out_features=512, bias=False)
            (down_proj): Linear(in_features=512, out_features=2048, bias=False)
            (act_fn): SiLU()
          )
          (shared_expert_gate): Linear(in_features=2048, out_features=1, bias=False)
        )
        (input_layernorm): Qwen3NextRMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen3NextRMSNorm((2048,), eps=1e-06)
      )
      (3): Qwen3NextDecoderLayer(
        (self_attn): Qwen3NextAttention(
          (q_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2048, bias=False)
          (q_norm): Qwen3NextRMSNorm((256,), eps=1e-06)
          (k_norm): Qwen3NextRMSNorm((256,), eps=1e-06)
        )
        (mlp): Qwen3NextSparseMoeBlock(
          (gate): Linear(in_features=2048, out_features=512, bias=False)
          (experts): ModuleList(
            (0-511): 512 x Qwen3NextMLP(
              (gate_proj): Linear(in_features=2048, out_features=512, bias=False)
              (up_proj): Linear(in_features=2048, out_features=512, bias=False)
              (down_proj): Linear(in_features=512, out_features=2048, bias=False)
              (act_fn): SiLU()
            )
          )
          (shared_expert): Qwen3NextMLP(
            (gate_proj): Linear(in_features=2048, out_features=512, bias=False)
            (up_proj): Linear(in_features=2048, out_features=512, bias=False)
            (down_proj): Linear(in_features=512, out_features=2048, bias=False)
            (act_fn): SiLU()
          )
          (shared_expert_gate): Linear(in_features=2048, out_features=1, bias=False)
        )
        (input_layernorm): Qwen3NextRMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen3NextRMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen3NextRMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen3NextRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
```

- 80B-A3B
- 


Comparison:

- performnace of Qwen3-32B model, using less than 10% of its training cost (GPU hours)
- In inference, especially with context lengths over 32K tokens, it delivers more than 10x higher throughput — achieving extreme efficiency in both training and inference.
