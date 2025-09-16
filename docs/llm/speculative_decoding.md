# Speculative decoding MTP in Qwen3-Next


> ![Note] @AI_content writen by Gemini 2.5 Pro with context about Qwen3-Next MTP
> Context of implementation in vLLM 0.10.2 was provided + model architecture.

### **Technical Summary: Speculative Decoding in Qwen3-Next (MTP Method)**

**Document Date:** September 16, 2025

#### **1. High-Level Concept: Overcoming Latency**

The primary goal of speculative decoding is to accelerate inference speed by reducing the number of slow, expensive forward passes required by the main model. Large Mixture-of-Experts (MoE) models like Qwen3-Next are powerful but have high per-token latency.

The MTP method uses a "Professor & Student" analogy:

* **The Main Model (Professor):** The full, deep (e.g., 48-layer) Qwen3-Next model. It is extremely accurate but slow.
* **The MTP Drafter (Student):** A specialized, fast predictor. It generates a "draft" of multiple future tokens which the Professor then proofreads all at once.

#### **2. Architecture of the MTP Drafter**

Based on the model's weight files, the MTP Drafter is not a simple MLP. It is a complete, self-contained **`Qwen3NextDecoderLayer`**.

* **Structure:** It has the identical architecture to a single layer of the main model, including:
    * A `self_attn` block (`Qwen3NextAttention`).
    * An `mlp` block which is also a `Qwen3NextSparseMoeBlock` (with its own router, experts, and shared expert).
    * Standard `input_layernorm` and `post_attention_layernorm`.
* **Why it's Fast:** Its speed comes from its extreme **shallowness**. The generation of a single draft token requires a pass through only **one** of these complex layers, whereas the main model requires a pass through **48** of them.

#### **3. The End-to-End Generation Pipeline**

This is the complete process for generating a draft of `k` tokens (e.g., `k=2`).

**Phase 1: Initial Context Generation (1 Expensive Pass)**

1.  **Input:** The initial prompt (e.g., `"The cat sat"`).
2.  **Process:** The prompt is fed through the **Main Model** (all 48 layers).
3.  **Output:** The model produces the final hidden state from its last layer for the last token ("sat"). We'll call this `H_main`. This is the "thought vector" or rich context for what comes next.

**Phase 2: MTP Drafting Loop (Fast Iterative Process)**

This phase is executed by the MTP Drafter and is computationally cheap.

4.  **Draft Token 1:**
    * **Input:** The original `H_main` is concatenated with the embedding of the last known token ("sat"). Shape `[1, hidden_size * 2]`.
    * **Projection:** This double-sized tensor is passed through the `mtp.fc` linear layer to project it back down to `hidden_size`.
    * **Process:** The result is passed through the single `mtp.layers.0` block.
    * **Output:** A new hidden state, `H_spec_1`.
    * **Sampling:** The system calculates logits (`lm_head(H_spec_1)`) and samples `token_1` (e.g., `"on"`).

5.  **Draft Token 2:**
    * **Input:** The **original** `H_main` is concatenated with the embedding of the newly drafted `token_1` ("on").
    * **Projection & Process:** The same projection and `mtp.layers.0` block are used again.
    * **Output:** A new hidden state, `H_spec_2`.
    * **Sampling:** The system calculates logits (`lm_head(H_spec_2)`) and samples `token_2` (e.g., `"the"`).

6.  **Result of Drafting:** The phase ends with a candidate sequence: `Draft = {"on", "the"}`.

**Phase 3: Verification & Acceptance (1 Expensive Pass)**

7.  **Input:** The full sequence `prompt + Draft` (e.g., `"The cat sat on the"`) is fed into the **Main Model**.
8.  **Process:** The Main Model runs a single forward pass, calculating the "correct" hidden states and logits for every token position in parallel.
9.  **Comparison:** The system compares its draft with the main model's verified outputs:
    * It checks if `token_1` ("on") is what the main model would have generated after "sat".
    * If yes, it continues. It checks if `token_2` ("the") is what the main model would have generated after "on".
10. **Outcome:**
    * **Full Acceptance:** If all tokens match, the new sequence becomes `"The cat sat on the"`.
    * **Partial Acceptance:** If only `token_1` matched, the new sequence becomes `"The cat sat on"`.
    * **Full Rejection:** If not even `token_1` matched, the system discards the entire draft. It salvages the correct first token (which it calculated during this verification pass anyway) and proceeds.

#### **4. Cost-Benefit Analysis: The "Profit"**

* **Standard Method Cost:** To generate `N` tokens, it requires **`N`** expensive passes.
* **Speculative Method Cost:** To generate `N` tokens (assuming full acceptance), it requires **`2`** expensive passes (1 Initial + 1 Verification).

* **Profit (Passes Saved):** `Profit = N - 2`
    * For 10 accepted tokens, the profit is `10 - 2 = 8` saved passes.

* **Worst-Case Cost (0 Accepted Tokens):**
    * The standard method would have taken **1** pass to get the one correct token.
    * The speculative method took **2** passes to get that same one correct token.
    * **Loss (Wasted Passes):** `Loss = 1` pass.

The system is profitable as long as the average number of accepted tokens per cycle is greater than 1.