[Good source](https://roadmap.sh/r/ml-engineer-3dqvu)

# ML
1. ViT
    * official
    * 
    * 
2. Self-attention
3. Cross-attention
3. 

# Example of what actually necessary.

    # 1. Matrix multiplication (you use this everywhere)
    Q @ K.transpose(-1, -2)  # What's actually happening here?

    # 2. Broadcasting rules
    attention_scores = dots + attention_mask  # Why does this work?

    # 3. Tensor reshaping
    rearrange(x, 'b n (h d) -> b h n d', h=heads)  # What's the shape math?

    # 4. Eigenvalues/eigenvectors (for understanding)
    # Why does layer normalization stabilize training?
    # How does gradient clipping relate to matrix norms?