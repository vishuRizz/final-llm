
## Mini Decoder-Only Transformer – Study Notes

This project implements a small decoder-only Transformer in PyTorch, step by step.  
Below are the main topics, formulas, and terms you can use to learn the theory.

---

## 1. Token Embeddings

- **Idea**: Map discrete tokens (IDs) to continuous vectors so the model can work with them.
- **Function**: \( E: V \rightarrow \mathbb{R}^d \) where:
  - \( V \): vocabulary set
  - \( d \): embedding dimension
- **Embedding matrix**:
  - \( \mathbf{W}_{emb} \in \mathbb{R}^{|V| \times d} \)
  - If token \( t_i \) has ID \( i \), then its embedding is row \( i \) of \( \mathbf{W}_{emb} \).
- **Key terms**:
  - **Vocabulary** \(|V|\)
  - **Embedding dimension** \(d\)
  - **Embedding matrix / lookup table**

---

## 2. Sinusoidal Positional Encoding

Transformers are position-agnostic, so we inject order information.

- **Formulas** for position `pos` and dimension index \( i \):
  - For even dimensions \( 2i \):
    \[
    \text{PE}_{pos, 2i} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
    \]
  - For odd dimensions \( 2i+1 \):
    \[
    \text{PE}_{pos, 2i+1} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
    \]
- **Terms**:
  - \( d_{\text{model}} \): model (embedding) dimension
  - **max\_len**: maximum sequence length used to precompute PE
  - **Wavelengths / frequencies**: controlled by \( 10000^{2i/d_{\text{model}}} \)
- **Why sinusoidal**:
  - Encodes **absolute** and **relative** positions.
  - Can generalize to **longer sequences** via trigonometric identities.

---

## 3. Self-Attention and Scaled Dot-Product Attention

Given Query \( Q \), Key \( K \), Value \( V \):

- **Scaled dot-product attention**:
  \[
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
  \]
  - \( d_k \): dimension of keys/queries.
  - \( Q \in \mathbb{R}^{n \times d_k} \), \( K \in \mathbb{R}^{m \times d_k} \), \( V \in \mathbb{R}^{m \times d_v} \).
- **Interpretation**:
  - \( QK^\top \): similarity scores between queries and keys.
  - division by \( \sqrt{d_k} \): stabilizes gradients.
  - **softmax**: converts scores to a probability distribution over values.

---

## 4. Multi-Head Self-Attention (MHSA)

Instead of a single attention, use multiple heads to look at different “subspaces”.

- **Per head \( j \)**:
  \[
  Q_j = X W_j^Q,\quad K_j = X W_j^K,\quad V_j = X W_j^V
  \]
  - \( W_j^Q, W_j^K \in \mathbb{R}^{d_{\text{model}} \times d_k} \)
  - \( W_j^V \in \mathbb{R}^{d_{\text{model}} \times d_v} \)
- **Head output**:
  \[
  \text{head}_j = \text{Attention}(Q_j, K_j, V_j)
  \]
- **Concatenation and output projection**:
  \[
  \text{MultiHead}(Q, K, V) =
  \text{Concat}(\text{head}_1, …, \text{head}_h) W^O
  \]
  - \( W^O \in \mathbb{R}^{(h \cdot d_v) \times d_{\text{model}}} \)
- **Typical choice**:
  - \( d_k = d_v = d_{\text{model}} / h \) (so concatenation returns to \( d_{\text{model}} \)).

**Key terms**:
- \( h \): number of heads
- \( d_{\text{model}} \): model dimension
- **head\_dim**: \( d_{\text{model}} / h \)

---

## 5. Masked (Causal) Self-Attention

Used in **decoder-only** models so a position cannot see the future.

- **Causal mask**: lower-triangular matrix  
  - For sequence length \( L \):
    - Mask \( M \in \{0,1\}^{L \times L} \)
    - \( M_{ij} = 1 \) if \( j \le i \), else \( 0 \).
- **Apply mask to scores**:
  \[
  \text{Scores}_{\text{masked}} =
  \frac{QK^\top}{\sqrt{d_k}} + M'
  \]
  where \( M'_{ij} = 0 \) for allowed positions and \(-\infty\) (or a very negative number) for disallowed ones.
- In code: `masked_fill(mask == 0, float('-inf'))` then softmax.

**Effect**: token at position \( i \) only attends to positions \( \le i \).

---

## 6. Layer Normalization (LayerNorm)

Normalizes features **per token** (not across batch).

- For an input vector \( \mathbf{x} = (x_1, \dots, x_D) \):
  - Mean:
    \[
    \mu = \frac{1}{D} \sum_{i=1}^D x_i
    \]
  - Variance:
    \[
    \sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2
    \]
  - Normalized output:
    \[
    \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
    \]
  - With learnable scale and bias:
    \[
    y_i = \gamma \hat{x}_i + \beta
    \]
- **Terms**:
  - \( \gamma \): learnable scale
  - \( \beta \): learnable shift
  - \( \epsilon \): small constant for numerical stability
  - **normalized\_shape** in PyTorch: which dimensions to normalize over (usually `d_model`).

---

## 7. Position-wise Feed-Forward Network (FFN)

Applies the same MLP to each position independently.

- **Structure**:
  \[
  \text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
  \]
  - \( x \in \mathbb{R}^{d_{\text{model}}} \)
  - \( W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}} \)
  - \( W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}} \)
  - \( d_{ff} \): hidden (expanded) dimension, often \( 4 \cdot d_{\text{model}} \)
- **Activation**: ReLU in your implementation (GELU is also common).

---

## 8. Residual Connections (“Skip Connections”) and Add & Norm

Help gradients flow and allow deeper networks.

- **Basic residual**:
  \[
  y = x + \text{sublayer}(x)
  \]
- **Add & Norm pattern in Transformers**:
  \[
  \text{output} = \text{LayerNorm}(x + \text{Dropout}(\text{sublayer}(x)))
  \]
- In your notebook this is wrapped in a `ResidualConnection` module.

**Key ideas**:
- Information can flow around a sublayer if it is not helpful.
- Stabilizes training of deep stacks of attention + FFN blocks.

---

## 9. Decoder Block

A single Transformer **decoder block** in your model contains:

1. **Masked multi-head self-attention**  
2. **Add & Norm** (residual + LayerNorm)  
3. **Feed-Forward Network (FFN)**  
4. **Add & Norm** again  

Shapes:
- Input/output of block: \((\text{batch\_size}, \text{seq\_len}, d_{\text{model}})\).

---

## 10. Final Linear Output Layer

Maps hidden states to vocabulary logits for prediction.

- For each position:
  - Input: \( h \in \mathbb{R}^{d_{\text{model}}} \)
  - Output logits:
    \[
    z = h W_{\text{out}} + b_{\text{out}}
    \]
    with \( W_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times |V|} \).
- Then usually:
  - **softmax** over vocabulary dimension to get probabilities.
  - **Cross-entropy loss** during training.

---

## 11. Full Decoder-Only Transformer

Your `DecoderOnlyTransformer` composes everything:

1. **TokenEmbedding**: token IDs → embeddings  
2. **PositionalEncoding**: add sinusoidal positions  
3. **Dropout** on embeddings  
4. **Stack of DecoderBlocks** (each with masked MHSA + FFN + residuals + LayerNorm)  
5. **FinalLinearOutput**: hidden states → vocabulary logits  
6. **Causal mask** built from `torch.tril` to enforce autoregressive behavior  

Input shape:
- Token IDs: \((\text{batch\_size}, \text{seq\_len})\)

Output shape:
- Logits: \((\text{batch\_size}, \text{seq\_len}, |V|)\)

---

## 12. Quick Reference – Symbols and Terms

- \( V \): vocabulary, \(|V|\) its size  
- \( d \), \( d_{\text{model}} \): embedding / model dimension  
- \( d_k \), \( d_v \): key/value dimensions per head  
- \( d_{ff} \): FFN hidden dimension  
- \( h \): number of attention heads  
- **Q, K, V**: Query, Key, Value matrices  
- **PE**: positional encoding  
- **MHSA**: Multi-Head Self-Attention  
- **LayerNorm**: layer normalization  
- **FFN**: feed-forward network  
- **Causal mask**: lower-triangular mask for decoder  

You can cross-check each of these sections with the corresponding code cells in `main.ipynb` to connect the math with the implementation.
