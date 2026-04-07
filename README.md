# GPT-2 From Scratch (124M) 

## Overview

This project walks through building a **GPT-2 style language model (124M parameters)** completely from scratch and using it to **generate text autoregressively**.

It follows two major stages:

1. **Model Construction** — Implementing the GPT-2 architecture
2. **Text Generation** — Using the model to predict and generate tokens

---

## Video Overview

### 1. Understanding Transformers

[![Understanding Transformers](https://img.youtube.com/vi/3QlcY-cV1sY/0.jpg)](https://youtu.be/3QlcY-cV1sY)


### 2. GPT-2 From Scratch 

[![Understanding the GPT 2 Architecture](https://img.youtube.com/vi/4lwlkSbIBEI/0.jpg)](https://youtu.be/4lwlkSbIBEI)

---

## Architecture Overview

GPT-2 is a **Transformer-based autoregressive language model**.

### Core Components

* **Token Embeddings** — Convert token IDs into dense vectors
* **Positional Embeddings** — Encode sequence order information
* **Transformer Blocks** (stacked multiple times):

  * Multi-Head Self-Attention
  * Feedforward Neural Network (MLP)
  * Layer Normalization
  * Residual Connections
* **Final Linear Layer (LM Head)** — Projects hidden states to vocabulary logits

---

## Forward Pass Flow

```text
Input Tokens
   ↓
Token Embeddings + Positional Embeddings
   ↓
Transformer Block × N
   ↓
LayerNorm
   ↓
Linear Projection (Logits)
   ↓
Softmax → Probabilities
```

---

## Model Configuration (124M GPT-2)

Typical configuration includes:

* Layers: 12
* Heads: 12
* Embedding size: 768
* Context length: 1024
* Parameters: ~124 million

---

## Transformer Block Details

Each transformer block contains:

### 1. Multi-Head Self-Attention

* Computes relationships between tokens
* Uses queries, keys, and values
* Applies causal masking (no future tokens)

### 2. Feedforward Network (MLP)

* Expands hidden dimension (usually 4×)
* Applies non-linear activation (e.g., GELU)

### 3. Residual Connections

* Helps stabilize training
* Allows gradient flow across layers

### 4. Layer Normalization

* Applied before or after sublayers
* Keeps activations stable

---

## Autoregressive Text Generation

After building the model, text generation works as follows:

### Step-by-Step Process

1. Start with an initial prompt
2. Feed tokens into the model
3. Get logits for next token
4. Convert logits → probabilities
5. Sample or select next token
6. Append token to sequence
7. Repeat

---

## Generation Loop (Pseudo Code)

```python
for _ in range(max_new_tokens):
    logits = model(input_tokens)
    next_token_logits = logits[:, -1, :]
    probs = softmax(next_token_logits)
    next_token = sample(probs)
    input_tokens = concat(input_tokens, next_token)
```

---

## Sampling Strategies

Different methods affect output quality:

* **Greedy** — Always pick highest probability (deterministic)
* **Sampling** — Randomly sample from distribution
* **Top-k Sampling** — Choose from top k tokens
* **Top-p (Nucleus)** — Choose from smallest set of tokens with cumulative probability p

---

## Key Concepts

### Autoregression

The model predicts one token at a time based on previous tokens.

### Causal Masking

Prevents tokens from attending to future positions.

### Logits vs Probabilities

* Logits = raw outputs
* Probabilities = softmax(logits)

---


## Note

* This implementation focuses on **understanding**, not optimization
* Training is not covered in detail here

