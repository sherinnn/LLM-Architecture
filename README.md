# LLM Transformer Block Implementation from Scratch

This project focuses on coding the entire **Transformer block**, the fundamental building block and "engine" behind large language models like GPT. This implementation is part of a series on building GPT architecture from the ground up, specifically targeting the **gpt2 (124M parameter)** configuration.

## Project Overview

The Transformer block is designed to process input sequences (converted into vector embeddings) and output **context vectors** of the same dimensionality. This preservation of dimensionality allows for multiple blocks (12 in gpt2) to be stacked effectively to form a deep neural network.

### Key Components

The Transformer block integrates five core sub-components:

1.  **Layer Normalization:** Applied before the attention and feed-forward layers (Pre-layer Norm) to ensure training stability, prevent exploding/vanishing gradients, and solve internal covariate shift.
2.  **Masked Multi-Head Attention:** Converts embedding vectors into context vectors by capturing the relationships between different tokens in a sequence.
3.  **Feed-Forward Neural Network (FFN):** A two-layer network that expands the input dimension by 4x to explore a richer parameter space before compressing it back to the original size.
4.  **GELU Activation:** A smooth, differentiable variation of ReLU used within the FFN to solve the "dead neuron" problem.
5.  **Shortcut (Skip) Connections:** Adds the input of a layer back to its output to facilitate stable gradient flow and solve the vanishing gradient problem.
6.  **Dropout:** Randomly sets a percentage of neurons (10% in this config) to zero during training to prevent overfitting and improve generalization.

## Architecture Configuration

The implementation follows the **gpt2-small** specifications:

*   **Embedding Dimension:** 768
*   **Context Length:** 1024 tokens
*   **Attention Heads:** 12
*   **Transformer Layers:** 12
*   **Dropout Rate:** 0.1 (10%)
*   **Vocabulary Size:** 50,257

## Technical Implementation

### The TransformerBlock Class
The core logic resides in the `TransformerBlock` class, which manages the sequential workflow of data through the block.

**Workflow in the `forward` method:**
1.  **First Sub-block:** Layer Norm $\rightarrow$ Multi-Head Attention $\rightarrow$ Dropout $\rightarrow$ Shortcut Connection.
2.  **Second Sub-block:** Layer Norm $\rightarrow$ Feed-Forward Network (with GELU) $\rightarrow$ Dropout $\rightarrow$ Shortcut Connection.

### Dimensionality Preservation
A critical feature of this implementation is that the input and output tensors maintain the exact same shape (e.g., `[batch_size, num_tokens, 768]`). This one-to-one relationship ensures scalability across the entire GPT architecture.

## Usage Example

To initialize a block and process a sample input tensor:

```python
# Configuration for gpt2-124M
cfg = {
    "emb_dim": 768,
    "context_length": 1024,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Initialize the block
block = TransformerBlock(cfg)

# Sample input: 2 batches, 4 tokens each, 768 dimensions
x = torch.randn(2, 4, 768)

# Process through the Transformer block
output = block(x)

print(output.shape) # Expected: torch.Size()
```


## Future Development
While the Transformer block is the heart of the engine, the next stage involves integrating this into the full **GPT Model**, including tokenization, positional embeddings, and post-processing steps for next-word prediction.