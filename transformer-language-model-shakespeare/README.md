# 🧠 Transformer Language Model (Shakespeare)

This repository implements a **decoder-only Transformer language model from scratch** using **PyTorch**, trained on the **Tiny Shakespeare dataset**.  
The project focuses on implementing the **core components of Transformers**, evaluating **pre-norm vs post-norm architectures**, and analyzing **text generation quality**.

This work was completed as part of an NLP coursework assignment on **language modeling with Transformers**.

---

## 🌟 Project Overview

Language modeling is the task of predicting the **next token given prior context**, which underlies modern LLMs such as GPT-3 and GPT-4.  
In this project, a **causal (decoder-only) Transformer** is trained to model Shakespearean text and generate stylistically similar dialogue.

### Key objectives:
- Implement **multi-head self-attention** from scratch
- Implement **Transformer blocks** supporting pre-norm and post-norm variants
- Train and evaluate the model using **perplexity**
- Compare architectural choices
- Analyze **generated Shakespeare-like text**

---

## 🧩 Model Architecture

The model follows a **GPT-style decoder-only Transformer** architecture:

- Token embeddings + learned positional embeddings
- Stack of **3 Transformer blocks**
  - Multi-head self-attention (causal masking)
  - Feed-forward networks
  - Residual connections
  - Layer normalization (pre-norm or post-norm)
- Linear projection to vocabulary logits

### 🔹 Implemented Components
- Multi-Head Self-Attention
- Transformer Block (Pre-Norm & Post-Norm)
- Causal Masking for autoregressive modeling
- Autoregressive text generation

---

## 📂 Dataset

- **Tiny Shakespeare Dataset**
- ~40,000 lines of Shakespeare plays (~1MB text)
- Tokenized using the **GPT-2 tokenizer** via `tiktoken`
- Includes dialogue-heavy text with speakers, punctuation, and archaic English

---

## ⚙️ Training Setup

| Parameter | Value |
|---------|------|
| Model Type | Decoder-only Transformer |
| Layers | 3 |
| Attention Heads | 4 |
| Embedding Dim | 64 |
| Tokenizer | GPT-2 (tiktoken) |
| Optimizer | Adam |
| Epochs | 3 |
| Metric | Perplexity |

---

## 📊 Model Performance & Analysis

### 🔹 Final Development Set Perplexity

| Architecture | Dev Perplexity |
|-------------|---------------|
| **Post-Norm** | **184.81** |
| Pre-Norm | 206.19 |

The GPT-2 tokenizer produces a vocabulary of ~50,257 tokens, meaning a **uniform baseline** would have perplexity ≈ 50,257.  
Achieving perplexity in the **180–220 range** shows that the model has learned **meaningful language structure**, far outperforming the baseline :contentReference[oaicite:0]{index=0}.

---

### 🔹 Pre-Norm vs Post-Norm Comparison

**Which converged faster?**
- In this small 3-layer model, **post-norm converged faster**
- Post-norm reached lower dev perplexity earlier (by epoch 2)

**Stability Analysis**
- Pre-norm showed **smoother loss curves**
- Post-norm showed larger fluctuations and mild overfitting later
- This matches theory:
  - Pre-norm stabilizes gradient flow
  - Post-norm can optimize faster in shallow models

**When to prefer which?**
- **Pre-norm**: deep models, long training, stability-critical settings
- **Post-norm**: shallow models, short training runs, faster early optimization

---

## ✍️ Text Generation Analysis

The trained model can generate Shakespeare-style dialogue from prompts.

### Example Prompts Used
- `ROMEO:`
- `JULIET:`
- `KING:`
- `To be or not to be`

### Observed Patterns
- Character-based dialogue formatting (`ROMEO:`, `KING:`)
- Frequent use of dramatic themes (honor, death, love)
- Script-like structure with alternating speakers
- Consistent use of colons after character names

Despite the small model size, the generated text **captures Shakespeare’s tone and rhythm surprisingly well** :contentReference[oaicite:1]{index=1}.

---

### ⚠️ Common Errors
- **Grammatical errors**: “We am”, “I have’s”
- **Nonsensical tokens**: invented words, broken phrases
- **Repetition**: repeated sentence structures
- **Punctuation errors**: mismatched quotes, misplaced colons

### Long-Range Dependencies
What the model does well:
- Maintains dialogue format
- Preserves thematic consistency over short spans

What it struggles with:
- Coherent long narratives
- Speaker consistency over many lines
- Matching brackets and quotes

These limitations are expected for a **3-layer, 64-dim Transformer**.

---

## 🚀 How to Improve the Model

Based on experimental analysis:

### Model Architecture
- Increase embedding dimension (256–512)
- Increase number of layers (6–12)
- Increase attention heads
- Prefer pre-norm for deeper models

### Training Procedure
- Train for more epochs
- Use larger or additional Shakespeare-like datasets

### Generation Strategy
- Lower temperature for reduced randomness
- Use top-k or nucleus sampling
- Avoid greedy decoding

### Post-Training
- Fine-tuning
- Longer context windows
- Regularization techniques

---

## 🚀 How to Run

```bash
pip install torch numpy matplotlib tiktoken
python a4.py
