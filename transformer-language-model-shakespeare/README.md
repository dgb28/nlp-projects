# 🧠 Transformer Language Model (Shakespeare)

This repository implements a **decoder-only Transformer language model from scratch** using **PyTorch**, trained on the **Tiny Shakespeare dataset**.  
The project focuses on understanding and implementing the **core components of Transformers**, including **multi-head self-attention**, **Transformer blocks**, and **causal language modeling**.

This work was completed as part of an NLP coursework assignment on **language modeling with Transformers** :contentReference[oaicite:0]{index=0}.

---

## 🌟 Overview

Language modeling is the task of predicting the **next token given previous context**, a foundational problem behind modern large language models such as GPT.  
In this project, we build a **causal (decoder-only) Transformer** that learns to model Shakespearean text and generate stylistically similar sequences.

Key goals of the project:
- Implement Transformer components **without using high-level abstractions**
- Compare **pre-norm vs post-norm** Transformer architectures
- Evaluate models using **perplexity**
- Analyze **generated text** and **attention patterns**

---

## 🧩 Model Architecture

The model follows a **GPT-style decoder-only Transformer** architecture:

- Token embeddings + learned positional embeddings
- Stack of Transformer blocks:
  - Multi-head self-attention (causal masking)
  - Feed-forward networks
  - Residual connections
  - Layer normalization (pre-norm or post-norm)
- Linear projection to vocabulary for next-token prediction

### 🔹 Implemented Components
- Multi-Head Self-Attention
- Transformer Block (Pre-Norm & Post-Norm)
- Learned Positional Embeddings
- Autoregressive Text Generation
- Attention Visualization

---

## 📂 Dataset

- **Tiny Shakespeare Dataset**
- Approximately 40,000 lines of Shakespeare plays
- Tokenized using the **GPT-2 tokenizer** via `tiktoken`
- Example training text includes dialogues, stage directions, and classical Shakespearean structure

---

## ⚙️ Training Setup

Default training configuration:

| Parameter | Value |
|---------|------|
| Model Type | Decoder-only Transformer |
| Tokenizer | GPT-2 (tiktoken) |
| Max Sequence Length | 256 |
| Optimization | Adam |
| Loss Function | Cross-Entropy |
| Evaluation Metric | Perplexity |

The training loop:
- Computes token-level cross-entropy loss
- Uses gradient clipping for stability
- Evaluates performance on a development set
- Reports perplexity as the primary metric

---

## 📊 Evaluation & Analysis

### 🔹 Perplexity
- Model performance is evaluated using **perplexity**, a standard metric for language models
- Lower perplexity indicates better next-token prediction
- Results are compared against a **uniform baseline** (vocabulary-size perplexity)

### 🔹 Pre-Norm vs Post-Norm
- Both architectures were trained and compared
- Pre-norm models generally:
  - Converge faster
  - Train more stably
  - Are preferred for deeper Transformer stacks

Loss curves are saved and analyzed to compare convergence behavior :contentReference[oaicite:1]{index=1}.

---

## ✍️ Text Generation

The trained model can generate Shakespeare-style text given a prompt.

Generation features:
- Temperature-based sampling
- Autoregressive decoding
- Stops at end-of-text token
- Captures dialogue structure and archaic language patterns

Generated samples are analyzed for:
- Grammatical consistency
- Repetition
- Long-range dependencies
- Stylistic similarity to Shakespeare

---

## 👁️ Attention Visualization

The project includes functionality to:
- Extract attention weights from all layers and heads
- Visualize attention as heatmaps
- Analyze how tokens attend to previous context

This provides interpretability into how the Transformer processes language internally.

---

## 📁 Repository Structure

