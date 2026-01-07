# 📚 NLP-Projects

This repository contains a collection of **Natural Language Processing (NLP)** projects exploring text understanding, representation learning, and summarization across multiple paradigms — from **classical machine learning** to **neural embeddings**, **Transformers**, and **multimodal models**.

Each project emphasizes **from-scratch implementation, experimentation, evaluation, and analysis**, making this repository both a learning resource and a technical portfolio.

---

## 🧩 Projects Overview

### 1️⃣ 📝 Logistic Regression – Sentiment Analysis  
Classifies text reviews as **positive or negative** using bag-of-words features and custom engineered features.  
Includes:
- Hyperparameter tuning (learning rate, number of steps)
- Analysis of most influential positive and negative words
- Evaluation on train, development, and test datasets

---

### 2️⃣ 🤖 Human vs Machine Text Classification  
Distinguishes **human-written** text from **machine-generated** text using **bigram features**.  
Includes:
- Logistic Regression and Decision Tree models
- Feature importance analysis and regularization effects (L1 vs L2)
- Decision tree interpretability and rule tracing
- Extensive hyperparameter experimentation and comparison

---

### 3️⃣ 🧠 Word2Vec CBOW & Skip-gram (PyTorch)  
Implements **Word2Vec from scratch** using PyTorch, focusing on:
- **Skip-gram with Negative Sampling**
- **Continuous Bag-of-Words (CBOW)**

Includes:
- Training word embeddings on the **Brown Corpus**
- Analysis of learned semantic similarities
- Study of contextual window size and its effect on embeddings
- Qualitative evaluation of nearest-neighbor word relationships

This project demonstrates how **distributional semantics** emerge from raw text without supervision.

---

### 4️⃣ 🧠 Transformer Language Model – Shakespeare  
Implements a **decoder-only Transformer language model from scratch** using PyTorch and trains it on the Tiny Shakespeare dataset.  
Includes:
- Causal self-attention and Transformer blocks
- Pre-norm vs post-norm architectural comparison
- Evaluation using perplexity
- Autoregressive Shakespeare-style text generation
- Analysis of attention behavior and generation quality

---

### 5️⃣ 🔁 MRedditSum – Multimodal Thread Summarization (Final Project)  
Reproduces and extends the **Cluster-based Multi-Stage (CMS)** summarization pipeline proposed in the MRedditSum paper.  
Includes:
- Single-stage vs CMS summarization comparison
- Text-only vs image-caption-augmented inputs
- Semantic clustering using sentence embeddings
- Multi-stage abstractive summarization with T5, BART, and LongT5
- Evaluation using ROUGE, BERTScore, and qualitative analysis

---

## 🔧 Common Features Across Projects
- 🧠 Classical ML, neural, and Transformer-based models  
- 📝 Bag-of-words, bigram, and embedding-based representations  
- ⚙️ Hyperparameter tuning and ablation studies  
- 📊 Evaluation using standard NLP metrics (Accuracy, ROUGE, Perplexity)  
- 🔍 Interpretability through feature weights, attention analysis, and clustering  
- 🧪 Reproducible experiments with clear reporting  

---

## 🚀 Purpose of This Repository
- Demonstrate **progressive mastery of NLP techniques**
- Bridge theory with practical implementation
- Serve as a portfolio of **coursework and research-oriented projects**

---
