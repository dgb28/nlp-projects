# 🧠 Word2Vec CBOW & Skip-gram (PyTorch)

This repository contains **from-scratch PyTorch implementations of Word2Vec models**, focusing on **Skip-gram with Negative Sampling** and **Continuous Bag-of-Words (CBOW)**.  
The project explores how word embeddings are learned from raw text and how contextual window size affects semantic similarity.

This work was completed as part of an NLP coursework assignment and emphasizes **implementation correctness, training behavior, and qualitative evaluation of learned embeddings**.

---

## 📌 Models Implemented

### 🔹 Skip-gram with Negative Sampling
- Predicts surrounding context words given a center word
- Uses **two embedding matrices**:
  - Center word embeddings
  - Context word embeddings
- Optimized using **binary cross-entropy with negative sampling**
- Provided as a reference implementation

### 🔹 Continuous Bag-of-Words (CBOW)
- Predicts the **target word from surrounding context words**
- Context embeddings are averaged before prediction
- Implemented as **multi-class classification** over the vocabulary
- Trained using **CrossEntropyLoss**

---

## 📂 Dataset
- **Brown Corpus** (plain text format)
- Vocabulary filtered using a minimum frequency threshold
- Example text includes a wide range of domains such as news, fiction, academic writing, and conversations

---

## ⚙️ Training Configuration

Default hyperparameters used for CBOW training:

| Parameter | Value |
|---------|-------|
| Embedding Dimension | 50 |
| Window Size | 2 |
| Batch Size | 128 |
| Epochs | 10 |
| Learning Rate | 0.003 |
| Minimum Word Frequency | 5 |

Training logs include:
- Batch-level loss updates
- Average loss per epoch
- Word similarity outputs after each epoch

---

## 📊 Results & Observations

### 🔹 Final Training Loss (CBOW, Window = 2)
- **Epoch 10 Average Loss:** `4.6688` :contentReference[oaicite:0]{index=0}

### 🔹 Learned Semantic Similarities

**Top words similar to _person_:**
- man
- woman
- dream
- thing
- child

**Top words similar to _good_:**
- satisfactory
- bad
- fair
- wonderful
- perfect

**Additional explored words:**
- *hard* → easy, right, weak
- *book* → story, letter, article
- *female* → (unexpected but interesting associations)

These results show that Word2Vec captures **contextual similarity rather than strict synonymy**, which is a core characteristic of distributional semantics.

---

## 🔍 Effect of Window Size

A second CBOW model was trained with **window size = 5**.

### Observed Differences:
- Larger window led to **broader contextual associations**
- Words like *person* became closer to **roles and professions** (e.g., teacher, lawyer)
- For *good*, some less intuitive associations appeared (e.g., rainy, tedious)

This highlights the tradeoff:
- Smaller window → syntactic & local semantics
- Larger window → topical & contextual semantics

---
