# 🔁 MRedditSum — Reproduction & Analysis (CMS Pipeline)

This project reproduces and analyzes the **Cluster-based Multi-Stage (CMS)** summarization pipeline proposed in the **MRedditSum** paper.  
We extend the original work with controlled comparisons between **text-only**, **image-caption-augmented**, and **CMS-based** summarization pipelines.

The full experimental setup, results, and analysis are documented in the accompanying project report.

> Original paper: *MREDDITSUM: A Multimodal Abstractive Summarization Dataset of Reddit Threads with Images*  
> Overbay et al., ACL 2023

---

## 🔎 Project Overview

- **Goal**  
  Reproduce the CMS pipeline and evaluate how **image captions** and **semantic comment clustering** impact abstractive summarization quality on the MRedditSum dataset.

- **Approach**
  - Train encoder–decoder summarization models using:
    - **Text-only input**
    - **Text + image captions**
  - Apply **semantic clustering** over Reddit comments using sentence embeddings.
  - Generate **cluster-level summaries**, followed by **multi-stage synthesis** into a final thread summary.
  - Evaluate using **ROUGE (R-1, R-2, R-L)** and **BERTScore**, supported by qualitative analysis.

---

## 📚 Key Findings

- **CMS significantly outperforms single-stage summarization**, with gains of approximately **+20 ROUGE-1 points** in several reproduced settings.
- **Image captions consistently improve performance**, both in single-stage and CMS pipelines, though the gains are smaller than those provided by clustering.
- For some cluster configurations, **BART within the CMS framework achieved the highest ROUGE-1**, indicating that model choice and cluster granularity play an important role.
- Overall, the reproduction confirms the original MRedditSum conclusion that **multimodal inputs combined with cluster-based processing lead to better summaries**.

---

## 🧪 Reproduced Quantitative Results (High-Level)

> The following results summarize representative comparisons between single-stage and CMS pipelines.  
> Refer to the report for full tables and additional configurations.

### Single-Stage (Example)
- **T5-Base (With Image Captions)**  
  - ROUGE-1: 21.34  
  - ROUGE-2: 12.21  
  - ROUGE-L: 19.14  

### CMS (Cluster-Based Multi-Stage) Highlights
- **CMS T5-Base (With Image Captions)**  
  - ROUGE-1 ≈ 41.40  
  - ROUGE-2 ≈ 14.48  
  - ROUGE-L ≈ 24.20  

- **CMS BART-Base (With Image Captions)**  
  - ROUGE-1 ≈ 45.17  
  - ROUGE-2 ≈ 19.19  
  - ROUGE-L ≈ 29.94  

---

## 🛠 Implementation Details

### Models
- **T5-Base** (text-only / + image captions)
- **BART-Base** (text-only / + image captions)
- **LongT5-Base** (for long-context experiments)
- **Vision-guided variants** (VG-T5, VG-BART) using ViT image embeddings *(experimental)*

### Clustering
- Sentence embeddings: `all-distilroberta-v1` (SentenceTransformer)
- Agglomerative clustering:
  - Distance: cosine
  - Linkage: average
  - Threshold: ≈ 0.5

### Training
- Optimizer: AdamW
- Learning rate:
  - `3e-5` for pretrained parameters
  - `1.5e-4` for newly introduced visual layers
- Epochs: 20–50 (model-dependent)
- Sequence length:
  - Up to 1,024 tokens (standard models)
  - Up to 4,096 tokens (LongT5)
- Precision: **bfloat16** used for LongT5 to avoid FP16 instability

### Decoding
- Beam search (beam size 4–5)
- Length penalty applied during CMS synthesis

---
