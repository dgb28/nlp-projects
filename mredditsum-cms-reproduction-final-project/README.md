# 🔁 MRedditSum — Reproduction & Extended Analysis (CMS Pipeline)

This project reproduces and extends the **Cluster-based Multi-Stage (CMS)** summarization pipeline proposed in the *MRedditSum* paper.  
We systematically evaluate how **semantic comment clustering** and **image information** affect abstractive summarization of Reddit threads.

In addition to reproducing the original methodology, we conduct **controlled ablation studies**, **model comparisons**, and **quantitative + qualitative analyses** using multiple encoder–decoder architectures.

---

## 📌 Project Motivation

Summarizing Reddit threads is challenging due to:
- Long, noisy, multi-speaker discussions
- Redundant and contradictory comments
- Optional multimodal context (images)

The **CMS pipeline** addresses these challenges by:
1. **Clustering comments semantically**
2. **Summarizing each cluster independently**
3. **Synthesizing a final global summary**

This project evaluates whether:
- CMS improves over flat, single-stage summarization
- Image captions meaningfully help
- Different models behave differently inside CMS

---

## 🔎 Project Overview

### Goals
- Reproduce the CMS pipeline on the **MRedditSum dataset**
- Compare **single-stage vs CMS** summarization
- Evaluate **text-only vs image-augmented** inputs
- Analyze the impact of **model choice and cluster granularity**

### Approach
- Fine-tune multiple pretrained encoder–decoder models
- Implement semantic clustering using sentence embeddings
- Run CMS-style multi-stage summarization
- Evaluate using **ROUGE**, **BERTScore**, and qualitative analysis

---

## 🧠 Models Evaluated

### Encoder–Decoder Models
- **T5-Base**
- **BART-Base**
- **LongT5-Base** (for long-context inputs)

Each model is evaluated under:
- Text-only input
- Text + image captions
- CMS (cluster-based multi-stage) pipeline

### Experimental Variants
- Single-stage summarization
- CMS with varying cluster sizes
- Vision-guided variants using ViT image embeddings *(experimental)*

---

## 🧩 CMS Pipeline (Implemented)

1. **Comment Embedding**
   - SentenceTransformer: `all-distilroberta-v1`

2. **Clustering**
   - Agglomerative clustering
   - Cosine distance
   - Average linkage
   - Distance threshold ≈ 0.5

3. **Cluster-Level Summarization**
   - Each cluster summarized independently

4. **Synthesis Stage**
   - Cluster summaries concatenated
   - Final abstractive summary generated

This design reduces redundancy and encourages topic-level coverage.

---

## 📊 Evaluation Metrics

- **ROUGE-1, ROUGE-2, ROUGE-L**
- **BERTScore**
- Qualitative analysis of generated summaries
- Comparison to reported results in the original paper

---

## 📚 Key Findings

### 1️⃣ CMS vs Single-Stage Summarization
- CMS consistently outperforms single-stage models
- Improvements of **~20 ROUGE-1 points** observed across multiple configurations
- Gains are due to:
  - Reduced redundancy
  - Better topic coverage
  - Improved handling of long discussions

### 2️⃣ Impact of Image Captions
- Adding image captions improves performance in:
  - Single-stage models
  - CMS pipeline
- Gains are **modest but consistent**
- Image information complements textual signals but does not replace structural modeling

### 3️⃣ Model-Specific Behavior
- **BART inside CMS** achieved the highest ROUGE-1 in some cluster settings
- **T5** models were more stable across configurations
- **LongT5** enabled longer contexts but required careful training due to memory constraints

### 4️⃣ Cluster Granularity Matters
- Too few clusters → topic mixing
- Too many clusters → fragmented summaries
- Moderate cluster sizes yielded the best performance

---

## 🧪 Quantitative Results (Representative)

### Single-Stage Example
| Model | Input | R-1 | R-2 | R-L |
|-----|------|----|----|----|
| T5-Base | Text + Img | 21.34 | 12.21 | 19.14 |

### CMS Highlights
| Model | Input | R-1 | R-2 | R-L |
|-----|------|----|----|----|
| CMS T5-Base | Text + Img | ~41.40 | ~14.48 | ~24.20 |
| CMS BART-Base | Text + Img | ~45.17 | ~19.19 | ~29.94 |

*(Full tables and additional runs are available in the report.)*

---

## 🛠 Training Details

### Optimization
- Optimizer: AdamW
- Learning rates:
  - `3e-5` for pretrained parameters
  - `1.5e-4` for new visual layers
- Epochs: 20–50 (model-dependent)

### Input Length
- Up to 1,024 tokens (T5 / BART)
- Up to 4,096 tokens (LongT5)

### Precision
- FP16 for most models
- **BFloat16** for LongT5 to avoid FP16 instability on A100 GPUs

### Decoding
- Beam search (beam size 4–5)
- Length penalty applied during CMS synthesis

---

## 🧪 Qualitative Observations

### Strengths
- CMS summaries are:
  - Less repetitive
  - More structured
  - Better at capturing diverse viewpoints
- Multimodal summaries include more concrete details

### Failure Modes
- Occasional hallucinations
- Over-compression of minority viewpoints
- Sensitivity to clustering threshold

---
