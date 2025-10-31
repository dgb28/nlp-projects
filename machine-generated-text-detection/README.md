# 🤖 Human vs Machine Text Classification

## 🌟 Overview
This project distinguishes **human-written** from **machine-generated text** using **bigram analysis**, **Logistic Regression**, and **Decision Trees**. It focuses on **feature importance**, **regularization effects**, and **model interpretability**.

## 🔧 Methodology

### Dataset
- Text samples (train, dev, test)
- Features: Bag-of-bigrams (vocabulary 10k–25k)

### Models & Hyperparameters

#### Logistic Regression
- C = 0.1, Penalty = L2, Max Iter = 500, Vocab = 25,000
- Train Accuracy: 0.9029  
- Dev Accuracy: 0.7424  
- Test Accuracy: 0.7087

#### Decision Tree
- Max Depth = 100, Min Samples Split = 500, Vocab = 10,000
- Train Accuracy: 0.7326  
- Dev Accuracy: 0.6554  
- Test Accuracy: 0.6283

## 🔍 Key Insights
- **Human-like bigrams:** `('.', 'cmv')`, `('change', 'my')`, `('told', 'bbc')`
- **Machine-like bigrams:** `('read', 'more')`, `('per', 'cent')`, `('etc.', ',')`
- Punctuation and formal connectors are **important discriminators**.
- **Regularization effects:**
  - L1 promotes sparsity and highlights key bigrams.
  - L2 distributes weights more evenly for smoother generalization.
- Logistic Regression **generalizes better** than Decision Trees.

## 🌳 Decision Tree Example Paths

| Path | Splits | Leaf Prediction |
|------|--------|----------------|
| 1    | “a footnote” → “such as” → “the” | Machine |
| 2    | “a footnote” → “such as” → “url1” | Human |
| 3    | “a footnote” → False → “,” → “due to” | Machine |

## 📊 Hyperparameter Summary

### Logistic Regression
| C    | Penalty | Max Iter | Vocab | Train Acc | Dev Acc |
|------|---------|----------|-------|-----------|---------|
| 0.1  | L2      | 500      | 25,000| 0.9029    | 0.7424 ✅ |

### Decision Tree
| Max Depth | Min Samples Split | Vocab | Train Acc | Dev Acc |
|-----------|-----------------|-------|-----------|---------|
| 100       | 500             | 10,000| 0.7326    | 0.6554 ✅ |

## 🚀 Future Work
- Implement **neural sequence models** like LSTM or Transformer to capture context.
- Explore **ensemble methods** combining Logistic Regression and Decision Trees.
- Visualize **bigram importance** for enhanced interpretability.
