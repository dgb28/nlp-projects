# 📈 Logistic Regression Sentiment Analysis

## 🌟 Overview
This project implements a **Logistic Regression model** to classify text reviews as **positive or negative**. The goal is to explore **feature engineering**, **hyperparameter tuning**, and **word importance analysis** to understand what drives sentiment in textual data.

## 🔧 Methodology

### Dataset
- Text reviews (split into train, development, and test sets)
- Features:
  - Bag-of-words representation
  - Log-scaled review length
  - Count of positive words
  - Count of negative words

### Model
- **Logistic Regression** with different hyperparameters:
  - Learning rate (`n`): `[0.1, 0.01, 0.001]`
  - Number of steps: `[10k, 100k, 500k]`

### Experiments
- Tested multiple learning rates and iterations.
- Evaluated development set accuracy to select the best model.

## 📊 Best Model
- Learning rate: `0.01`  
- Steps: `500k`  
- Features: Base bag-of-words only  
- Development Accuracy: `0.7790`  
- Test Accuracy: `0.7790`

## 🔍 Insights
- Adding extra features (review length, positive/negative counts) **did not improve accuracy**, suggesting the base bag-of-words features are already strong.
- Top positive words: `awesome, perfect, amazing, excellent`  
- Top negative words: `horrible, worst, rude, terrible`  
- Logistic Regression weights align well with **intuitive human sentiment guesses**.

## 📊 Accuracy Table

| Learning Rate | Steps  | Dev Accuracy |
|---------------|--------|--------------|
| 0.1           | 10k    | 0.7620       |
| 0.1           | 100k   | 0.7760       |
| 0.1           | 500k   | 0.7780       |
| 0.01          | 10k    | 0.7480       |
| 0.01          | 100k   | 0.7765       |
| 0.01          | 500k   | 0.7790 ✅    |
| 0.001         | 10k    | 0.6975       |
| 0.001         | 100k   | 0.7480       |
| 0.001         | 500k   | 0.7685       |

## 🚀 Future Work
- Experiment with **TF-IDF features** instead of raw bag-of-words.
- Explore **deep learning approaches** such as LSTM or BERT for improved sentiment understanding.
- Implement **feature selection** to reduce noise from extra features.

