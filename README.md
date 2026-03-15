# 🎯 Amazon Product Review Sentiment Analysis

NLP project analyzing 568,000+ Amazon food reviews using two approaches — traditional Machine Learning (TF-IDF + Logistic Regression) and a HuggingFace Transformer (DistilBERT).

## 📊 Project Overview
This project builds a sentiment classifier to categorize Amazon product reviews as Positive, Negative, or Neutral using both a classical ML pipeline and a state-of-the-art transformer model.

## 📁 Dataset
- **Source:** [Amazon Fine Food Reviews - Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Size:** 568,454 reviews
- **Features:** Star rating (1-5), review summary, full review text

## 🔍 Key Findings
- Dataset is heavily imbalanced — 78% positive reviews (typical for Amazon)
- Neutral (3-star) reviews are the hardest to classify for both models
- DistilBERT achieves near-perfect confidence on clear positive/negative reviews
- TF-IDF + Logistic Regression achieves 72% accuracy — fast and lightweight

## 🤖 Models Compared

| Approach | Accuracy | Positive F1 | Negative F1 | Neutral F1 |
|----------|----------|-------------|-------------|------------|
| TF-IDF + Logistic Regression | 72% | 0.81 | 0.76 | 0.40 |
| HuggingFace DistilBERT | ~100% confidence | ✅ | ✅ | ➖ |

## 🧠 Approach 1 — TF-IDF + Logistic Regression
- Converts text to numerical features using TF-IDF (10,000 features)
- Handles class imbalance by downsampling majority class
- Fast training, interpretable, good baseline

## 🤗 Approach 2 — HuggingFace DistilBERT
- Pre-trained transformer model fine-tuned on SST-2 sentiment dataset
- No training required — works out of the box
- Understands context, sarcasm, and nuanced language far better than TF-IDF

## 💡 When to Use Each
- **TF-IDF + LR** — Large scale, speed critical, interpretability needed
- **DistilBERT** — High accuracy needed, nuanced text, smaller batches

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (TF-IDF, Logistic Regression)
- HuggingFace Transformers (DistilBERT)
- Matplotlib, Seaborn

## 📓 Notebook
See `Amazon_Review_Sentiment_Analysis.ipynb` for the full analysis.
