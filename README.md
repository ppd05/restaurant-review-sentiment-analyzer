# 🍽️ Restaurant Review Sentiment Analyzer

A **Streamlit** web app for analyzing restaurant review sentiment (**Positive 👍** / **Negative 👎**).  
Upload your dataset, train multiple machine learning models, pick the best one automatically, and make predictions instantly.  
Supports saving/loading trained models for quick reuse.

---

## ✨ Features
- Upload a restaurant review dataset (`Restaurant_Reviews.tsv`)
- Automatic **text preprocessing** (lowercasing, punctuation removal, stemming, stopword filtering)
- Trains **Logistic Regression**, **Random Forest**, **SVM**, **Naive Bayes** and picks the best model by accuracy
- Displays **cross-validation metrics** and classification report
- Save & load `.joblib` model files for reuse
- Predict sentiment for new customer reviews
- Confidence score (when supported by model)

---

## 📂 Dataset Format
Your dataset should be **tab-separated (`.tsv`)** with:
| Review                                      | Liked |
|---------------------------------------------|-------|
| "Amazing food and service"                  |   1   |
| "Terrible experience, food was cold"        |   0   |

- **Review** → text of the review  
- **Liked** → 1 for positive, 0 for negative

## 🚀 How to Run
### 1️⃣ Clone the repository
