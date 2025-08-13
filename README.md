# 🍽️ Restaurant Review Sentiment Analyzer

A **Streamlit** web app for analyzing restaurant review sentiment (**Positive 👍** / **Negative 👎**).  
Upload your dataset, train multiple machine learning models, pick the best one automatically, and make predictions instantly.  
Supports saving/loading trained models for quick reuse.

---

##  Features
- Upload a restaurant review dataset (`Restaurant_Reviews.tsv`)
- Automatic **text preprocessing** (lowercasing, punctuation removal, stemming, stopword filtering)
- Trains **Logistic Regression**, **Random Forest**, **SVM**, **Naive Bayes** and picks the best model by accuracy
- Displays **cross-validation metrics** and classification report
- Save & load `.joblib` model files for reuse
- Predict sentiment for new customer reviews
- Confidence score (when supported by model)

---

##  Dataset Format
Your dataset should be **tab-separated (`.tsv`)** with:
| Review                                      | Liked |
|---------------------------------------------|-------|
| "Amazing food and service"                  |   1   |
| "Terrible experience, food was cold"        |   0   |

- **Review** → text of the review  
- **Liked** → 1 for positive, 0 for negative

##  How to Run
### 1️⃣ Clone the repository
### 2️⃣ Install dependencies
### 3️⃣ Add dataset Place `Restaurant_Reviews.tsv` in the project root folder (or upload through the app).
### 4️⃣ Start the Streamlit app

<<<<<<< HEAD
## ⚙️ How It Works
=======
## How It Works
>>>>>>> e2826a141759de4559685ae1031c8a30ba2fb484
1. **Preprocessing**
   - Lowercases text
   - Removes non-letter characters
   - Removes stopwords
   - Applies Porter stemming
2. **Model Training**
   - Trains Logistic Regression, Random Forest, Naive Bayes, and SVM on TF-IDF features
   - Selects the best-performing model by test accuracy
3. **Saving & Loading**
   - Saves the trained model to `restaurant_sentiment_model.joblib`
   - Can load any `.joblib` file uploaded to the UI
4. **Prediction**
   - Enter a review → app shows sentiment (Positive or Negative) + confidence score (if available)

---

##  Example Predictions
| Review                                        | Prediction | Confidence |
|-----------------------------------------------|------------|------------|
| Amazing food and incredible service!          | Positive 👍 | 97%        |
| Food was cold and the service was terrible    | Negative 👎 | 94%        |
| It was okay. Nothing special but decent price | Positive 👍 | 66%        |

<<<<<<< HEAD
---

---
=======
>>>>>>> e2826a141759de4559685ae1031c8a30ba2fb484

##  Credits
- **NLTK** for stopwords and stemming
- **scikit-learn** for ML algorithms & TF-IDF vectorization
- **Streamlit** for the UI
<<<<<<< HEAD
- Code written and assembled by Prateek Prasad Deshpande
=======
- Code written and assembled by Prateek P D
>>>>>>> e2826a141759de4559685ae1031c8a30ba2fb484
