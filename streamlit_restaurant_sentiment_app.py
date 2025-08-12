import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import string
import joblib
from datetime import datetime

# NLP / ML imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


nltk.download('stopwords', quiet=True)


st.set_page_config(page_title="Restaurant Review Sentiment Analyzer", layout="centered")

# Global words
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))
MODEL_FILENAME = "restaurant_sentiment_model.joblib"
DATA_FILENAME = "Restaurant_Reviews.tsv"  # expected file name if you add dataset to repo


# Text preprocessing

def text_process(review: str) -> str:
    """Basic cleaning + stemming + stopword removal. Returns cleaned string."""
    review = str(review).lower()
    # keep only letters and whitespace
    review = re.sub(r'[^a-zA-Z\s]', ' ', review)
    tokens = review.split()
    cleaned = [stemmer.stem(w) for w in tokens if (w not in STOPWORDS and len(w) > 2)]
    return ' '.join(cleaned)



# Data loading helper

def load_dataset_from_path(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep='\t')
    if 'Review' not in df.columns or 'Liked' not in df.columns:
        raise ValueError("Dataset must contain 'Review' and 'Liked' columns.")
    df = df.copy()
    df['cleaned_review'] = df['Review'].apply(text_process)
    return df


# Model training/selection


def train_and_select_best_model(df: pd.DataFrame, random_state: int = 42):
    """Train a few models and returning the best pipeline."""
    X = df['cleaned_review'].values
    y = df['Liked'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'SVM': SVC(kernel='rbf', probability=True, random_state=random_state),
        'Naive Bayes': MultinomialNB()
    }

    results = {}
    trained_pipelines = {}

    for name, clf in models.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)),
            ('classifier', clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        trained_pipelines[name] = pipeline

    best_name = max(results.keys(), key=lambda k: results[k])
    best_pipeline = trained_pipelines[best_name]

    # Cross-validation on training portion
    cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='accuracy')

    # Final evaluation on test set
    final_pred = best_pipeline.predict(X_test)
    final_acc = accuracy_score(y_test, final_pred)
    report = classification_report(y_test, final_pred, target_names=['Negative', 'Positive'])

    metadata = {
        'best_model_name': best_name,
        'model_accuracy': final_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    return best_pipeline, results, metadata, report

# Streamlit app UI

def main():
    st.title("🍽️ Restaurant Review Sentiment Analyzer (Streamlit)")
    st.write("This app loads a trained model if present, or trains one from a TSV dataset. No plots included ")

    col1, col2 = st.columns([3, 1])

    # Left column: model status + training controls
    with col1:
        st.subheader("Model status & training")

        # Optional model file upload (if you already have a .joblib model file)
        uploaded_model = st.file_uploader("(Optional) Upload a pre-trained .joblib model", type=['joblib'])
        if uploaded_model is not None:
            try:
                loaded = joblib.load(uploaded_model)
                joblib.dump(loaded, MODEL_FILENAME)  # persist it for later predictions
                st.success("Uploaded model saved locally as: {}".format(MODEL_FILENAME))
            except Exception as e:
                st.error(f"Failed to load uploaded model: {e}")

        # Option to force retrain
        force_retrain = st.checkbox('Force retrain (rebuild model even if saved model exists)')

        model_ready = False
        pipeline = None

        # If saved model exists and user did not request retrain -> load it
        if os.path.exists(MODEL_FILENAME) and not force_retrain:
            try:
                pipeline = joblib.load(MODEL_FILENAME)
                st.success(f"Loaded saved model: {MODEL_FILENAME}")
                model_ready = True
            except Exception as e:
                st.error(f"Failed to load saved model: {e}")

        # If model not ready, allow dataset upload or check repo file
        if not model_ready:
            st.info("No saved model found or retrain requested. Provide a dataset (TSV with 'Review' and 'Liked') to train.")

            uploaded = st.file_uploader("Upload Restaurant_Reviews.tsv (tab-separated)", type=['tsv', 'txt', 'csv'])

            df = None
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded, sep='\t')
                    st.write(f"Dataset uploaded: {df.shape[0]} rows")
                except Exception as e:
                    st.error(f"Failed to read uploaded file: {e}")

            # If no uploader used, check if data file exists in repo directory
            if df is None and os.path.exists(DATA_FILENAME):
                try:
                    df = load_dataset_from_path(DATA_FILENAME)
                    if df is not None:
                        st.write(f"Dataset loaded from disk: {DATA_FILENAME} ({df.shape[0]} rows)")
                except Exception as e:
                    st.error(f"Failed to load dataset from {DATA_FILENAME}: {e}")

            # Train button
            if st.button('Train model'):
                if df is None:
                    st.error("No dataset provided. Upload the TSV or add it to the repo with filename: {}".format(DATA_FILENAME))
                else:
                    with st.spinner('Training models — this may take a little time...'):
                        # ensure cleaned column exists in case the user uploaded raw file
                        if 'cleaned_review' not in df.columns:
                            df = df.copy()
                            df['cleaned_review'] = df['Review'].apply(text_process)

                        best_pipeline, results, metadata, report = train_and_select_best_model(df)
                        joblib.dump(best_pipeline, MODEL_FILENAME)
                        st.success(f"Training completed. Best model: {metadata['best_model_name']} (accuracy: {metadata['model_accuracy']:.4f})")
                        st.write('Cross-val mean accuracy:', round(metadata['cv_mean'], 4))
                        st.text('Classification Report (test set):')
                        st.text(report)

                        pipeline = best_pipeline
                        model_ready = True

    # Right column: quick info and samples
    with col2:
        st.subheader("Quick tips")
        st.markdown("- Put `Restaurant_Reviews.tsv` in the repo root or upload through the app.\n- Use the 'Train model' button to build the model.\n- Once model is saved, use the prediction box below.")
        st.markdown("**Sample reviews:**")
        st.write("• Amazing food and incredible service! Will definitely be back!")
        st.write("• Terrible experience. Food was cold and service was slow.")
        st.write("• It was okay. Nothing special but decent for the price.")

    st.markdown('---')

    # Prediction area (works if a model is available)
    st.subheader('Predict sentiment')
    review_text = st.text_area('Enter a restaurant review to predict its sentiment', height=120)

    if st.button('Predict sentiment'):
        if pipeline is None:
            st.error('Model is not ready. Train the model first or upload a saved .joblib model.')
        elif not review_text.strip():
            st.warning('Please enter a review text to predict.')
        else:
            cleaned = text_process(review_text)
            try:
                pred = pipeline.predict([cleaned])[0]
                try:
                    prob = pipeline.predict_proba([cleaned])[0]
                    confidence = max(prob) * 100
                except Exception:
                    confidence = None

                sentiment = 'Positive 👍' if int(pred) == 1 else 'Negative 👎'
                st.write('**Sentiment:**', sentiment)
                if confidence is not None:
                    st.write(f'**Confidence:** {confidence:.1f}%')
            except Exception as e:
                st.error(f'Prediction failed: {e}')

    st.markdown('---')
    st.write('App last updated:', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))


if __name__ == '__main__':
    main()
