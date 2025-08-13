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

import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Assuming these functions exist in your original code
# from your_module import text_process, train_and_select_best_model, load_dataset_from_path

MODEL_FILENAME = 'restaurant_sentiment_model.joblib'
DATA_FILENAME = 'Restaurant_Reviews.tsv'

def create_confidence_gauge(confidence):
    """Create a confidence gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen" if confidence > 70 else "orange" if confidence > 50 else "red"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_sentiment_distribution_chart(predictions_history):
    """Create a pie chart of sentiment predictions"""
    if not predictions_history:
        return None
    
    sentiments = [pred['sentiment'] for pred in predictions_history]
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545'}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="RestaurantScope - AI Sentiment Analyzer",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .prediction-positive {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-negative {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-weight: bold;
        height: 3rem;
    }
    
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        color: #333333;
    }
    
    .sidebar-info ol {
        color: #333333;
        margin: 0;
        padding-left: 1.2rem;
    }
    
    .sidebar-info li {
        color: #333333;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-info strong {
        color: #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ RestaurantScope</h1>
        <h3>AI-Powered Restaurant Review Sentiment Analysis</h3>
        <p>Discover what customers really think about their dining experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for predictions history
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []
    
    # Sidebar for model management
    with st.sidebar:
        # Use a food/restaurant emoji as logo
        st.markdown("### 🍽️ RestaurantScope")
        st.markdown("---")
        
        st.markdown("### 🔧 Model Management")
        
        # Model upload section
        with st.expander("📤 Upload Pre-trained Model", expanded=False):
            uploaded_model = st.file_uploader("Upload .joblib model file", type=['joblib'])
            if uploaded_model is not None:
                try:
                    loaded = joblib.load(uploaded_model)
                    joblib.dump(loaded, MODEL_FILENAME)
                    st.success("✅ Model uploaded successfully!")
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")
        
        # Training controls
        st.markdown("### 🎯 Training Controls")
        force_retrain = st.checkbox('🔄 Force retrain model', help="Rebuild model even if one exists")
        
        # Dataset upload
        st.markdown("### 📊 Dataset Upload")
        uploaded_data = st.file_uploader(
            "Upload training data (TSV format)", 
            type=['tsv', 'txt', 'csv'],
            help="File should contain 'Review' and 'Liked' columns"
        )
        
        # Quick stats
        if st.session_state.predictions_history:
            st.markdown("### 📈 Session Stats")
            total_predictions = len(st.session_state.predictions_history)
            positive_count = sum(1 for p in st.session_state.predictions_history if p['sentiment'] == 'Positive')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", total_predictions)
            with col2:
                st.metric("Positive %", f"{(positive_count/total_predictions)*100:.0f}%")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "🏋️ Train Model", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.markdown("### 🎯 Sentiment Prediction")
        
        # Check model status
        model_ready = False
        pipeline = None
        
        if os.path.exists(MODEL_FILENAME) and not force_retrain:
            try:
                pipeline = joblib.load(MODEL_FILENAME)
                st.success("✅ Model loaded and ready for predictions!")
                model_ready = True
            except Exception as e:
                st.error(f"❌ Failed to load model: {e}")
        else:
            st.warning("⚠️ No trained model found. Please train a model first.")
        
        if model_ready:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Enter Restaurant Review")
                review_text = st.text_area(
                    "Type your restaurant review here...",
                    height=150,
                    placeholder="e.g., The food was absolutely amazing! The service was quick and the atmosphere was perfect for a date night. Highly recommended!",
                    help="Enter any restaurant review to analyze its sentiment"
                )
                
                predict_button = st.button("🚀 Analyze Sentiment", type="primary")
                
                if predict_button and review_text.strip():
                    try:
                        # Assuming text_process function exists
                        cleaned = text_process(review_text)
                        pred = pipeline.predict([cleaned])[0]
                        
                        try:
                            prob = pipeline.predict_proba([cleaned])[0]
                            confidence = max(prob) * 100
                        except Exception:
                            confidence = None
                        
                        sentiment = 'Positive' if int(pred) == 1 else 'Negative'
                        emoji = '😊' if sentiment == 'Positive' else '😞'
                        
                        # Store prediction in history
                        st.session_state.predictions_history.append({
                            'review': review_text,
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'timestamp': datetime.now()
                        })
                        
                        # Display result
                        result_class = "prediction-positive" if sentiment == 'Positive' else "prediction-negative"
                        st.markdown(f"""
                        <div class="{result_class}">
                            <h2>{emoji} {sentiment} Sentiment</h2>
                            {f"<p>Confidence: {confidence:.1f}%</p>" if confidence else ""}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
                
                elif predict_button:
                    st.warning("⚠️ Please enter a review text to analyze.")
            
            with col2:
                if st.session_state.predictions_history:
                    latest = st.session_state.predictions_history[-1]
                    if latest.get('confidence'):
                        st.markdown("#### Confidence Level")
                        fig_gauge = create_confidence_gauge(latest['confidence'])
                        st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown("#### 💡 Sample Reviews")
                samples = [
                    "Amazing food and incredible service! Will definitely be back!",
                    "Terrible experience. Food was cold and service was slow.",
                    "It was okay. Nothing special but decent for the price.",
                    "Best restaurant in town! Every dish was perfectly prepared.",
                    "Overpriced and underwhelming. Would not recommend."
                ]
                
                for sample in samples:
                    if st.button(f"Try: {sample[:30]}...", key=sample):
                        st.text_area("Review", value=sample, key="sample_review")
    
    with tab2:
        st.markdown("### 🏋️ Model Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            df = None
            
            # Handle uploaded data
            if uploaded_data is not None:
                try:
                    df = pd.read_csv(uploaded_data, sep='\t')
                    st.success(f"✅ Dataset uploaded: {df.shape[0]} rows, {df.shape[1]} columns")
                    
                    # Show data preview
                    st.markdown("#### Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Failed to read uploaded file: {e}")
            
            # Check for local data file
            elif os.path.exists(DATA_FILENAME):
                try:
                    df = load_dataset_from_path(DATA_FILENAME)
                    if df is not None:
                        st.info(f"📁 Using local dataset: {DATA_FILENAME} ({df.shape[0]} rows)")
                        st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Failed to load local dataset: {e}")
            
            else:
                st.warning("⚠️ No dataset found. Please upload a TSV file or add Restaurant_Reviews.tsv to your project.")
            
            # Training button
            if st.button("🚀 Start Training", type="primary", disabled=(df is None)):
                if df is None:
                    st.error("❌ No dataset available for training.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("🔄 Preprocessing data...")
                        progress_bar.progress(20)
                        
                        # Ensure cleaned column exists
                        if 'cleaned_review' not in df.columns:
                            df = df.copy()
                            df['cleaned_review'] = df['Review'].apply(text_process)
                        
                        progress_bar.progress(40)
                        status_text.text("🤖 Training models...")
                        
                        # Train model (assuming this function exists)
                        best_pipeline, results, metadata, report = train_and_select_best_model(df)
                        
                        progress_bar.progress(80)
                        status_text.text("💾 Saving model...")
                        
                        joblib.dump(best_pipeline, MODEL_FILENAME)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Training completed!")
                        
                        # Display results
                        st.success(f"🎉 Training completed successfully!")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Best Model", metadata['best_model_name'])
                        with col_b:
                            st.metric("Accuracy", f"{metadata['model_accuracy']:.3f}")
                        with col_c:
                            st.metric("CV Score", f"{metadata['cv_mean']:.3f}")
                        
                        with st.expander("📋 Detailed Classification Report"):
                            st.text(report)
                        
                        # Update pipeline for immediate use
                        pipeline = best_pipeline
                        model_ready = True
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {e}")
                        progress_bar.empty()
                        status_text.empty()
        
        with col2:
            st.markdown("#### 📝 Training Tips")
            st.markdown("""
            <div class="sidebar-info">
            <h4>💡 For best results:</h4>
            <ul>
            <li>Use balanced datasets (equal positive/negative reviews)</li>
            <li>Include diverse review styles and lengths</li>
            <li>Ensure data quality (no duplicates, clear labels)</li>
            <li>Have at least 1000+ samples for robust training</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if df is not None:
                st.markdown("#### 📊 Dataset Info")
                st.write(f"**Rows:** {df.shape[0]:,}")
                st.write(f"**Columns:** {df.shape[1]}")
                
                if 'Liked' in df.columns:
                    sentiment_dist = df['Liked'].value_counts()
                    st.write("**Sentiment Distribution:**")
                    for sentiment, count in sentiment_dist.items():
                        st.write(f"- {sentiment}: {count:,} ({count/len(df)*100:.1f}%)")
    
    with tab3:
        st.markdown("### 📊 Analytics Dashboard")
        
        if not st.session_state.predictions_history:
            st.info("🔮 No predictions made yet. Go to the Predict tab to start analyzing reviews!")
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_preds = len(st.session_state.predictions_history)
            positive_preds = sum(1 for p in st.session_state.predictions_history if p['sentiment'] == 'Positive')
            avg_confidence = np.mean([p.get('confidence', 0) for p in st.session_state.predictions_history if p.get('confidence')])
            
            with col1:
                st.metric("Total Predictions", total_preds)
            with col2:
                st.metric("Positive Reviews", positive_preds)
            with col3:
                st.metric("Positive Rate", f"{(positive_preds/total_preds)*100:.1f}%")
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = create_sentiment_distribution_chart(st.session_state.predictions_history)
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Confidence distribution
                confidences = [p.get('confidence', 0) for p in st.session_state.predictions_history if p.get('confidence')]
                if confidences:
                    fig_hist = px.histogram(
                        x=confidences,
                        title="Confidence Score Distribution",
                        nbins=10,
                        color_discrete_sequence=['#667eea']
                    )
                    fig_hist.update_layout(
                        xaxis_title="Confidence (%)",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            # Recent predictions table
            st.markdown("#### 📝 Recent Predictions")
            recent_df = pd.DataFrame([{
                'Review': p['review'][:100] + '...' if len(p['review']) > 100 else p['review'],
                'Sentiment': p['sentiment'],
                'Confidence': f"{p.get('confidence', 0):.1f}%" if p.get('confidence') else 'N/A',
                'Time': p['timestamp'].strftime('%H:%M:%S')
            } for p in st.session_state.predictions_history[-10:]])
            
            st.dataframe(recent_df, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear Prediction History"):
                st.session_state.predictions_history = []
                st.experimental_rerun()
    
    with tab4:
        st.markdown("### ℹ️ About RestaurantScope")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **RestaurantScope** is an advanced AI-powered sentiment analysis tool specifically designed for restaurant reviews. 
            It helps restaurants, food critics, and food enthusiasts understand the emotional tone behind customer feedback.
            
            #### 🎯 Key Features:
            - **Real-time Sentiment Analysis**: Get instant feedback on review sentiment
            - **Confidence Scoring**: Understand how certain the AI is about its predictions
            - **Model Training**: Train custom models on your own restaurant data
            - **Interactive Analytics**: Visualize sentiment trends and patterns
            - **User-friendly Interface**: Clean, modern UI designed for ease of use
            
            #### 🔧 How it Works:
            1. **Text Processing**: Reviews are cleaned and preprocessed
            2. **Feature Extraction**: Key linguistic features are identified
            3. **ML Classification**: Advanced algorithms determine sentiment
            4. **Confidence Calculation**: Statistical measures provide confidence scores
            
            #### 📊 Supported Formats:
            - TSV files with 'Review' and 'Liked' columns
            - Pre-trained .joblib model files
            - Real-time text input for predictions
            """)
        
        with col2:
            st.markdown("#### 🚀 Quick Start")
            st.markdown("""
            <div class="sidebar-info">
            <ol>
            <li><strong>Upload Data:</strong> Add your restaurant review dataset</li>
            <li><strong>Train Model:</strong> Build your sentiment analyzer</li>
            <li><strong>Make Predictions:</strong> Analyze new reviews</li>
            <li><strong>View Analytics:</strong> Explore your results</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 💡 Tips for Better Results")
            st.markdown("""
            - Use diverse, high-quality training data
            - Include reviews of varying lengths
            - Ensure balanced positive/negative samples  
            - Regularly retrain with new data
            - Monitor confidence scores for reliability
            """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"🕐 Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    with col2:
        st.write("🤖 Powered by AI & Machine Learning")
    with col3:
        st.write("🍽️ RestaurantScope v2.0")

if __name__ == '__main__':
    main()
