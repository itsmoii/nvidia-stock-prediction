import re
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
from collections import Counter

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class SentimentAnalyzer:
    """Enhanced Random Forest Sentiment Analysis with comprehensive tuning"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None
        self.best_params = None
        self.feature_names = None

    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def create_synthetic_labels(self, comments, seed=42):
        """Create synthetic sentiment scores between 0 (negative) and 1 (positive)"""
        np.random.seed(seed)
        scores = []

        # Define positive and negative keywords for more realistic labeling
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'wonderful',
                          'fantastic', 'awesome', 'perfect', 'happy', 'pleased', 'satisfied']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointed',
                          'angry', 'frustrated', 'poor', 'useless', 'fail', 'broken']

        for comment in comments:
            comment_lower = str(comment).lower()
            pos_count = sum(1 for word in positive_words if word in comment_lower)
            neg_count = sum(1 for word in negative_words if word in comment_lower)

            # Base score between 0 and 1
            base_score = np.random.beta(pos_count + 1, neg_count + 1)

            # Add some noise and ensure it's between 0 and 1
            final_score = np.clip(base_score + np.random.normal(0, 0.1), 0, 1)
            scores.append(final_score)

        return np.array(scores)

    def build_pipeline(self):
        """Build the TF-IDF + Random Forest pipeline"""
        # Enhanced TF-IDF parameters
        tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.8,
            sublinear_tf=True,
            use_idf=True
        )

        rf = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            oob_score=True
        )

        self.pipeline = Pipeline([
            ('tfidf', tfidf),
            ('regressor', rf)
        ])

        return self.pipeline

    def get_param_grid(self):
        """Comprehensive parameter grid for Random Forest tuning"""
        return {
            'tfidf__max_features': [5000, 8000, 12000],
            'tfidf__ngram_range': [(1, 2), (1, 3)],
            'tfidf__min_df': [2, 3, 5],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'regressor__n_estimators': [200, 300, 500],
            'regressor__max_depth': [20, 30, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2', 0.8],
            'regressor__bootstrap': [True]
        }

    def train_and_tune(self, X_train, y_train, cv_folds=5):
        st.write("Building pipeline...")
        self.build_pipeline()

        st.write("Setting up cross-validation...")
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid=self.get_param_grid(),
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        st.write("Starting hyperparameter tuning...")
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()

        st.write(f"Tuning completed in {(end_time - start_time) / 60:.2f} minutes")

        self.pipeline = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        self.feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()

        return grid_search

    def evaluate_model(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)

        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        return y_pred, metrics

    def get_feature_importance(self, top_n=20):
        if self.pipeline is None:
            return None

        rf_model = self.pipeline.named_steps['regressor']
        feature_importance = rf_model.feature_importances_

        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        top_features = [(self.feature_names[i], feature_importance[i]) for i in top_indices]

        return top_features

    def create_comprehensive_visualization(self, y_test, y_pred, metrics):
        st.title("Sentiment Analysis Results")

        st.subheader("Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", f"{metrics['mse']:.4f}")
        col2.metric("RMSE", f"{metrics['rmse']:.4f}")
        col3.metric("MAE", f"{metrics['mae']:.4f}")
        col4.metric("R² Score", f"{metrics['r2']:.4f}")

        st.subheader("Actual vs Predicted Sentiment Scores")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax1)
        ax1.plot([0, 1], [0, 1], 'r--')
        ax1.set_xlabel("Actual Sentiment Score")
        ax1.set_ylabel("Predicted Sentiment Score")
        ax1.set_title("Actual vs Predicted Sentiment Scores")
        st.pyplot(fig1)

        st.subheader("Residual Plot")
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_xlabel("Residuals")
        ax2.set_title("Distribution of Residuals")
        st.pyplot(fig2)

        st.subheader("Top 15 Most Important Features")
        top_features = self.get_feature_importance(15)
        if top_features:
            features, importance = zip(*top_features)
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            y_pos = np.arange(len(features))
            ax3.barh(y_pos, importance, color='skyblue', edgecolor='navy', alpha=0.7)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(features)
            ax3.set_xlabel('Feature Importance')
            ax3.set_title('Top 15 Most Important Features')
            st.pyplot(fig3)


def main():
    st.set_page_config(page_title="Sentiment Analysis", layout="wide")

    analyzer = SentimentAnalyzer(random_state=42)

    # 1. Load and prepare data
    st.header("1. Data Loading and Preparation")
    df = pd.read_csv(r'C:\Users\kavis\OneDrive\Desktop\CS412\A2Test\Data\cleaned_comments.csv')

    initial_size = len(df)
    df = df.dropna(subset=['cleaned_comment'])
    df = df[df['cleaned_comment'].str.strip() != '']
    st.write(f"Data loaded: {initial_size} → {len(df)} comments after cleaning")

    st.header("2. Text Preprocessing")
    df['processed_comment'] = df['cleaned_comment'].apply(analyzer.preprocess_text)
    df = df[df['processed_comment'].str.len() > 10]
    st.write(f"After preprocessing: {len(df)} comments")

    st.header("3. Generating Synthetic Sentiment Scores")
    df['sentiment_score'] = analyzer.create_synthetic_labels(df['processed_comment'])

    st.subheader("Sentiment Score Distribution")
    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
    sns.histplot(df['sentiment_score'], bins=20, kde=True, ax=ax_dist)
    ax_dist.set_title("Distribution of Sentiment Scores")
    st.pyplot(fig_dist)

    st.header("4. Data Splitting")
    X = df['processed_comment']
    y = df['sentiment_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    st.write(f"Training set: {len(X_train)} samples (80%)")
    st.write(f"Test set: {len(X_test)} samples (20%)")

    st.header("5. Model Training and Tuning")
    with st.spinner("Training and tuning the model..."):
        grid_search = analyzer.train_and_tune(X_train, y_train, cv_folds=5)

    st.subheader("Best Parameters Found")
    st.write(analyzer.best_params)
    st.write(f"Best cross-validation RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

    st.header("6. Model Evaluation")
    y_pred, metrics = analyzer.evaluate_model(X_test, y_test)

    st.subheader("Test Set Performance")
    st.write(f"MSE: {metrics['mse']:.4f}")
    st.write(f"RMSE: {metrics['rmse']:.4f}")
    st.write(f"MAE: {metrics['mae']:.4f}")
    st.write(f"R² Score: {metrics['r2']:.4f}")

    st.header("7. Visualizations")
    analyzer.create_comprehensive_visualization(y_test, y_pred, metrics)

    st.header("8. Model Saving")
    if st.button("Save Model"):
        joblib.dump(analyzer.pipeline, 'enhanced_rf_sentiment_model.joblib')
        st.success("Model saved as 'enhanced_rf_sentiment_model.joblib'")


if __name__ == "__main__":
    main()
