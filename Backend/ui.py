import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
# Load model and vectorizer
model_path = Path(__file__).parent / "sentiment_model.pkl"
vectorizer_path = Path(__file__).parent / "tfidf_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Label mapping
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
emoji_map = {"Negative": "❌ 😞", "Neutral": "🤔", "Positive": "✅ 😀"}

# Streamlit App
st.title("💬 Sentiment Analysis of Customer Review")

review = st.text_input("Enter a review:")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]
        probabilities = model.predict_proba(review_vector)[0]

        # Get readable label
        label = label_map.get(prediction, str(prediction))
        emoji = emoji_map.get(label, "🤷‍♂️")
        confidence = round(np.max(probabilities), 4)

        # Display result
        st.success(f"Prediction: **{label}** {emoji}")
        st.info(f"Confidence Score: `{confidence}`")

        # Bar Chart of all class probabilities
        st.subheader("📊 Class Probabilities")
        labels = [label_map[i] for i in range(len(probabilities))]
        plt.figure(figsize=(6, 3))
        bars = plt.bar(labels, probabilities, color=["red", "gray", "green"])
        plt.ylim([0, 1])
        plt.ylabel("Probability")
        plt.title("Sentiment Confidence by Class")
        st.pyplot(plt)
