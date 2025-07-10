import pandas as pd
import joblib

# Load the trained model and vectorizer
model = joblib.load('../backend/sentiment_model.pkl')
vectorizer = joblib.load('../backend/tfidf_vectorizer.pkl')

# Load Yelp dataset (or a subset)
yelp_data = pd.read_json('D:/projects/sentiment-analysis-project/training/yelp_review.json', lines=True)
# If using a subset: yelp_data = pd.read_json('yelp_subset.json', lines=True)

# Predict sentiments
reviews = yelp_data['text']
reviews_tfidf = vectorizer.transform(reviews)
predictions = model.predict(reviews_tfidf)
probabilities = model.predict_proba(reviews_tfidf)[:, 1]

# Add predictions to DataFrame
yelp_data['predicted_sentiment'] = ['Positive' if p == 1 else 'Negative' for p in predictions]
yelp_data['sentiment_score'] = probabilities

# Save results
yelp_data.to_csv('D:/projects/sentiment-analysis-project/yelp_analysis/yelp_predictions.csv', index=False)
print("Predictions saved to yelp_predictions.csv")