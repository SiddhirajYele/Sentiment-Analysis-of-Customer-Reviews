import sys
import joblib

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Get the review from command-line arguments
if len(sys.argv) > 1:
    review = sys.argv[1]
else:
    review = input("Enter a review: ")

# Process review
review_tfidf = vectorizer.transform([review])
prediction = model.predict(review_tfidf)[0]
probability = model.predict_proba(review_tfidf)[0][1]

# Convert prediction to label
sentiment = 'Positive' if prediction == 1 else 'Negative'

# Send only final output to stdout
print(f"{sentiment},{probability}")
