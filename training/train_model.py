import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the first 10,000 rows of Yelp dataset
yelp_data = pd.read_json('D:/projects/sentiment-analysis-project/training/yelp_review.json', lines=True, nrows=10000)

# Filter out neutral reviews (3 stars) and map stars to sentiment
yelp_data = yelp_data[yelp_data['stars'] != 3]
yelp_data['sentiment'] = yelp_data['stars'].apply(lambda x: 1 if x >= 4 else 0)

# Extract reviews and sentiments
reviews = yelp_data['text']
sentiments = yelp_data['sentiment']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model and vectorizer
joblib.dump(model, '../backend/sentiment_model.pkl')
joblib.dump(vectorizer, '../backend/tfidf_vectorizer.pkl')
print("Model and vectorizer saved to backend folder.")