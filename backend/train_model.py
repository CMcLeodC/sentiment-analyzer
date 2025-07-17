import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv("imdb_small.csv")

# Features and labels
X = df["review"]
y = df["label"].map({"pos": 1, "neg": 0})  # convert to 1/0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Define save path
MODEL_DIR = os.path.join("backend", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model + vectorizer
joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
print("Model and vectorizer saved ✅")
