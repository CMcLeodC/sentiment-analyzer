from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# Load model + vectorizer at startup
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

app = FastAPI()

# Allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request format
class ReviewRequest(BaseModel):
    review: str

# POST endpoint
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    text = request.review

    if not text.strip():
        raise HTTPException(status_code=400, detail="Review text is empty")

    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]

    sentiment = "positive" if prediction == 1 else "negative"
    confidence = float(np.max(proba))

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 3)
    }
