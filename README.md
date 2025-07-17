# 🎬 Sentiment Analyzer — Movie‑Review Classifier

A lightweight web app that judges the **sentiment** (positive / negative) of user‑submitted movie reviews.

**Built with**

| Layer | Tech |
|-------|------|
| Model | `TfidfVectorizer` + `LogisticRegression` (scikit‑learn) |
| API   | FastAPI |
| UI    | Vanilla HTML / CSS / JS |
| Deploy (suggested) | Render (API) · Netlify (frontend) |

---

## 🚀 Live Demo

> Paste a review, hit **Check Sentiment**, get instant feedback.

![Screenshot](screenshots/demo.png)

 Check out the live app here: [Sentiment Analyzer on Netlify](https://connors-sentiment-analyser.netlify.app)
`▶️ https://connors-sentiment-analyser.netlify.app`

Note: The deployed model was trained locally on a sample of the IMDb dataset.
For full reproducibility, follow the setup steps and run the training script.

First request may take 10–30s due to cold start (Render free tier)

---

## 🔍 How It Works

1. **Data** – Sample of 2 000 labelled reviews from the [IMDb Large Movie Review corpus](http://ai.stanford.edu/~amaas/data/sentiment/).  
2. **Vectorize** – Convert text → TF‑IDF features (`max_features=5 000`).  
3. **Model** – Train a logistic‑regression binary classifier (> 80 % accuracy on hold‑out set).  
4. **Serve** – Load the model once, expose `POST /predict` returning JSON:  

   ```json
   { "sentiment": "positive", "confidence": 0.92 }
   ```
5. **UI** – Browser calls the endpoint via `fetch()` and colours the result green/red.

---

## 🛠️ Run Locally

```bash
# 1  Clone
git clone https://github.com/CMcLeodC/sentiment-analyzer.git
cd sentiment-analyzer

# 2  (Optional) venv
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3  Install deps
pip install -r requirements.txt

# 4  (First time) prepare data + train
python data/prepare_data.py      # creates imdb_small.csv
python backend/train_model.py    # saves model & vectorizer

# 5  Run API
python -m uvicorn backend.main:app --reload

# 6  Open UI
open frontend/index.html         # or just double‑click
```

---

## 🚀 Deploy in 5 min

| Part | Platform | How |
|------|----------|-----|
| **API** | **Render** (Free Web Service) | New → Web Service → connect repo → Build command `pip install -r requirements.txt && python backend/train_model.py` → Start command `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| **Frontend** | **Netlify** (Drop‑in) | Drag‑and‑drop `frontend/` or connect repo. Edit `fetch()` URL in `index.html` to Render endpoint. |

---

## 🧭 Why I Built This

As an AI engineer with a background in education, I wanted a hands-on project that would give me experience with training a real ML model (start to finish), including vectorization, model loading

I chose sentiment analysis because it's a classic NLP task with a clear use case and simple user interaction. The perfect starting-off point.

---

## 🧠 What I Learned

- How to preprocess real-world text data
- How to use TF-IDF and Logistic Regression for classification
- How to serve ML models with FastAPI

---

## 📂 Project Structure

```
sentiment-analyzer/
├─ backend/
│  ├─ main.py            ← FastAPI app
│  ├─ train_model.py      ← model training script
│  ├─ sentiment_model.pkl │
│  └─ tfidf_vectorizer.pkl│
├─ data/
│  ├─ download_dataset.py ← grabs IMDb corpus
│  └─ prepare_data.py     ← builds imdb_small.csv
├─ frontend/
│  └─ index.html          ← minimal UI
├─ requirements.txt
└─ README.md
```

---


> _Built by **Connor Clements** – AI engineer._  
> 🔗 [[LinkedIn](https://www.linkedin.com/in/connor-andrew-clements/)]
