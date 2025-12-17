from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="SHL Assessment Recommendation API")

# Load data
df = pd.read_csv("shl_catalog.csv")

# Vectorizer (lightweight)
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["NAME"])

@app.get("/")
def health():
    return {"status": "API is running"}

@app.get("/recommend")
def recommend(query: str, top_k: int = 5):
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices][["NAME", "URL", "TEST_TYPE"]]

    return {
        "query": query,
        "results": results.to_dict(orient="records")
    }
