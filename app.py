from fastapi import FastAPI
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI(title="SHL Assessment Recommendation API")

# Load data
df = pd.read_csv("shl_catalog.csv")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
texts = df["NAME"].tolist()
embeddings = model.encode(texts)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

@app.get("/")
def health():
    return {"status": "API is running"}

@app.get("/recommend")
def recommend(query: str, top_k: int = 5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = df.iloc[indices[0]][["NAME", "URL", "TEST_TYPE"]]
    return {
        "query": query,
        "results": results.to_dict(orient="records")
    }
