import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load data
df = pd.read_csv("shl_catalog.csv")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert test names to embeddings
texts = df["NAME"].tolist()
embeddings = model.encode(texts, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Recommendation function
def recommend(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return df.iloc[indices[0]]

# Test the recommender
if __name__ == "__main__":
    query = "Looking for .NET developer"
    results = recommend(query)
    print("\nRecommendations for:", query)
    print(results[["NAME", "URL", "TEST_TYPE"]])
