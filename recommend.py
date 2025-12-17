def recommend(query: str, top_k: int = 5):
    return {
        "message": "This function is deprecated. Use /recommend API instead.",
        "query": query,
        "top_k": top_k
    }

if __name__ == "__main__":
    print("recommend.py is inactive. Use app.py")
