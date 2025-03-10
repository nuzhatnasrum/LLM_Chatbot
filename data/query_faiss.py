import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# ✅ Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing! Check your .env file.")

VECTOR_DB_DIR = "data/vectors/"
bangla_pkl_path = os.path.join(VECTOR_DB_DIR, "bangla_vector_store.pkl")
english_pkl_path = os.path.join(VECTOR_DB_DIR, "english_vector_store.pkl")

# ✅ Load FAISS Index from .pkl
def load_faiss_from_pkl(pkl_path):
    """Loads FAISS index and stored text chunks from a .pkl file."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"❌ {pkl_path} not found! Run `faiss_index.py` first.")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return data["faiss_index"], data["documents"]

# ✅ Load both FAISS indexes
bangla_index, bangla_chunks = load_faiss_from_pkl(bangla_pkl_path)
english_index, english_chunks = load_faiss_from_pkl(english_pkl_path)

# ✅ Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def retrieve_text(query, language="bangla", top_k=2):
    """Retrieves the most relevant text chunks for a given query."""
    
    # ✅ Convert Query to Vector
    query_vector = np.array([embedding_model.embed_query(query)]).astype("float32")

    # ✅ Select Index & Text Chunks Based on Language
    if language.lower() == "bangla":
        index, text_chunks = bangla_index, bangla_chunks
    else:
        index, text_chunks = english_index, english_chunks

    # ✅ Perform Similarity Search
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        if idx < 0 or idx >= len(text_chunks):  # Avoid invalid index errors
            continue
        chunk_text = text_chunks[idx].page_content  # Extract content from Document object
        results.append((chunk_text, distances[0][i]))

    return results if results else [("No relevant text found.", None)]

# ✅ Example Queries
query_bangla = "বাংলাদেশের স্বাধীনতা যুদ্ধ কবে সংঘটিত হয়?"
query_english = "When did the independence war of Bangladesh occur?"

# ✅ Retrieve Results
print("\n🔍 Bangla Search Results:")
for text, distance in retrieve_text(query_bangla, language="bangla"):
    print(f"- {text} (Distance: {distance})")

print("\n🔍 English Search Results:")
for text, distance in retrieve_text(query_english, language="english"):
    print(f"- {text} (Distance: {distance})")
