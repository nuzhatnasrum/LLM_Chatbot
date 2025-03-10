import os
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# ✅ Load API Key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing! Check your .env file.")

# ✅ Load FAISS Indices
if not os.path.exists("faiss_bangla.bin") or not os.path.exists("faiss_english.bin"):
    raise FileNotFoundError("❌ FAISS index files not found! Ensure they are generated.")

bangla_index = faiss.read_index("faiss_bangla.bin")
english_index = faiss.read_index("faiss_english.bin")

# ✅ Load Text Chunks
def load_text_chunks(file_path):
    """Loads text chunks from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

bangla_chunks = load_text_chunks("data/chunked_text/textbook_bangla_chunks.txt") + \
                load_text_chunks("data/chunked_text/teachers_guide_bangla_chunks.txt")

english_chunks = load_text_chunks("data/chunked_text/textbook_english_chunks.txt") + \
                 load_text_chunks("data/chunked_text/teachers_guide_english_chunks.txt")

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
        chunk_text = text_chunks[idx]
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
