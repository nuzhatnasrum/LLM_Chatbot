import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

VECTOR_DB_DIR = "data/vectors/"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

embedding_model = OpenAIEmbeddings(openai_api_key="OPENAI_API_KEY")  # ✅ Correct

def generate_embedding(text_chunks):
    """Generates embeddings for a list of text chunks using OpenAI."""
    return embedding_model.embed_documents(text_chunks)

def store_in_faiss(chunked_text_file, vector_store_name):
    """Reads chunked text, generates embeddings, and stores in both .pkl and .faiss."""
    with open(chunked_text_file, "r", encoding="utf-8") as f:
        chunks = f.readlines()

    documents = [Document(page_content=chunk.strip(), metadata={"source": vector_store_name}) for chunk in chunks if chunk.strip()]
    embeddings = generate_embedding([doc.page_content for doc in documents])

    # ✅ Initialize FAISS index
    faiss_index = FAISS.from_texts([doc.page_content for doc in documents], embedding_model)

    # ✅ Save FAISS index separately
    faiss_file_path = os.path.join(VECTOR_DB_DIR, f"{vector_store_name}.faiss")
    faiss_index.save_local(faiss_file_path)

    # ✅ Save only document metadata in `.pkl`
    docstore_pkl_path = os.path.join(VECTOR_DB_DIR, f"{vector_store_name}.pkl")
    with open(docstore_pkl_path, "wb") as f:
        pickle.dump({"documents": documents}, f)  # ✅ Correct indentation

    return f"Stored {len(documents)} chunks in {faiss_file_path} and {docstore_pkl_path}"

def process_and_store_all():
    """Processes both English and Bangla chunked texts and stores them in FAISS."""
    chunked_text_dir = "data/chunked_text/"
    english_files = ["textbook_english_chunks.txt", "teachers_guide_english_chunks.txt"]
    bangla_files = ["textbook_bangla_chunks.txt", "teachers_guide_bangla_chunks.txt"]

    english_vectors = []
    bangla_vectors = []

    for file in english_files:
        file_path = os.path.join(chunked_text_dir, file)
        if os.path.exists(file_path):
            english_vectors.append(file_path)

    for file in bangla_files:
        file_path = os.path.join(chunked_text_dir, file)
        if os.path.exists(file_path):
            bangla_vectors.append(file_path)

    # Store English documents in FAISS
    if english_vectors:
        store_in_faiss(english_vectors[0], "english_vector_store")  
        if len(english_vectors) > 1:
            store_in_faiss(english_vectors[1], "english_vector_store")  

    # Store Bangla documents in FAISS
    if bangla_vectors:
        store_in_faiss(bangla_vectors[0], "bangla_vector_store")  
        if len(bangla_vectors) > 1:
            store_in_faiss(bangla_vectors[1], "bangla_vector_store")  

    return "FAISS storage completed for both English and Bangla books!"

print(process_and_store_all())