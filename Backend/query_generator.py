import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langdetect import detect


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to the environment variables or pass it directly.")

def detect_language(text):
    """Detects whether the text is in English or Bangla."""
    try:
        detected_lang = detect(text)
        if detected_lang == "en":
            return "english"
        elif detected_lang == "bn":
            return "bangla"
        else:
            raise ValueError("Unsupported language detected. Only English and Bangla are supported.")
    except Exception as e:
        raise ValueError(f"Error detecting language: {str(e)}")

def get_store(language: str):
    """Load the FAISS store based on language."""
    store_name = f"{language}_vector_store"
    store_path = f"data/vectors/{store_name}"
    store = FAISS.load_local(store_path, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
    return store

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

def process_query(query: str):
    language = detect_language(query)

    if language == "unknown":
        raise ValueError("Unsupported language detected. Only English and Bangla are supported.")

    store = get_store(language)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=store.as_retriever()
    )

    result = chain({"query": query})
    return result
query = "What is pithagorus theorum?"  # Example query

try:
    response = process_query(query)
    print("Response:", response["result"])
except Exception as e:
    print("Error:", str(e))

