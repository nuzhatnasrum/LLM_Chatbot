import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/query/"


st.set_page_config(page_title="LLM Chatbot", layout="centered")
st.title("🤖 LLM Chatbot")


user_query = st.text_input("Ask me anything:", "")

if st.button("Get Answer"):
    if user_query.strip():
        try:
            response = requests.post(API_URL, json={"question": user_query})
            if response.status_code == 200:
                st.success(response.json()["response"])
            else:
                st.error("❌ Failed to get response. Check API!")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please enter a query.")
