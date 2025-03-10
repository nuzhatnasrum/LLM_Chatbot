from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing! Please check your .env file.")

# Initialize the AI model
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=API_KEY)

def handle_query(user_input: str):
    """Processes user input using OpenAI Chat Model and returns a response."""
    print(f"🔍 Received Query: {user_input}")  # Debugging

    try:
        response = llm.invoke(user_input)  # ✅ Get full response
        
        # ✅ Extract only the answer
        if hasattr(response, "content"):  
            response_text = response.content  
        else:
            response_text = str(response)  

        print(f"✅ AI Response: {response_text}")  # Debugging
        return response_text

    except Exception as e:
        print(f"❌ Error: {e}")
        return "Sorry, an error occurred while processing your query."
