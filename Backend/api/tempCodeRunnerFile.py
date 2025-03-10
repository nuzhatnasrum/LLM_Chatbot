from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
from app.services.chat_handler import handle_query  # ✅ Use relative module path


# ✅ Load API Key from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("⚠️ API_KEY is missing! Please check your .env file.")

# ✅ Initialize FastAPI
chatbot_api = FastAPI()

# ✅ Define Request Model
class UserQuery(BaseModel):
    question: str  # ✅ Renamed 'query' to 'question'

# ✅ Define Chatbot Endpoint
@chatbot_api.post("/query/")
async def get_response(user_input: UserQuery):
    """Processes user query and returns chatbot response."""
    try:
        answer = handle_query(user_input.question)  # ✅ Using modified function name
        return {"response": answer}
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail="Internal server error.")
