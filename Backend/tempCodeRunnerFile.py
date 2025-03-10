import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Fetch the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the key is loaded properly
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")