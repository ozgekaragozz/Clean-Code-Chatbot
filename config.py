import os
from dotenv import load_dotenv
from pathlib import Path 

load_dotenv(dotenv_path=Path('.') / '.env')

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_DEPLOYMENT_NAME:
    raise ValueError("Azure OpenAI API settings are missing! Please check the .env file.")
