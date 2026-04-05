import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

def ask_llm(user_message: str, system_prompt: str = "You are a helpful beginner-friendly AI assistant.") -> str:
    response = client.responses.create(
        model=MODEL,
        instructions=system_prompt,
        input=user_message
    )
    return response.output_text
