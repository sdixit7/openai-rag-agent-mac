from llm import ask_llm
from rag import ask_rag

def calculator_tool(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception:
        return "Sorry, I could not calculate that."

def run_agent(user_input: str) -> str:
    text = user_input.lower()

    if any(word in text for word in ["document", "pdf", "notes", "file", "from my docs", "from my notes"]):
        return ask_rag(user_input)

    if any(ch in text for ch in ["+", "-", "*", "/"]) and any(char.isdigit() for char in text):
        return calculator_tool(user_input)

    return ask_llm(user_input)
