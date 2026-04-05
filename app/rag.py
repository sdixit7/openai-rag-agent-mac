import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DB_DIR = "db"
COLLECTION_NAME = "my_docs"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def embed_query(text: str):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response.data[0].embedding

def retrieve_context(query: str, top_k: int = 3):
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return docs, metas

def ask_rag(query: str) -> str:
    docs, metas = retrieve_context(query)

    context = "\n\n".join(docs) if docs else "No relevant context found."

    prompt = f"""
Use the context below to answer the question.
If the answer is not in the context, say you are not sure.

Context:
{context}

Question:
{query}
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )
    return response.output_text
