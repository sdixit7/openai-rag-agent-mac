import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DOCS_DIR = Path("data/docs")
DB_DIR = "db"
COLLECTION_NAME = "my_docs"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_texts(texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]

def main():
    docs = []
    ids = []
    metadatas = []

    i = 0
    for file_path in DOCS_DIR.iterdir():
        if file_path.suffix.lower() == ".txt":
            text = read_txt(file_path)
        elif file_path.suffix.lower() == ".pdf":
            text = read_pdf(file_path)
        else:
            continue

        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)

        for chunk, embedding in zip(chunks, embeddings):
            docs.append(chunk)
            ids.append(f"doc_{i}")
            metadatas.append({"source": file_path.name})
            i += 1

        collection.add(
            ids=ids[-len(chunks):],
            documents=docs[-len(chunks):],
            embeddings=embeddings,
            metadatas=metadatas[-len(chunks):]
        )

    print(f"Ingested {len(ids)} chunks into Chroma.")

if __name__ == "__main__":
    main()
