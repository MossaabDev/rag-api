from fastapi import FastAPI
from pydantic import BaseModel
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import nltk

# Download tokenizer
nltk.download("punkt")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Init FastAPI app
app = FastAPI()

# Define input format
class TextInput(BaseModel):
    text: str
    max_sentences: int = 1

# Helper: chunk text
def chunk_text(text, max_sentences=1):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        chunks.append(chunk)
    return chunks

# Route: POST /embed
@app.post("/embed")
def embed_text(input_data: TextInput):
    chunks = chunk_text(input_data.text, input_data.max_sentences)
    results = []
    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        results.append({
            "chunk": chunk,
            "embedding": embedding
        })
    return {"results": results}
