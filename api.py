# main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import nltk

# Init
app = FastAPI()
nltk.download('punkt')
model = SentenceTransformer("all-MiniLM-L6-v2")

class TextInput(BaseModel):
    text: str
    max_sentences: int = 1

@app.post("/embed")
def embed_report(data: TextInput):
    sentences = sent_tokenize(data.text)
    chunks = []
    for i in range(0, len(sentences), data.max_sentences):
        chunk = " ".join(sentences[i:i+data.max_sentences])
        embedding = model.encode(chunk).tolist()
        chunks.append({
            "chunk": chunk,
            "embedding": embedding
        })
    return {"result": chunks}
