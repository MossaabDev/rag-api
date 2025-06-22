import nltk
nltk.download('punkt_tab')  # Download the punkt tokenizer models
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def chunk_text(text, max_sentences=2):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        chunks.append(chunk)
    return chunks

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # small + fast

def get_embedding(text):
    return model.encode(text).tolist()

report = "Lezrak Mohamed Haitem, 22 years old, underwent three blood tests between 23 December 2023 and 10 January 2024. On 23 December, he showed high levels of bilirubin (3.1 mg/dL), with slightly elevated liver enzymes (ALT and AST), suggesting possible liver stress or early signs of hepatitis. His white blood cell count was normal at that time. By 5 January 2024, his bilirubin level had decreased to 1.6 mg/dL, indicating improvement, but still above the normal range. However, his white blood cells increased significantly, suggesting a possible immune response or inflammation, potentially related to a recent infection or recovery phase. On 10 January 2024, all test results returned to normal, including bilirubin and WBC. No further abnormalities were detected, and no ongoing condition was evident. His recovery appears complete, with no current signs of liver or immune system issues. No further follow-up is necessary unless symptoms return."
chunks = chunk_text(report, max_sentences=1)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}\n")
    embedding = get_embedding(chunk)
    print(f"Embedding {i+1}: {embedding}\n")
    print("-" * 80)
    


