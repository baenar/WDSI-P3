# Główny plik chatbota kulinarnego opartego na RAG.
# Wczytuje indeksy zbudowane przez ingest.py, a następnie w pętli przyjmuje
# pytania od użytkownika, wyszukuje Top-K przepisów wybraną metodą (FAISS,
# BM25 lub obie naraz) i przesyła je jako kontekst do modelu Llama 3.3-70b
# przez Groq API, wyświetlając odpowiedź strumieniowo.

import os
import re
import pickle
import faiss
import nltk

from nltk.corpus import stopwords
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
BM25_PATH = os.path.join(DATA_DIR, "bm25.pkl")
DOCS_PATH = os.path.join(DATA_DIR, "docs.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a helpful culinary assistant. Use the provided recipes as context "
    "to answer the user's question. If the context doesn't contain relevant "
    "information, say so. Answer in the same language as the user's question."
)

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    return index, docs, bm25


def retrieve_faiss(query, index, docs, model, top_k=TOP_K):
    embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(embedding)
    scores, indices = index.search(embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({"score": float(score), **docs[idx]})
    return results


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return [word for word in words if word not in STOP_WORDS]


def retrieve_bm25(query, bm25_index, docs, top_k=TOP_K):
    tokenized_query = tokenize(query)
    
    doc_scores = bm25_index.get_scores(tokenized_query)
    top_indices = doc_scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        score = doc_scores[idx]
        if score > 0:
            results.append({"score": float(score), **docs[idx]})
    return results


def build_prompt(query, results):
    context = "\n\n---\n\n".join(r["text"] for r in results)
    return f"Context (retrieved recipes):\n{context}\n\nUser question: {query}"


def main():
    print("Loading index...")
    index, docs, bm25 = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = Groq()

    while True:
        mode = input("Retriever [faiss/bm25/both]: ").strip().lower()
        if mode in ("faiss", "bm25", "both"):
            break
        print("Please type faiss, bm25 or both.")

    print(f"\nCulinary chatbot ready! (mode: {mode}) Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if not query or query.lower() == "quit":
            break

        if mode == "faiss":
            results = retrieve_faiss(query, index, docs, model)
            print(f"\n--- Top {TOP_K} FAISS results ---")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']} (score: {r['score']:.3f})")
            print("---------------------------------\n")

        elif mode == "bm25":
            results = retrieve_bm25(query, bm25, docs)
            print(f"\n--- Top {TOP_K} BM25 results ---")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']} (score: {r['score']:.3f})")
            print("--------------------------------\n")

        else:  # both
            faiss_results = retrieve_faiss(query, index, docs, model)
            bm25_results = retrieve_bm25(query, bm25, docs)

            print(f"\n--- Top {TOP_K} FAISS results ---")
            for i, r in enumerate(faiss_results, 1):
                print(f"{i}. {r['title']} (score: {r['score']:.3f})")

            print(f"\n--- Top {TOP_K} BM25 results ---")
            for i, r in enumerate(bm25_results, 1):
                print(f"{i}. {r['title']} (score: {r['score']:.3f})")
            print("--------------------------------\n")

            results = faiss_results + bm25_results

        user_message = build_prompt(query, results)

        print("Bot: ", end="", flush=True)
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_completion_tokens=1024,
            stream=True,
        )
        for chunk in completion:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
