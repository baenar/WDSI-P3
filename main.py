import os
import pickle

import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
DOCS_PATH = os.path.join(DATA_DIR, "docs.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a helpful culinary assistant. Use the provided recipes as context "
    "to answer the user's question. If the context doesn't contain relevant "
    "information, say so. Answer in the same language as the user's question."
)


def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    return index, docs


def retrieve(query, index, docs, model, top_k=TOP_K):
    embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(embedding)
    scores, indices = index.search(embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({"score": float(score), **docs[idx]})
    return results


def build_prompt(query, results):
    context = "\n\n---\n\n".join(r["text"] for r in results)
    return f"Context (retrieved recipes):\n{context}\n\nUser question: {query}"


def main():
    print("Loading index...")
    index, docs = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = Groq()

    print("Culinary chatbot ready! Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if not query or query.lower() == "quit":
            break

        results = retrieve(query, index, docs, model)

        print("\n--- Top 3 retrieved recipes ---")
        for i, r in enumerate(results[:3], 1):
            print(f"{i}. {r['title']} (score: {r['score']:.3f})")
        print("-------------------------------\n")

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
