import json
import os
import pickle
import re

import faiss
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
BM25_PATH = os.path.join(DATA_DIR, "bm25.pkl")
DOCS_PATH = os.path.join(DATA_DIR, "docs.pkl")
RESULTS_DIR = "compare_results"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

QUERIES = [
    "hearty warm dish for cold weather",            # FAISS better (semantic understanding / vibe)
    "traditional Eastern European side dish",       # FAISS better, but still wrong (geographical context)
    "what to cook with fish, potatoes and carrot?", # BM25 better (exact ingredient matching)
    "potato dish without cheese or dairy",          # Both fail (negation trap - ignores "without")
    "quick potato dish ready in under 15 minutes"   # Both fail (numerical constraint - needs metadata filtering)
]


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return [w for w in text.split() if w not in STOP_WORDS]


def retrieve_faiss(query, index, docs, model, top_k=TOP_K):
    embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(embedding)
    scores, indices = index.search(embedding, top_k)
    return [
        {
            "rank": i + 1,
            "title": docs[idx]["title"],
            "score": round(float(score), 4),
            "directions": docs[idx]["text"].split("Directions:")[-1].strip(),
        }
        for i, (score, idx) in enumerate(zip(scores[0], indices[0]))
    ]


def retrieve_bm25(query, bm25, docs, top_k=TOP_K):
    tokenized = tokenize(query)
    scores = bm25.get_scores(tokenized)
    top_indices = scores.argsort()[-top_k:][::-1]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        score = float(scores[idx])
        if score > 0:
            results.append(
                {
                    "rank": rank,
                    "title": docs[idx]["title"],
                    "score": round(score, 4),
                    "directions": docs[idx]["text"].split("Directions:")[-1].strip(),
                }
            )
    return results


def main():
    print("Loading indexes...")
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    model = SentenceTransformer(EMBEDDING_MODEL)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for i, query in enumerate(QUERIES, 1):
        print(f"Running query {i}/{len(QUERIES)}: '{query}'")

        faiss_results = retrieve_faiss(query, index, docs, model)
        bm25_results = retrieve_bm25(query, bm25, docs)

        result = {
            "query_id": i,
            "query": query,
            "faiss": faiss_results,
            "bm25": bm25_results,
        }

        out_path = os.path.join(RESULTS_DIR, f"compare_result_{i}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  FAISS top1: {faiss_results[0]['title'] if faiss_results else 'none'} "
              f"(score: {faiss_results[0]['score'] if faiss_results else '-'})")
        print(f"  BM25  top1: {bm25_results[0]['title'] if bm25_results else 'none'} "
              f"(score: {bm25_results[0]['score'] if bm25_results else '-'})")
        print(f"  Saved → {out_path}")

    print(f"\nDone. {len(QUERIES)} result files saved to '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()
