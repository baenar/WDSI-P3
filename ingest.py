import os
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
DOCS_PATH = os.path.join(DATA_DIR, "docs.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_dataset():
    print("Downloading dataset...")
    df = pd.read_csv("hf://datasets/CodeKapital/CookingRecipes/Data.csv")
    print(f"Loaded {len(df)} recipes")
    return df


FILTER_INGREDIENT = "potatoes"


def filter_by_ingredient(df, ingredient):
    import ast

    def has_ingredient(ner):
        try:
            return ingredient.lower() in [i.lower().strip() for i in ast.literal_eval(ner)]
        except Exception:
            return False

    mask = df["NER"].apply(has_ingredient)
    filtered = df[mask].reset_index(drop=True)
    print(f"Filtered to {len(filtered)} recipes containing '{ingredient}'")
    return filtered


def build_documents(df):
    docs = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        ingredients = str(row.get("ingredients", ""))
        directions = str(row.get("directions", ""))
        text = f"Title: {title}\nIngredients: {ingredients}\nDirections: {directions}"
        docs.append({"title": title, "text": text})
    return docs


def build_index(docs):
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Creating embeddings (this may take a while)...")
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_dataset()
    df = filter_by_ingredient(df, FILTER_INGREDIENT)
    docs = build_documents(df)

    index = build_index(docs)

    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"Saved FAISS index ({index.ntotal} vectors) to {INDEX_PATH}")
    print(f"Saved documents to {DOCS_PATH}")


if __name__ == "__main__":
    main()
