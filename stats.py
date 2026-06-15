# Skrypt pomocniczy do analizy datasetu.
# Pobiera pełny dataset CookingRecipes, filtruje przepisy z ziemniakami,
# a następnie zlicza częstość wystąpień każdego składnika w kolumnie NER.
# Służył do podjęcia decyzji o wyborze składnika filtrującego dataset
# przed uruchomieniem ingest.py (nie jest potrzebny do działania chatbota).

import ast
from collections import Counter

import pandas as pd

print("Loading dataset...")
df = pd.read_csv("hf://datasets/CodeKapital/CookingRecipes/Data.csv")
print(f"Total recipes: {len(df)}")

FILTER_INGREDIENT = "potatoes"

def has_ingredient(ner):
        try:
            return FILTER_INGREDIENT.lower() in [i.lower().strip() for i in ast.literal_eval(ner)]
        except Exception:
            return False

mask = df["NER"].apply(has_ingredient)
filtered = df[mask].reset_index(drop=True)
print(f"Filtered to {len(filtered)} recipes containing '{FILTER_INGREDIENT}'")

df = filtered

counter = Counter()
for ner in df["NER"].dropna():
    try:
        ingredients = ast.literal_eval(ner)
        counter.update(i.lower().strip() for i in ingredients)
    except Exception:
        pass

print("\nTop 50 ingredients:\n")
print(f"{'Rank':<6} {'Ingredient':<30} {'Count':>10}")
print("-" * 48)
for rank, (ingredient, count) in enumerate(counter.most_common(50), 1):
    print(f"{rank:<6} {ingredient:<30} {count:>10,}")
