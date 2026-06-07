import ast
from collections import Counter

import pandas as pd

print("Loading dataset...")
df = pd.read_csv("hf://datasets/CodeKapital/CookingRecipes/Data.csv")
print(f"Total recipes: {len(df)}")

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
