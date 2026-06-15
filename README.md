Wybrany temat to chatbot kulinarny wykorzystujący RAG, korzystający z datasetu https://huggingface.co/datasets/CodeKapital/CookingRecipes. Przepisy zaindeksowane zostały jako embeddingi w wektorowej bazie FAISS, a wyszukane dokumenty służą jako kontekst dla modelu językowego generującego odpowiedzi. W ramach ewaluacji porównana została jakość retrieval (FAISS vs BM25) na zestawie zapytań testowych.

## Struktura plików

```
WDSI-P3/
├── ingest.py          # (KROK 1) Buduje indeksy FAISS i BM25 z datasetu
├── main.py            # (KROK 2) Chatbot kulinarny RAG - interfejs konsolowy
├── compare.py         # (KROK 3) Ewaluacja: porównanie FAISS vs BM25 na zapytaniach testowych
├── stats.py           # Pomocniczy: analiza częstości składników w datasecie
├── requirements.txt   # Zależności Python
├── .env.example       # Szablon pliku z kluczem API
├── data/              # Generowany przez ingest.py (ignorowany przez git)
│   ├── faiss.index    # Indeks wektorowy FAISS
│   ├── bm25.pkl       # Indeks BM25 (pickle)
│   └── docs.pkl       # Lista dokumentów (pickle)
└── compare_results/   # Generowany przez compare.py
    ├── compare_result_1.json
    ├── compare_result_2.json
    └── ...
```

## Odtworzenie wyników krok po kroku

### 0. Wymagania wstępne

```bash
pip install -r requirements.txt
```

Należy skopiować plik `.env.example` do `.env` i uzupełnić klucz Groq API:

```
GROQ_API_KEY=klucz_api
```

Klucz dostępny bezpłatnie pod adresem https://console.groq.com

### 1. Zbudowanie indeksów (`ingest.py`)

```bash
python ingest.py
```

Skrypt pobiera dataset z HuggingFace (~2.2M przepisów), filtruje do przepisów
zawierających ziemniaki, buduje indeks FAISS i BM25, a następnie zapisuje je
do folderu `data/`. Operacja trwa ~30 minut.

### 2. Uruchomienie chatbota (`main.py`)

```bash
python main.py
```

Na starcie należy wybrać metodę retrieval:

```
Retriever [faiss/bm25/both]: both
```

Następnie możliwe jest zadawanie pytań w języku naturalnym. Wpisanie `quit` kończy działanie programu.

### 3. Porównanie metod retrieval (`compare.py`)

```bash
python compare.py
```

Skrypt uruchamia oba retrievery na 5 zapytaniach testowych i zapisuje wyniki
do `compare_results/compare_result_N.json`. Każdy plik zawiera Top-5 wyników
obu metod (tytuł przepisu, wynik, kroki przygotowania).

Zapytania testowe i ich cel:

| # | Zapytanie | Spodziewany zwycięzca |
|---|-----------|----------------------|
| 1 | `hearty warm dish for cold weather` | FAISS (semantyczne) |
| 2 | `traditional Eastern European side dish` | FAISS (geograficzny kontekst) |
| 3 | `what to cook with fish, potatoes and carrot?` | BM25 (konkretne składniki) |
| 4 | `potato dish without cheese or dairy` | Obie metody zawodzą (negacja) |
| 5 | `quick potato dish ready in under 15 minutes` | Obie metody zawodzą (ograniczenie czasowe) |
