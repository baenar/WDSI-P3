# Kulinarny Chatbot oparty na RAG
## Porównanie metod wyszukiwania: FAISS vs BM25

**Autorzy:** Mikołaj Bejnar, Jakub Bereza  
**Data:** 15 czerwca 2026

---

## 1. Wprowadzenie

### 1.1. Opis problemu
W dobie ogromnej dostępności danych w internecie, w tym milionów przepisów kulinarnych, tradycyjne systemy wyszukiwania często nie spełniają oczekiwań użytkowników. Wyszukiwanie oparte na słowach kluczowych (leksykalne) jest precyzyjne, ale nie radzi sobie z synonimami i abstrakcyjnymi intencjami (np. ,,sycące zimowe danie''). Z kolei nowoczesne duże modele językowe (LLM), choć wysoce elastyczne w rozumieniu zapytań, mają tendencję do tzw. halucynacji faktograficznych, jeśli nie zostaną osadzone w twardych, zewnętrznych danych. 

Rozwiązaniem tego problemu jest architektura RAG (Retrieval-Augmented Generation). Podejście to polega na zasileniu modelu językowego zewnętrzną, zweryfikowaną bazą wiedzy. Proces składa się z dwóch głównych etapów: najpierw system przeszukuje bazę w celu znalezienia (ang. *Retrieval*) dokumentów najbardziej trafnych dla zapytania użytkownika. Następnie wyselekcjonowane dane są dołączane do promptu jako kontekst, na podstawie którego model formułuje i generuje (ang. *Generation*) ostateczną odpowiedź. O sukcesie całej architektury, a w konsekwencji o rzetelności odpowiedzi chatbota, decyduje jednak skuteczność samego mechanizmu pobierania dokumentów (Retrievera).

### 1.2. Cel projektu
Głównym celem projektu było zaprojektowanie, zaimplementowanie i krytyczna ewaluacja kulinarnego chatbota opartego na architekturze RAG. Projekt skupiał się na zestawieniu dwóch odmiennych paradygmatów wyszukiwania informacji: wektorowego (FAISS) oraz statystycznego (BM25). Na podstawie przeprowadzonych eksperymentów celem było zbadanie mocnych i słabych stron obu rozwiązań przy różnych typach zapytań użytkownika, a ostatecznie wykazanie zasadności wdrożenia wyszukiwania hybrydowego.

---

## 2. Metody i Dane

### 2.1. Dane
Projekt wykorzystuje anglojęzyczny zbiór danych *CookingRecipes* dostępny na platformie HuggingFace, zawierający bazowo ponad 2.2 miliona przepisów kulinarnych. Z powodu ograniczeń obliczeniowych, do testów wyekstrahowano podzbiór danych (odfiltrowany do przepisów zawierających m.in. ziemniaki). Każdy dokument w bazie został skonkatenowany do ujednoliconego łańcucha znaków według schematu: tytuł, lista składników oraz instrukcje wykonania (Title + Ingredients + Directions), co stanowiło ostateczny kontekst dla modelu.

### 2.2. Technologia i Metody
System opiera się na trójetapowym przepływie działania: przetworzenie zapytania, wyszukanie najlepiej dopasowanych dokumentów (Top-K) oraz przekazanie ich jako kontekstu do dużego modelu językowego (LLM). Do generacji tekstu wykorzystano model `llama-3.3-70b-versatile` za pośrednictwem Groq API (parametry: streaming aktywny, *temperature*=0.7, *max_tokens*=1024), wyposażony w odpowiedni *system prompt* asystenta kulinarnego.

Podsystem wyszukiwania (Retriever) zrealizowano w dwóch wariantach:

1. **Wyszukiwanie semantyczne (FAISS):**
   Każdy przepis został przetworzony na wektor osadzeń (embedding) w przestrzeni $\mathbb{R}^{384}$ przy użyciu modelu `all-MiniLM-L6-v2` z biblioteki *sentence-transformers*. Wyszukiwanie w bazie realizowano za pomocą biblioteki FAISS (Facebook AI Similarity Search). Po uprzedniej normalizacji wektorów L2 zastosowano indeks `IndexFlatIP`, sprowadzając wyszukiwanie do obliczania iloczynu skalarnego, odpowiadającego w tym przypadku podobieństwu kosinusowemu: $\text{sim}(q, e_i) = q \cdot e_i$.
   
2. **Wyszukiwanie słów kluczowych (BM25):**
   Druga metoda wykorzystywała algorytm *Okapi BM25*. Zastosowano rygorystyczny preprocessing oparty na bibliotece NLTK (usunięcie znaków interpunkcyjnych, konwersja na małe litery, usunięcie stop-słów dla języka angielskiego). Ocenę trafności dokumentu $d$ w stosunku do zapytania $q$ obliczano ze standardowego wzoru BM25:

   $$score(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d)(k_1 + 1)}{f(t, d) + k_1\left(1 - b + b \frac{|d|}{\text{avgdl}}\right)}$$

   gdzie przyjęto standardowe parametry $k_1 = 1.5$ oraz $b = 0.75$. Dodatkowo, $f(t,d)$ oznacza częstotliwość termu w dokumencie, funkcja $\text{IDF}(t)$ (Inverse Document Frequency) to odwrotna częstotliwość dokumentowa, która nadaje wyższą wagę słowom rzadkim i unikalnym w całym zbiorze danych. Z kolei człon $|d|/\text{avgdl}$ to stosunek długości ocenianego dokumentu do średniej długości wszystkich przepisów w bazie, pełniący funkcję normalizującą (zapobiega niesprawiedliwemu faworyzowaniu bardzo długich tekstów).

---

## 3. Wyniki

### 3.1. Prezentacja i porównanie pełnych wyników (Top 3)
Zbudowany system działał poprawnie w czasie rzeczywistym, sprawnie komunikując się z modelem Llama 3 poprzez API i zwracając zwięzłe rekomendacje poparte wyciągniętym kontekstem. Poniższe zestawienie tabelaryczne przedstawia pełne wyniki Top 3 dla obu algorytmów, co pozwala na bezpośrednią ocenę jakości zwracanego kontekstu:

| Zapytanie testowe | FAISS – Wyniki Top 3 (Wynik) | BM25 – Wyniki Top 3 (Wynik) | Obserwacja / Status |
| :--- | :--- | :--- | :--- |
| **1. Semantyka:**<br>*Hearty warm dish for cold weather* | 1. Hearty Hot Dish (0.590)<br>2. Hearty Winter Soup (0.580)<br>3. Hearty Winter Soup (0.573) | 1. Potato Sourdough Starter (19.4)<br>2. Cold Weather Soup (15.4)<br>3. Potato Soup For Hot Weather (12.2) | **FAISS wygrywa**.<br>BM25 zawodzi na poz. #1 przez słowo *,,warm''* w instrukcji (*,,warm water''*). |
| **2. Składniki:**<br>*What to cook with fish, potatoes and carrot?* | 1. Baked Fish With Potatoes (0.763)<br>2. Baked Fish On Potatoes (0.754)<br>3. Barb. Fish, Crisp Potatoes (0.753) | 1. Fish Chowder (17.3)<br>2. Fish Chowder (16.1)<br>3. Rainbow Trout w/ Potatoes (15.7) | **BM25 wygrywa**.<br>FAISS gubi marchewkę, skupiając się na ciężkich wektorach ryby i ziemniaków. |
| **3. Koncepcja:**<br>*Traditional Eastern European side dish* | 1. Germanic Casserole (0.527)<br>2. German Dinner (0.519)<br>3. Gjelle Me Zarzavata (0.504) | 1. Traditional Eastern Curry (14.1)<br>2. Potato-Pea Salad (13.1)<br>3. Hangover Soup (12.6) | **FAISS wygrywa**.<br>BM25 promuje Curry na poz. #1 tylko za dosłowne słowa *,,Traditional Eastern''*. |
| **4. Liczby (Czas):**<br>*Quick potato dish ready in under 15 minutes* | 1. Quick-Baked Potatoes (35 min)<br>2. Quick Potatoes (25 min)<br>3. Potato Bud Casserole (25-30 min) | 1. Gram'S Quick Casserole (85 min)<br>2. Quick Corned Beef... (45 min)<br>3. Quick Scalloped... (30-35 min) | **Obie metody zawodzą**.<br>FAISS dobrze rozumie pojęcie *,,quick''* (krótsze czasy). BM25 losowo szuka liczby 15. |
| **5. Negacja:**<br>*Potato dish without cheese or dairy* | 1. Cheese Potatoes (Low-Fat) (0.708)<br>2. No-Fat Cheese Potatoes (0.700)<br>3. Potato Cheese Patties (0.696) | 1. Cheesy Mashed Potato... (10.88)<br>2. Party Potatoes (10.64)<br>3. Potato Dish Casserole (10.54) | **Obie metody zawodzą**.<br>Brak obsługi logicznego wykluczenia (słowo *,,without''* zignorowane). |

### 3.2. Szczegółowe obserwacje i analiza błędów
Na podstawie powyższych testów zidentyfikowano kluczowe cechy operacyjne obu systemów:
* **Komponent semantyczny (FAISS):** Doskonale radzi sobie z mapowaniem ogólnego zamysłu użytkownika na odpowiedni profil smakowy lub geograficzny (Zapytania 1 i 3). Wykazuje jednak tendencję do "wygładzania" nietypowych szczegółów – obecność bardzo silnych powiązań wektorowych (np. ryba + ziemniaki) potrafi całkowicie przesłonić obecność mniejszych, ale równie ważnych słów kluczowych (np. marchewka).
* **Komponent leksykalny (BM25):** Gwarantuje stuprocentową obecność pożądanych słów kluczowych w nadrzędnych wynikach, co czyni go niezrównanym w zapytaniach recepturowych. Jest jednak całkowicie podatny na błędy kontekstowe, wynikające z obecności słów kluczowych w niespodziewanych miejscach dokumentu (np. w instrukcjach technicznych przygotowania potrawy, a nie liście składników).
* **Wspólne ograniczenia systemów NLP:** Testy wykluczenia (negacji) oraz ograniczeń numerycznych jednoznacznie dowodzą, że surowe systemy wyszukiwania tekstowego nie posiadają wewnętrznych struktur logiczno-matematycznych. Zarówno odcięcie stop-słów w BM25, jak i uśrednianie wektorów w FAISS, uniemożliwiają poprawną interpretację zwrotów takich jak "without" czy "< 15".

---

## 4. Kluczowe obserwacje
Żadna z przebadanych metod wyszukiwania nie jest w stanie samodzielnie sprostać przekrojowemu spektrum zapytań ludzkich. Wyszukiwarki wektorowe (FAISS) świetnie rozumieją ogólny koncept, ,,klimat'' zapytania oraz synonimy, co eliminuje problem braku dokładnego dopasowania słów. Niestety, są one powolniejsze w indeksacji i wysoce podatne na gubienie bardzo specyficznych intencji z długich opisów. Algorytm leksykalny (BM25) jest bezbłędny, błyskawiczny i tani obliczeniowo, gdy zależy nam na ścisłym wystąpieniu rzadkich składników, lecz jest całkowicie ślepy na kontekst gramatyczny i wieloznaczność językową. 

Z przeprowadzonych eksperymentów wynika jednoznacznie, że współczesny, w pełni niezawodny chatbot kulinarny musi opierać się na architekturze hybrydowej (Hybrid Search). Optymalnym rozwiązaniem jest przeprowadzanie równoległego wyszukiwania przy użyciu obu systemów, a następnie fuzja ich wyników za pomocą algorytmów takich jak Reciprocal Rank Fusion (RRF). Dodatkowo testy udowodniły, że obsługa atrybutów takich jak czas pieczenia czy alergeny nie powinna być rozwiązywana przez wyszukiwarki tekstowe, ale poprzez twarde filtry na poziomie metadanych bazy wektorowej (Metadata Filtering).

---

## 5. Bibliografia i Narzędzia

1. **FAISS (Facebook AI Similarity Search)** -- Biblioteka w języku C++/Python opracowana przez inżynierów Meta, służąca do niezwykle wydajnego wyszukiwania podobieństw i klastrowania gęstych wektorów.
2. **rank_bm25** -- Zbiór pythonowych implementacji algorytmów z rodziny BM25 (w tym użytego Okapi BM25), wykorzystany do budowy i przeszukiwania indeksu leksykalnego.
3. **Sentence-Transformers** -- Wiodący framework NLP bazujący na PyTorch, z którego zaczerpnięto model `all-MiniLM-L6-v2` do bezproblemowego generowania osadzeń tekstowych (embeddingów).
4. **NLTK (Natural Language Toolkit)** -- Klasyczna i wszechstronna platforma do przetwarzania języka naturalnego w Pythonie, użyta do tokenizacji, oczyszczania tekstu i usuwania angielskich stop-słów.
5. **Groq API & Llama 3 (Meta)** -- Zewnętrzny interfejs chmurowy oparty na dedykowanych układach LPU, wykorzystany jako silnik generacyjny (LLM) zintegrowany z modelem `llama-3.3-70b-versatile`.
6. **Hugging Face (Datasets)** -- Otwarta platforma i repozytorium zasobów sztucznej inteligencji, z której pobrano i odfiltrowano wyjściowy korpus danych *CookingRecipes*.