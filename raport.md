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

### 3.1. Prezentacja systemu i obserwacje
Zbudowany system działał poprawnie w czasie rzeczywistym, sprawnie komunikując się z modelem Llama 3 poprzez API i zwracając zwięzłe rekomendacje poparte wyciągniętym kontekstem. Ewaluacja samego mechanizmu Retrieval została przeprowadzona na spreparowanym zbiorze zapytań testowych, z których każde miało na celu zbadanie innej luki technologicznej.

Zaobserwowano następujące rozbieżności między systemami:

* **Zapytania semantyczne i konceptualne:** W przypadku zapytań o charakterze opisowym (np. *,,hearty warm dish for cold weather''* lub *,,traditional Eastern European side dish''*), FAISS wykazywał bezapelacyjną przewagę. Potrafił powiązać abstrakcyjne pojęcia z zimowym klimatem potraw, ignorując brak dosłownego występowania tych słów. BM25 często zwracał wyniki przypadkowe (np. zakwas na chleb lub potrawy z curry, gdy słowo *Eastern* pojawiło się w tytule).
* **Wyszukiwanie konkretnych składników:** Kiedy zapytanie przyjmowało postać precyzyjnej listy wymogów (np. *,,what to cook with fish, potatoes and carrot''*), model wektorowy miał tendencję do gubienia specyficznych rzadszych składników (np. marchewki), uśredniając znaczenie zapytania. Algorytm BM25 był tu wysoce precyzyjny, windując na szczyt rankingu przepisy zawierające wszystkie wymagane ciągi znaków.
* **Pułapka negacji (Wykluczenie):** Podczas testowania intencji wykluczających (np. *,,potato dish without cheese or dairy''*) **obie metody poniosły porażkę**. FAISS skupił się na wektorach dla słów *cheese* oraz *dairy*, serwując głównie dania z nabiałem. BM25 całkowicie zignorował słowo *without* jako tzw. stop-word, nagradzając obecność słowa *cheese*. W takim przypadku system ratował wyłącznie LLM, który mógł przeanalizować narzucone wyniki z bazy i poinformować użytkownika o braku odpowiednich opcji.
* **Wartości numeryczne i czasowe:** Zapytanie z narzuconym limitem (*,,ready in under 15 minutes''*) obnażyło słabość NLP w dziedzinie czystej matematyki. Żaden z silników nie zrozumiał twardej relacji operacyjnej ($< 15$). Należy tu jednak docenić model FAISS -- choć zignorował on samą liczbę, doskonale uchwycił semantyczną koncepcję pojęcia *,,quick''*, zwracając przepisy, które faktycznie należały do grupy relatywnie szybkich w przygotowaniu (zazwyczaj 25--35 minut). Z kolei algorytm BM25 poszukiwał znaku ,,15'' całkowicie na ślepo, znajdując go w błędnych kontekstach (np. *,,bake 15 minutes longer''*), co skutkowało rekomendowaniem bardzo czasochłonnych dań, wymagających nawet 85 minut pieczenia.

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