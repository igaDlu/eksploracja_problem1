# Opis działania kodu solution.py

## Ogólny cel
Kod implementuje system rekomendacji filmów oparty na głębokiej sieci neuronowej (Deep Learning) przy użyciu PyTorch. Przetwarza duże zbiory danych ocen użytkowników bez ładowania wszystkiego do pamięci RAM, trenuje model na podstawie interakcji użytkownik-film i pozwala na predykcję ocen dla nowych par.

## Kluczowe komponenty

### 1. Konfiguracja (CONFIG)
- Ścieżki do plików CSV z filmami i ocenami.
- Parametry treningu: duży batch_size (4096) dla optymalizacji GPU RTX 9070XT, embedding_dim=64 dla lepszej reprezentacji, epochs=2 dla demonstracji.
- Device: automatyczne wykrycie CUDA jeśli dostępne.

### 2. Funkcja create_mappings()
- Tworzy mapowania oryginalnych ID użytkowników/filmów na indeksy (0-based) używane w modelu.
- Używa streamingowego czytania plików w chunkach, aby uniknąć out-of-memory na dużych danych (np. MovieLens).
- Zapisuje mapowania do plików pickle dla późniejszego użycia.

### 3. Klasa RatingsDataset (IterableDataset)
- Strumieniowo czyta dane ocen w chunkach, mapuje ID na indeksy, normalizuje oceny do [0,1] (lepsza stabilność treningu).
- Dzieli dane na train/val w proporcjach 90/10 na poziomie chunków (deterministycznie).
- Obsługuje wielowątkowość DataLoader (num_workers=4) dla wydajności.

### 4. Klasa EnhancedRecommender (nn.Module)
- Model hybrydowy: embeddingi użytkowników/filmów + biasy + głęboka sieć MLP.
- Embeddingi: 64-wymiarowe dla użytkowników i filmów.
- Sieć: Linear(192,128) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(128,64) -> ReLU -> Dropout(0.2) -> Linear(64,1) -> Sigmoid.
- Forward: konkatenacja [u, m, u*m] -> MLP -> + biasy -> sigmoid (output w [0,1]).
- Sensowność: BatchNorm stabilizuje trening, Dropout zapobiega overfittingowi, interakcja u*m modeluje relacje.

### 5. Funkcja train()
- Tworzy mapowania, datasety, DataLoadery.
- Model: EnhancedRecommender, loss: MSE, optimizer: Adam, scheduler: ReduceLROnPlateau (adaptacyjny learning rate dla stabilności).
- Pętla epok: trening z tqdm, walidacja, obliczenie MSE/RMSE, krok scheduler'a.
- Zapisuje model.state_dict() i embeddingi do plików.

### 6. Funkcja predict()
- Ładuje mapowania i model jeśli nie podane.
- Dla danej pary user_id, movie_id: mapuje na indeksy, inferencja modelu, denormalizacja do skali 1-5.
- Zwraca przewidzianą ocenę.

## Jak uruchomić
1. Przygotuj pliki CSV: data/movie.csv (kolumny: movieId), data/rating.csv (kolumny: userId, movieId, rating).
2. Uruchom `python solution.py` - trenuje model i zapisuje pliki.
3. Użyj `predict(user_id, movie_id)` dla predykcji.

## Sensowność rozwiązań
- **Streaming danych**: Dla dużych zbiorów (miliony ocen) unika RAM overload.
- **Normalizacja ocen**: [0,1] lepsza dla sigmoid/MSE niż 1-5.
- **Duży batch_size**: Optymalizuje wykorzystanie GPU.
- **Embeddingi + MLP**: Łączy collaborative filtering z głębokim uczeniem dla lepszych rekomendacji.
- **Regularyzacja**: Dropout/BatchNorm zapobiega overfittingowi.
- **Scheduler LR**: Automatycznie zmniejsza LR gdy walidacja się zatrzymuje, przyspiesza konwergencję.
- **Wielowątkowość**: Przyspiesza ładowanie danych.

## Wymagania
- PyTorch, pandas, numpy, tqdm.
- GPU zalecane dla wydajności.