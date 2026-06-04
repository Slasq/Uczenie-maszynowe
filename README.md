# Uczenie maszynowe — Prace laboratoryjne

Prace laboratoryjne z przedmiotu uczenia maszynowego, wykonane w Google Colab.

---

## Zawartość

### Lab 1 — Wprowadzenie do klasteryzacji
`Labs reports/Lab 1 Wprowadzenie.ipynb`

Implementacja i porównanie algorytmów grupowania bez nadzoru:
- **K-Means** — klasteryzacja danych syntetycznych, dobór liczby klastrów metodą łokciową
- **DBSCAN** — klasteryzacja gęstościowa, obsługa punktów szumu
- Preprocessing: standaryzacja (StandardScaler) + redukcja wymiarów (PCA, 95% wariancji)
- Ocena jakości: Silhouette Score (K-Means po PCA przy k=6: ~0.42, DBSCAN po PCA: ~0.42)

---

### Lab 2 — Klasyfikacja obrazów BloodMNIST
`Labs reports/Lab 2 BloodMNIST.ipynb`

Klasyfikacja 8 klas komórek krwi ze zbioru MedMNIST v2 (obrazy 28×28 px) przy użyciu Keras:

| Model | Parametry | Test Accuracy |
|---|---|---|
| FC v1 (MLP baseline) | ~590 k | 57.56% |
| FC v2 (głębszy MLP) | ~1.1 M | 75.45% |
| CNN v1 (2 bloki) | ~420 k | 87.66% |
| CNN v2 (3 bloki) | ~290 k | **86.38%** |
| CNN v2 + augmentacja | ~290 k | 86.73% |

Zastosowane techniki: BatchNormalization, Dropout, GlobalAveragePooling2D, augmentacja danych (RandomFlip, RandomRotation, RandomZoom, RandomContrast).

---

### Lab 3 — Generowanie imion modelem bigramowym
`Labs reports/Lab 3 BigRams.ipynb`

Językowy model bigramowy oparty na bazie polskich imion, zaimplementowany w PyTorch:
- Macierz bigramów N_tab z częstościami wystąpień par znaków
- Generowanie imion metodą `torch.multinomial`
- Funkcja straty NLL (negative log-likelihood) — porównanie imion wygenerowanych z bazą
- Prosta sieć neuronowa (1 warstwa liniowa + softmax) jako odpowiednik macierzy bigramów
- Uczenie na pojedynczym imieniu i na pełnej bazie polskich imion
- Analiza: 10 imion angielskich o najniższym i najwyższym NLL względem polskiej bazy

---

### Lab 4 — Q-Learning z Gymnasium
`Labs reports/Lab 4 Gym.ipynb`

Implementacja algorytmu Q-learning na środowiskach dyskretnych z biblioteki `gymnasium`:

| Środowisko | Polityka losowa | Q-table |
|---|---|---|
| Blackjack-v1 | ~29.5% wygranych | — |
| Taxi-v4 | ~0% sukcesów | ~40% sukcesów |
| FrozenLake-v1 (`is_slippery=True`) | ~1% sukcesów | **~76% sukcesów** |

Parametry Q-learningu (FrozenLake): α=0.1, γ=0.99, ε: 1.0→0.01 (decay wykładniczy), 20 000 epizodów.

---

## Technologie

- Python 3 · Google Colab
- PyTorch
- Keras 3 / TensorFlow
- scikit-learn
- gymnasium · medmnist
- matplotlib · seaborn · numpy
