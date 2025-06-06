import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    rand_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score
)

# 1. Generăm date sintetice cu etichete adevărate
# make_blobs creează 150 de puncte în 2D, grupate în jurul a 3 centre
X, y_true = make_blobs(
    n_samples=150, # numărul total de eșantioane
    centers=3, # numărul de clustere reale
    n_features=2, # dimensiunea spațiului (2D)
    random_state=42 # pentru reproducibilitate
)

# 2. Clusterizare cu K-Means
# Definim modelul KMeans pentru 3 clustere
kmeans = KMeans(
    n_clusters=3, # numărul de clustere pe care vrem să le identificăm
    random_state=42 # inițializare stabilă a centroidelor
)
y_pred = kmeans.fit_predict(X)
# fit_predict:
# - se potrivește modelul pe datele X
# - returnează eticheta (0,1,2) a clusterului pentru fiecare punct

# 3. Metrici interne
# Aceste metrici nu necesită etichete adevărate (y_true)
# - Silhouette Score: măsoară coeziunea și separarea clusterelor (-1 la 1, mai mare e mai bun)
sil = silhouette_score(X, y_pred)

# - Davies-Bouldin Index: valoare mică indică clustere bine separate
db  = davies_bouldin_score(X, y_pred)

# - Calinski-Harabasz Index: valoare mare indică clustere dense și bine separate
ch  = calinski_harabasz_score(X, y_pred)

# 4. Metrici externe
# Aceste metrici compară etichetele prezise (y_pred) cu etichetele adevărate (y_true)

# Rand Index: proporția perechilor de puncte care sunt etichetate coerent (same/different) între y_true și y_pred
rand_idx = rand_score(y_true, y_pred)

# Adjusted Rand Index: similar cu Rand, dar corectat pentru șansă (valoare între -1 și 1, mai mare e mai bun)
adj_rand_idx = adjusted_rand_score(y_true, y_pred)

# Purity: suma maximelor din fiecare coloană a matricei de confuzie, împărțită la numărul total de puncte
def purity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # cm este o matrice (n_true_labels, n_pred_labels), unde fiecare element [i,j] este numărul de puncte
    # cu eticheta adevărată i și eticheta prezisă j
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)

purity = purity_score(y_true, y_pred)

# F1-score (macro): măsoară balanța între precizie și recall, calculată pentru fiecare etichetă și apoi mediată
# - average='macro' tratează fiecare cluster/etichetă egal
f_measure = f1_score(y_true, y_pred, average='macro')

# 5. Afișăm rezultatele
print("=== Internal Metrics (fără etichete adevărate) ===")
print(f"Silhouette Score:  {sil:.3f}  (interval: -1 la 1, mai mare e mai bun)")
print(f"Davies-Bouldin Index: {db:.3f}  (mai mic e mai bun)")
print(f"Calinski-Harabasz Index: {ch:.3f}  (mai mare e mai bun)")

print("\n=== External Metrics (compară y_pred cu y_true) ===")
print(f"Rand Index: {rand_idx:.3f}  (interval: 0 la 1, mai mare e mai bun)")
print(f"Adjusted Rand Index: {adj_rand_idx:.3f} (interval: -1 la 1, mai mare e mai bun)")
print(f"Purity: {purity:.3f}  (interval: 0 la 1, mai mare e mai bun)")
print(f"F-Measure (F1, macro): {f_measure:.3f} (interval: 0 la 1, mai mare e mai bun)")
