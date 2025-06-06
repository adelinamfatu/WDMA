import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Încărcăm și scalăm setul de date Iris (simulând un df_scaled)
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# - X_scaled: array shape (150, 4), fiecare caracteristică cu media=0, std=1

# 2. Potrivim fiecare metodă de clusterizare
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)

dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)

agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)

# 3. Extragem etichetele de cluster pentru fiecare metodă
labels_kmeans = kmeans.labels_

labels_dbscan = dbscan.labels_

labels_agg = agg.labels_

# 4. Calculăm scorurile Silhouette (numai dacă există cel puțin două clustere valide)
# - silhouette_score necesită cel puțin 2 clustere distincte și fără etichete unice
def compute_silhouette(X, labels):
    unique_labels = set(labels)
    # Excludem outlierii DBSCAN (-1) când numărăm clusterele
    n_clusters = len(unique_labels - {-1})
    # Dacă mai puțin de 2 clustere, Silhouette nu se poate calcula
    if n_clusters < 2:
        return None
    # Excludem punctele etichetate cu -1 înainte de calcul (dacă există outlieri)
    mask = labels != -1
    return silhouette_score(X[mask], labels[mask])

sil_kmeans = compute_silhouette(X_scaled, labels_kmeans)

sil_dbscan = compute_silhouette(X_scaled, labels_dbscan)

sil_agg = compute_silhouette(X_scaled, labels_agg)

# 5. Afișăm scorurile
print("Silhouette Score KMeans:  ", sil_kmeans if sil_kmeans is not None else "Nu se poate calcula")
print("Silhouette Score DBSCAN:  ", sil_dbscan if sil_dbscan is not None else "Nu se poate calcula")
print("Silhouette Score Agglom: ", sil_agg if sil_agg is not None else "Nu se poate calcula")
