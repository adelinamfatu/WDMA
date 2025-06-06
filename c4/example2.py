import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# 1. Generăm date sintetice cu trei grupuri (blobs)
# - make_blobs creează 200 de puncte în jurul a 3 centre, cu deviație standard 0.80
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.80, random_state=42)
# - 'X' este un array shape (200, 2) cu coordonatele punctelor
# - '_' reprezintă etichete adevărate pe care nu le folosim aici

# 2. Aplicăm algoritmul DBSCAN
# - eps = 0.8 definește raza maximă pentru a considera două puncte vecini
# - min_samples = 5 înseamnă că, pentru a fi considerate parte dintr-un cluster dens,
# un punct trebuie să aibă cel puțin 5 vecini în interiorul razei eps
dbscan = DBSCAN(eps=0.8, min_samples=5)
labels = dbscan.fit_predict(X)
# - fit_predict:
# * antrenează DBSCAN pe datele X
# * returnează eticheta fiecărui punct:
# - o valoare >= 0 indică indexul clusterului
# - -1 indică un outlier (punct considerat prea izolat)

# 3. Identificăm outlierii și punctele care fac parte din clustere
outliers = X[labels == -1] # selectăm punctele cu etichetă -1 (outliers)
clusters = X[labels != -1] # selectăm punctele cu etichete >= 0 (aparțin clusterelor)

# 4. (Comentariu) Vizualizarea a fost eliminată la cerere; în mod normal, am fi colorat punctele
# folosind plt.scatter(X[:, 0], X[:, 1], c=labels)

# 5. Afișăm câteva puncte detectate ca outlieri
print("Number of outliers:", len(outliers))
print("Outlier examples:\n", outliers[:5])
# - len(outliers) arată câte puncte au fost etichetate drept -1
# - Afișăm primele 5 dintre acestea pentru a vedea coordonatele lor
