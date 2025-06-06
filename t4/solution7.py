import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 1. Generăm date sintetice „normale”
# fiecare punct are două caracteristici, distribuite normal în jurul valorii 50 ± 10
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=(200, 2))

# 2. Generăm date sintetice „anomale”
# - puncte care sunt mult mai departe de centrul distribuției normale
outliers = np.array([
    [100, 100],
    [10,  90],
    [90,  10],
    [120, 40],
    [40,  120]
])

# 3. Combinăm datele normale cu cele anomale
X = np.vstack((normal_data, outliers))

# 4. Aplicăm DBSCAN pentru detectarea anomaliilor
# - eps definește raza de vecinătate în unități ale caracteristicilor
# - min_samples este numărul minim de puncte pe care DBSCAN le cere pentru a considera o regiune „densă”
dbscan = DBSCAN(eps=5, min_samples=5)
labels = dbscan.fit_predict(X)

# 5. Identificăm anomaliile (label == -1)
outlier_points = X[labels == -1]
normal_points = X[labels != -1]

# 6. Vizualizare (opțională)
# - Punctele normale apar colorate după cluster; anomaliile apar cu marker roșu 'x'
plt.scatter(normal_points[:, 0], normal_points[:, 1], c=labels[labels != -1], cmap='viridis', s=50)
plt.scatter(outlier_points[:, 0], outlier_points[:, 1], marker='x', color='red', s=100, label='Anomalies')
plt.title("Anomaly Detection with DBSCAN")
plt.xlabel("Feature 1 (e.g. Purchase Amount)")
plt.ylabel("Feature 2 (e.g. Usage Rate)")
plt.legend()
plt.show()

# 7. Raportare
print("Număr total de puncte:", X.shape[0])
print("Număr de anomalii detectate:", len(outlier_points))
print("Puncte detectate ca anomalii:\n", outlier_points)
